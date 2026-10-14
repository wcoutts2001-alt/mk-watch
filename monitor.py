#!/usr/bin/env python3
"""
mk-watch: pushes a phone notification (ntfy) when the Czech Ministry of Culture makes an
important announcement, wherever it appears (website, Facebook, Instagram, X, news).

Run every ~15 minutes. Each run:
  1. fetches every source in config.yaml
  2. keeps only items not seen before (state.json)
  3. asks Claude which are major announcements (keyword filter as fallback)
  4. pushes important items instantly, queues the rest for a daily digest
  5. alerts you if a source is failing or has gone suspiciously quiet

Usage:
  python monitor.py            normal run
  python monitor.py --dry-run  print instead of sending, don't save state
  python monitor.py --test     send a test notification and exit
  python monitor.py --backfill-hours 48   also report significant items from the last 48 h
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin
from zoneinfo import ZoneInfo

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
STATE_PATH = ROOT / "state.json"
SEEN_LIMIT = 500  # remembered IDs per source
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "cs,en;q=0.8",
}
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5")


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.isoformat(timespec="seconds")


def parse_iso(s):
    return datetime.fromisoformat(s) if s else None


def item_id(source_id, link, title):
    key = (link or "").strip() or (title or "").strip()
    return source_id + ":" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Fetchers — each returns a list of {"title", "link", "text", "category"}
# --------------------------------------------------------------------------- #

def fetch_mk_website(src, cfg):
    r = requests.get(src["url"], headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    items, seen_links = [], set()
    for a in soup.find_all("a", href=True):
        href = urljoin(src["url"], a["href"])
        if not re.search(r"/novinky-a-media-cs-4/\d+cs-", href) or href in seen_links:
            continue
        seen_links.add(href)
        text = " ".join(a.get_text(" ", strip=True).split())
        # Listing text looks like: "<title> <dd.mm.yyyy hh:mm>|<category><teaser>"
        m = re.search(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}", text)
        title = text[: m.start()].strip() if m else text[:200]
        rest = text[m.end():].lstrip("| ").strip() if m else ""
        published = None
        if m:
            published = datetime.strptime(m.group(0), "%d.%m.%Y %H:%M").replace(
                tzinfo=ZoneInfo("Europe/Prague")).astimezone(timezone.utc)
        category = ""
        for cat in ("příspěvkové organizace", "ministerstvo", "avíza"):
            if rest.lower().startswith(cat):
                category, rest = cat, rest[len(cat):].strip()
                break
        items.append({"title": title or href, "link": href, "text": rest, "category": category,
                      "published": published})
    if not items:
        raise RuntimeError("no news links found — page layout may have changed")
    return items


def _parse_feed(content, src_name):
    feed = feedparser.parse(content)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"{src_name}: could not parse feed ({feed.bozo_exception})")
    items = []
    for e in feed.entries:
        summary = BeautifulSoup(e.get("summary", ""), "html.parser").get_text(" ", strip=True)
        t = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime(*t[:6], tzinfo=timezone.utc) if t else None
        items.append({
            "published": published,
            "title": e.get("title", "").strip() or summary[:120],
            "link": e.get("link", ""),
            "text": summary[:1500],
            "category": "",
        })
    return items


def fetch_rss(src, cfg):
    r = requests.get(src["url"], headers=HEADERS, timeout=30)
    r.raise_for_status()
    return _parse_feed(r.content, src["name"])


def fetch_rss_bridge(src, cfg):
    params = dict(src["params"])
    params.update({"action": "display", "format": "Atom"})
    url = cfg["rss_bridge_url"].rstrip("/") + "/?" + urlencode(params)
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        snippet = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)[:200]
        raise RuntimeError(f"RSS-Bridge HTTP {r.status_code}: {snippet}")
    return _parse_feed(r.content, src["name"])


FETCHERS = {"mk_website": fetch_mk_website, "rss": fetch_rss, "rss_bridge": fetch_rss_bridge}


# --------------------------------------------------------------------------- #
# Importance filter
# --------------------------------------------------------------------------- #

CLASSIFY_PROMPT = """You monitor the Czech Ministry of Culture (Ministerstvo kultury ČR) for a journalist in the culture desk of Czech Radio.

For each item below decide whether it is a MAJOR ANNOUNCEMENT a culture journalist must know about promptly. Major: legislation and amendments, policy decisions, budget or grant/funding decisions and calls, appointments, dismissals or resignations (ministry leadership, heads of national institutions), selection procedures for top posts, official ministry positions/statements, anything about public-service media (Czech Radio, Czech TV) or licence fees, major controversies, crisis responses.
NOT major: routine events at museums or ministry-funded institutions, festivals, exhibitions, commemorations, awards/medals, ceremonial visits, generic news that merely mentions the ministry in passing, news about other countries' ministries (e.g. Slovakia).

Return ONLY a JSON array, one object per item, in the same order:
[{{"id": "<id>", "important": true|false, "summary": "<one sentence in {language}, what was announced>"}}]

Items:
{items}"""


def classify_with_claude(items, cfg):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    payload_items = "\n\n".join(
        f'id: {it["id"]}\nsource: {it["source_name"]}\ncategory: {it.get("category") or "-"}\n'
        f'title: {it["title"]}\ntext: {it["text"][:800]}'
        for it in items
    )
    prompt = CLASSIFY_PROMPT.format(language=cfg.get("summary_language", "English"), items=payload_items)
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": CLAUDE_MODEL, "max_tokens": 4000, "messages": [{"role": "user", "content": prompt}]},
        timeout=90,
    )
    r.raise_for_status()
    text = "".join(b.get("text", "") for b in r.json().get("content", []))
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise RuntimeError("Claude returned no JSON array")
    return {d["id"]: d for d in json.loads(m.group(0))}


def classify_with_keywords(items, cfg):
    kws = [k.lower() for k in cfg.get("important_keywords", [])]
    out = {}
    for it in items:
        blob = f'{it["title"]} {it["text"]}'.lower()
        routine = it.get("category") == "příspěvkové organizace"
        out[it["id"]] = {"important": (not routine) and any(k in blob for k in kws), "summary": ""}
    return out


def classify(items, cfg, notes):
    if not items:
        return {}
    try:
        result = classify_with_claude(items, cfg)
        if result is not None:
            missing = [it for it in items if it["id"] not in result]
            if missing:
                result.update(classify_with_keywords(missing, cfg))
            return result
    except Exception as e:  # never lose items because the filter failed
        notes.append(f"Claude filter failed ({e}); used keyword filter instead.")
    return classify_with_keywords(items, cfg)


# --------------------------------------------------------------------------- #
# Notifications (ntfy)
# --------------------------------------------------------------------------- #

def notify(title, message, click=None, priority=3, tags=None, dry_run=False):
    """Push a notification to the ntfy app. priority: 1 (min) … 5 (urgent)."""
    if dry_run:
        print(f"---- NTFY [{priority}] {title} ----\n{message}\n(click: {click})\n----------")
        return
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        raise RuntimeError("NTFY_TOPIC must be set")
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    payload = {"topic": topic, "title": title, "message": message[:3900], "priority": priority}
    if click:
        payload["click"] = click
    if tags:
        payload["tags"] = tags
    r = requests.post(server, json=payload, timeout=30)
    r.raise_for_status()


def fmt_item(it, verdict, with_link=False):
    summary = verdict.get("summary") or ""
    lines = [it["title"]]
    if summary:
        lines.append(summary)
    lines.append(f'({it["source_name"]})')
    if with_link:
        lines.append(it["link"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"sources": {}, "digest_queue": [], "last_digest_date": None}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def check_health(src, st, now, alerts):
    """Alert once when a source breaks or goes quiet; alert again when it recovers."""
    problem = None
    first_err = parse_iso(st.get("first_error_at"))
    if first_err and now - first_err > timedelta(hours=src.get("max_error_hours", 6)):
        problem = f'failing since {first_err:%d.%m. %H:%M} UTC: {st.get("last_error", "")[:200]}'
    else:
        last_new = parse_iso(st.get("last_new_item_at"))
        if last_new and now - last_new > timedelta(days=src.get("max_silence_days", 7)):
            problem = f"no new posts for {(now - last_new).days} days — may be broken, check manually"

    if problem and not st.get("alerted"):
        alerts.append(f'⚠️ {src["name"]}: {problem}')
        st["alerted"] = True
    elif not problem and st.get("alerted"):
        alerts.append(f'✅ {src["name"]} is working again.')
        st["alerted"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--backfill-hours", type=float, default=0,
                    help="test mode: also report significant items published in the last N hours")
    args = ap.parse_args()

    if args.test:
        notify("mk-watch connected", "You'll get Ministry of Culture announcements here.", tags=["white_check_mark"])
        print("Test message sent.")
        return

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    state = load_state()
    now = now_utc()
    tz = ZoneInfo(cfg.get("timezone", "Europe/Prague"))
    alerts, notes, new_items, backfill = [], [], [], []
    since = now - timedelta(hours=args.backfill_hours) if args.backfill_hours > 0 else None

    for src in cfg["sources"]:
        st = state["sources"].setdefault(src["id"], {"seen": []})
        first_run = "initialized" not in st
        try:
            items = FETCHERS[src["type"]](src, cfg)
            st.pop("first_error_at", None)
            st.pop("last_error", None)
            st["last_success_at"] = iso(now)
        except Exception as e:
            st.setdefault("first_error_at", iso(now))
            st["last_error"] = str(e)
            print(f"[{src['id']}] ERROR: {e}", file=sys.stderr)
            check_health(src, st, now, alerts)
            continue

        if since:
            for it in items:
                if it.get("published") and it["published"] >= since:
                    backfill.append(dict(it, id=item_id(src["id"], it["link"], it["title"]),
                                         source_id=src["id"], source_name=src["name"]))

        seen = set(st["seen"])
        fresh = []
        for it in items:
            iid = item_id(src["id"], it["link"], it["title"])
            if iid not in seen:
                it.update(id=iid, source_id=src["id"], source_name=src["name"])
                fresh.append(it)
                st["seen"].append(iid)
        st["seen"] = st["seen"][-SEEN_LIMIT:]

        if first_run:
            # Baseline: remember what exists today without flooding you.
            st["initialized"] = iso(now)
            st["last_new_item_at"] = iso(now)
            notes.append(f'{src["name"]}: baseline set ({len(items)} existing items).')
        else:
            if fresh:
                st["last_new_item_at"] = iso(now)
            new_items.extend(fresh)
        print(f"[{src['id']}] {len(items)} items, {0 if first_run else len(fresh)} new")
        check_health(src, st, now, alerts)

    # Collapse the same story arriving via several channels (same title).
    unique, titles = [], set()
    ids_seen = set()
    for it in new_items + backfill:
        if it["id"] in ids_seen:
            continue
        ids_seen.add(it["id"])
        norm = re.sub(r"\W+", " ", it["title"].lower()).strip()[:80]
        if norm in titles:
            continue
        titles.add(norm)
        unique.append(it)

    verdicts = classify(unique, cfg, notes)
    important = [it for it in unique if verdicts.get(it["id"], {}).get("important")]
    routine = [it for it in unique if not verdicts.get(it["id"], {}).get("important")]

    for it in important:
        notify("Ministerstvo kultury", fmt_item(it, verdicts[it["id"]]), click=it["link"],
               priority=4, tags=["bell"], dry_run=args.dry_run)

    if since:
        notify(f"Test run: last {args.backfill_hours:g} hours",
               f"Checked {len(unique)} items from all sources; {len(important)} significant "
               f"(sent as separate notifications). The rest go to the evening digest.",
               priority=3, tags=["test_tube"], dry_run=args.dry_run)

    for it in routine:
        state["digest_queue"].append({"text": fmt_item(it, verdicts.get(it["id"], {}), with_link=True)})

    if alerts:
        notify("mk-watch: source status", "\n\n".join(alerts), priority=4, tags=["warning"], dry_run=args.dry_run)

    local = now.astimezone(tz)
    today = local.date().isoformat()
    if local.hour >= cfg.get("digest_hour", 18) and state.get("last_digest_date") != today:
        queue = state["digest_queue"]
        if queue:
            body = "\n\n".join(q["text"] for q in queue[:15])
            more = f"\n\n…and {len(queue) - 15} more." if len(queue) > 15 else ""
            notify(f"Daily digest: {len(queue)} routine items", body + more, priority=2,
                   tags=["newspaper"], dry_run=args.dry_run)
        state["digest_queue"] = []
        state["last_digest_date"] = today

    for n in notes:
        print("note:", n)
    if not args.dry_run:
        save_state(state)


if __name__ == "__main__":
    main()
