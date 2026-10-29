"""Envoi Telegram (meme bot et meme conversation que Bot Pro : destination chiffree avec le jeton, jamais en clair).
Si la destination n'est pas connue, envoyer le code d'appairage au bot suffit (voir README)."""
import hashlib
import html
import logging
import os
import re
import time

from .data import SESSION

log = logging.getLogger("lab")
PAIR_HASH = "421400dea6994bedd848d60bc1b10ef0794b39c306ad35ee34487422a8b35fe1"
CHAT = {"id": None, "offset": 0, "ts": 0}
LAST = [0.0]
STATS = {"envoyes": 0, "erreurs": 0}


def esc(t):
    return html.escape(str(t), quote=False)


def _key():
    return int(hashlib.sha256(("pair|" + (os.environ.get("TELEGRAM_BOT_TOKEN") or "")).encode()).hexdigest()[:16], 16)


def pair_encode(cid):
    n = int(cid)
    return ("-" if n < 0 else "") + "%x" % (abs(n) ^ _key())


def pair_decode(v):
    try:
        neg = str(v).startswith("-")
        n = int(str(v).lstrip("-"), 16) ^ _key()
        return str(-n if neg else n)
    except Exception:
        return None


def setup(st):
    """Destination : TELEGRAM_CHAT_ID numerique, sinon l'appairage chiffre garde dans l'etat."""
    raw = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip().strip("\"'")
    m = re.search(r"-?\d{5,}", raw)
    if m:
        CHAT["id"] = m.group(0)
    elif st.get("tg_pair"):
        CHAT["id"] = pair_decode(st["tg_pair"])
    return CHAT["id"]


def pair_check(st):
    """Cherche le code d'appairage (PRO-XXXXXXXX) dans les messages recus par le bot, au plus une fois par minute."""
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    if CHAT["id"] or not tok or time.time() - CHAT["ts"] < 60:
        return False
    CHAT["ts"] = time.time()
    try:
        up = SESSION.get("https://api.telegram.org/bot%s/getUpdates" % tok,
                         params={"limit": 100, "offset": CHAT["offset"]}, timeout=15).json()
    except Exception:
        return False
    for u in up.get("result") or []:
        CHAT["offset"] = max(CHAT["offset"], int(u.get("update_id", 0)) + 1)
        msg = u.get("message") or u.get("channel_post") or {}
        for w in re.findall(r"PRO-[A-Z0-9]{8}", str(msg.get("text") or "").upper()):
            if hashlib.sha256(w.encode()).hexdigest() == PAIR_HASH and (msg.get("chat") or {}).get("id"):
                CHAT["id"] = str(msg["chat"]["id"])
                st["tg_pair"] = pair_encode(CHAT["id"])
                return True
    return False


def strip(t):
    return html.unescape(re.sub(r"</?(b|i|code)>", "", t))


def send(msg):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), CHAT["id"]
    log.info("\n" + re.sub(r"📦[^\n]*", "📦 (masqué)", strip(msg)))
    if not tok or not chat:
        return False
    if len(msg) > 4000:
        msg = msg[:3990].rsplit("\n", 1)[0] + "\n…"
    body = {"chat_id": chat, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
    for essai in range(4):
        time.sleep(max(0.0, LAST[0] + 1.1 - time.time()))
        LAST[0] = time.time()
        try:
            r = SESSION.post("https://api.telegram.org/bot%s/sendMessage" % tok, json=body, timeout=10)
            if r.ok:
                STATS["envoyes"] += 1
                return True
            if r.status_code == 429:
                time.sleep(min(60, float((r.json().get("parameters") or {}).get("retry_after") or 5) + 1))
                continue
            if r.status_code == 400 and "parse_mode" in body:
                body.pop("parse_mode")
                body["text"] = strip(msg)
                continue
            STATS["erreurs"] += 1
            STATS["derniere_erreur"] = "HTTP %d %s" % (r.status_code, str(r.json().get("description", ""))[:80])
            return False
        except Exception as ex:
            STATS["derniere_erreur"] = str(ex).replace(tok, "***")[:80]
        time.sleep(2 + 2 * essai)
    return False
