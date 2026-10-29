"""Donnees gratuites, sans inscription : Yahoo Finance (bougies), Coinbase / Bitstamp (carnets d'ordres crypto).
Prix de reference neutres (aucun courtier) : les niveaux sont aussi donnes en % pour s'appliquer partout."""
import logging
import time

import numpy as np
import requests

log = logging.getLogger("lab")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) strategy-lab/1.0", "Accept": "*/*"})
HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")
TF = {"15M": ("15m", "60d", 900), "1H": ("60m", "730d", 3600)}


def ysym(sym, kind):
    if kind == "cr":
        return sym + "-USD"
    if kind == "fx":
        return {"XAUUSD": "GC=F", "XAGUSD": "SI=F"}.get(sym, sym + "=X")
    return sym.replace(".", "-")


def label(sym, kind):
    return {"XAUUSD": "XAUUSD (or)", "XAGUSD": "XAGUSD (argent)"}.get(sym, sym)


def parse(res, step):
    t = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0] or {}
    cols = [q.get(k) or [] for k in ("open", "high", "low", "close", "volume")]
    rows = {}
    for i, ts in enumerate(t):
        try:
            o, h, l, c = (float(cols[j][i]) for j in range(4))
        except (TypeError, IndexError, ValueError):
            continue
        if min(o, h, l, c) <= 0 or int(ts) % step:
            continue                                # bougie en cours ou prix invalide
        v = cols[4][i] if i < len(cols[4]) and cols[4][i] is not None else 0.0
        rows[int(ts)] = (int(ts), o, max(h, o, c), min(l, o, c), c, float(v))
    return np.array([rows[k] for k in sorted(rows)], dtype=float).reshape(-1, 6)


def bars(sym, kind, tf, rng=None):
    """Tableau (n, 6) : temps, ouverture, haut, bas, cloture, volume. Leve une erreur si indisponible."""
    interval, rng0, step = TF[tf]
    rng = rng or rng0
    last = None
    for _ in range(2):
        for host in HOSTS:
            try:
                r = SESSION.get("%s/v8/finance/chart/%s" % (host, ysym(sym, kind)),
                                params={"range": rng, "interval": interval}, timeout=20)
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                res = ((r.json().get("chart") or {}).get("result") or [None])[0]
                if res:
                    return parse(res, step)
                last = "pas de donnees"
            except Exception as ex:
                last = str(ex)[:100]
        time.sleep(2)
    raise RuntimeError("Yahoo %s : %s" % (sym, last))


def order_book(sym):
    """Desequilibre du carnet d'ordres crypto (Coinbase + Bitstamp, API publiques sans compte), a +/- 0,5 %.
       Retourne (desequilibre entre -1 et +1, profondeur en $) ou None."""
    bids = asks = 0.0
    mid = None
    srcs = [("https://api.exchange.coinbase.com/products/%s-USD/book" % sym, {"level": 2}, "cb"),
            ("https://www.bitstamp.net/api/v2/order_book/%susd/" % sym.lower(), {}, "bs")]
    for url, params, _src in srcs:
        try:
            j = SESSION.get(url, params=params, timeout=10).json()
            b = [(float(x[0]), float(x[1])) for x in (j.get("bids") or [])[:400]]
            a = [(float(x[0]), float(x[1])) for x in (j.get("asks") or [])[:400]]
            if not b or not a:
                continue
            m = (b[0][0] + a[0][0]) / 2
            mid = mid or m
            bids += sum(p * q for p, q in b if p >= m * 0.995)
            asks += sum(p * q for p, q in a if p <= m * 1.005)
        except Exception:
            continue
    if not mid or bids + asks <= 0:
        return None
    return (bids - asks) / (bids + asks), bids + asks


def daily(sym):
    """Cloture journaliere (1 an) pour le regime de marche (SPY, VIX, DXY)."""
    r = None
    for host in HOSTS:
        try:
            r = SESSION.get("%s/v8/finance/chart/%s" % (host, sym), params={"range": "1y", "interval": "1d"}, timeout=20)
            res = ((r.json().get("chart") or {}).get("result") or [None])[0]
            if res:
                return parse(res, 1)
        except Exception:
            continue
    raise RuntimeError("Yahoo %s indisponible" % sym)
