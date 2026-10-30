"""Donnees gratuites, sans inscription : Yahoo Finance (bougies), Coinbase / Bitstamp (carnets d'ordres crypto).
Prix de reference neutres (aucun courtier) : les niveaux sont aussi donnes en % pour s'appliquer partout."""
import concurrent.futures as cf
import datetime
import logging
import lzma
import struct
import time

import os

import numpy as np
import requests

log = logging.getLogger("lab")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) strategy-lab/1.0", "Accept": "*/*"})
HOSTS = ("https://query2.finance.yahoo.com", "https://query1.finance.yahoo.com")
TF = {"5M": ("5m", "60d", 300), "15M": ("15m", "60d", 900), "30M": ("30m", "60d", 1800), "1H": ("60m", "730d", 3600)}
# historique long de la recherche (jours) : meme nombre de bougies environ pour chaque unite de temps
FX_DAYS = {"5M": ("FX_DAYS_5M", 90), "15M": ("FX_DAYS", 270), "30M": ("FX_DAYS_30M", 540)}
CR_DAYS = {"5M": ("CR_DAYS_5M", 80), "15M": ("CR_DAYS_15M", 240), "30M": ("CR_DAYS_30M", 480), "1H": ("CR_DAYS_1H", 1095)}


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


# ------------------------------------------------------------------ historiques longs (gratuits, sans inscription)
def _resample(rows, step):
    """Bougies 1 min (ou plus courtes) -> bougies de `step` secondes."""
    if not len(rows):
        return np.zeros((0, 6))
    rows = rows[np.argsort(rows[:, 0])]
    k = (rows[:, 0] // step) * step
    out = []
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    for s_, e_ in zip(starts, ends):
        g = rows[s_:e_]
        out.append((k[s_], g[0, 1], g[:, 2].max(), g[:, 3].min(), g[-1, 4], g[:, 5].sum()))
    return np.array(out, dtype=float)


DUKA_POINT = {"JPY": 1e-3, "XAU": 1e-3, "XAG": 1e-3}


def _duka_day(sym, day):
    url = "https://datafeed.dukascopy.com/datafeed/%s/%04d/%02d/%02d/BID_candles_min_1.bi5" % (
        sym, day.year, day.month - 1, day.day)
    try:
        r = SESSION.get(url, timeout=20)
        if r.status_code != 200 or not r.content:
            return None
        raw = lzma.decompress(r.content, format=lzma.FORMAT_ALONE)
    except Exception:
        return None
    pt = DUKA_POINT.get(sym[3:], DUKA_POINT.get(sym[:3], 1e-5))
    t0 = int(datetime.datetime(day.year, day.month, day.day, tzinfo=datetime.timezone.utc).timestamp())
    rows = []
    for off in range(0, len(raw) - 23, 24):
        t, o, c, lo, hi, v = struct.unpack(">5if", raw[off:off + 24])
        rows.append((t0 + t, o * pt, hi * pt, lo * pt, c * pt, v))
    return rows


def dukascopy(sym, step, days):
    """Forex / metaux : bougies 1 min Dukascopy (flux public), regroupees en bougies de `step` secondes."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    dates = [today - datetime.timedelta(days=i) for i in range(1, days + 1)]
    dates = [d for d in dates if d.weekday() != 5]               # samedi : marche ferme
    rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda d: _duka_day(sym, d), dates):
            if res:
                rows.extend(res)
    if len(rows) < 1000:
        raise RuntimeError("Dukascopy %s : historique indisponible" % sym)
    a = _resample(np.array(rows, dtype=float), step)
    a[:, 5] = 0.0                   # meme vue que le direct (Yahoo : pas de volume forex) -> aucun ecart recherche / direct
    return a


def coinbase(sym, step, days):
    """Crypto : bougies Coinbase (API publique), 300 par requete, sur `days` jours."""
    gran = 300 if step == 300 else 900 if step in (900, 1800) else 3600     # 30 min = 2 bougies de 15 min
    end = int(time.time()) // gran * gran
    start_all = end - days * 86400
    rows = {}
    t = start_all
    while t < end:
        t2 = min(end, t + 300 * gran)
        try:
            r = SESSION.get("https://api.exchange.coinbase.com/products/%s-USD/candles" % sym, timeout=20, params={
                "granularity": gran, "start": datetime.datetime.utcfromtimestamp(t).isoformat(),
                "end": datetime.datetime.utcfromtimestamp(t2).isoformat()})
            if r.status_code == 429:
                time.sleep(2)
                continue
            if r.status_code != 200:
                raise RuntimeError("HTTP %d" % r.status_code)
            for tt, lo, hi, op, cl, vol in r.json():
                rows[int(tt)] = (int(tt), float(op), float(hi), float(lo), float(cl), float(vol))
        except Exception as ex:
            if not rows:
                raise RuntimeError("Coinbase %s : %s" % (sym, str(ex)[:80]))
        t = t2
        time.sleep(0.12)
    a = np.array([rows[k] for k in sorted(rows)], dtype=float).reshape(-1, 6)
    if gran != step:
        a = _resample(a, step)
    return a


def long_bars(sym, kind, tf):
    """Historique long pour la recherche : Dukascopy (forex, 15 min), Coinbase (crypto) ; Yahoo en secours."""
    step = TF[tf][2]
    try:
        if kind == "fx" and tf in FX_DAYS:
            v, d = FX_DAYS[tf]
            return dukascopy(sym, step, int(os.environ.get(v, d)))
        if kind == "cr" and tf in CR_DAYS:
            v, d = CR_DAYS[tf]
            return coinbase(sym, step, int(os.environ.get(v, d)))
    except Exception as ex:
        log.warning("historique long %s %s : %s (Yahoo en secours)", sym, tf, ex)
    return bars(sym, kind, tf)


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


def funding(sym, days=1200):
    """Taux de financement horaire des perpetuels : Kraken Futures (public), Hyperliquid en secours.
       Retourne (n, 2) : temps, taux relatif par heure."""
    k = {"BTC": "XBT"}.get(sym, sym)
    try:
        r = SESSION.get("https://futures.kraken.com/derivatives/api/v4/historicalfundingrates",
                        params={"symbol": "PF_%sUSD" % k}, timeout=20)
        rates = r.json().get("rates", []) if r.status_code == 200 else []
        rows = [(datetime.datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")).timestamp(),
                 float(x.get("relativeFundingRate", x.get("fundingRate", 0.0)))) for x in rates]
        if len(rows) > 100:
            return np.array(sorted(rows), dtype=float)
    except Exception as ex:
        log.warning("funding Kraken %s : %s", sym, ex)
    rows = []
    start = int((time.time() - days * 86400) * 1000)
    for _ in range(60):                                   # Hyperliquid : 500 points par appel
        try:
            r = SESSION.post("https://api.hyperliquid.xyz/info", json={"type": "fundingHistory", "coin": sym,
                                                                         "startTime": start}, timeout=20)
            batch = r.json() if r.status_code == 200 else []
        except Exception as ex:
            log.warning("funding Hyperliquid %s : %s", sym, ex)
            break
        if not batch:
            break
        rows += [(x["time"] / 1000.0, float(x["fundingRate"])) for x in batch]
        start = int(batch[-1]["time"]) + 1
        if len(batch) < 500:
            break
        time.sleep(0.2)
    if len(rows) < 100:
        raise RuntimeError("funding %s indisponible" % sym)
    return np.array(sorted(set(rows)), dtype=float)

