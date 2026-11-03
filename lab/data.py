"""Donnees gratuites, sans inscription : Yahoo Finance (bougies), Coinbase / Bitstamp (carnets d'ordres crypto).
Prix de reference neutres (aucun courtier) : les niveaux sont aussi donnes en % pour s'appliquer partout."""
import concurrent.futures as cf
import datetime
import io
import json
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
TF = {"1H": ("60m", "730d", 3600)}
# historique long de la recherche (jours) : meme nombre de bougies environ pour chaque unite de temps
FX_DAYS = {"1H": ("FX_DAYS_1H", 3650)}        # forex : 10 ans de bougies 1 h Dukascopy
CR_DAYS = {"1H": ("CR_DAYS_1H", 1095)}        # crypto : 3 ans de bougies 1 h Coinbase


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


DUKA_URL = "https://datafeed.dukascopy.com/datafeed/%s/%04d/%02d/%02d/%s"


def _duka_get(sym, day, fname, cache):
    """Telecharge un fichier Dukascopy ; les jours termines sont gardes sur disque (DUKA_CACHE) pour ne pas
       les retelecharger a chaque recherche."""
    root = os.environ.get("DUKA_CACHE", "")
    path = os.path.join(root, sym, "%s_%s" % (day.isoformat(), fname)) if root and cache else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read() or None
    try:
        r = SESSION.get(DUKA_URL % (sym, day.year, day.month - 1, day.day, fname), timeout=20)
    except Exception:
        return None
    if r.status_code not in (200, 404):
        return None
    content = r.content if r.status_code == 200 else b""
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
    return content or None


def _duka_day(sym, day):
    old = day < datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=2)
    content = _duka_get(sym, day, "BID_candles_min_1.bi5", cache=old)
    if not content:
        return None
    try:
        raw = lzma.decompress(content, format=lzma.FORMAT_ALONE)
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
    with cf.ThreadPoolExecutor(12) as ex:
        for res in ex.map(lambda d: _duka_day(sym, d), dates):
            if res:
                rows.extend(res)
    if len(rows) < 1000:
        raise RuntimeError("Dukascopy %s : historique indisponible" % sym)
    a = _resample(np.array(rows, dtype=float), step)
    VOLUME[sym] = (a[:, 0].copy(), a[:, 5].copy())        # volume des ticks : order flow forex (a part)
    a[:, 5] = 0.0                   # meme vue que le direct temps reel (TrueFX : pas de volume) -> aucun ecart
    return a


def _duka_month(sym, year, month):
    """Bougies 1 h d'un mois termine (un seul petit fichier Dukascopy par mois), gardees en cache disque."""
    root = os.environ.get("DUKA_CACHE", "")
    path = os.path.join(root, sym, "%04d-%02d_H1.bi5" % (year, month)) if root else None
    content = None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            content = f.read() or None
    else:
        try:
            r = SESSION.get("https://datafeed.dukascopy.com/datafeed/%s/%04d/%02d/BID_candles_hour_1.bi5" % (
                sym, year, month - 1), timeout=30)
            if r.status_code in (200, 404):
                content = r.content if r.status_code == 200 else b""
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(content)
                content = content or None
        except Exception:
            content = None
    if not content:
        return None
    try:
        raw = lzma.decompress(content, format=lzma.FORMAT_ALONE)
    except Exception:
        return None
    pt = DUKA_POINT.get(sym[3:], DUKA_POINT.get(sym[:3], 1e-5))
    t0 = int(datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc).timestamp())
    rows = []
    for off in range(0, len(raw) - 23, 24):
        t, o, c, lo, hi, v = struct.unpack(">5if", raw[off:off + 24])
        rows.append((t0 + t, o * pt, hi * pt, lo * pt, c * pt, 0.0))
    return rows


def dukascopy_hours(sym, days):
    """Historique long 1 h : mois termines (fichiers mensuels) + mois en cours (bougies 1 min des jours passes)."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    start = today - datetime.timedelta(days=days)
    months = []
    y, m = start.year, start.month
    while (y, m) < (today.year, today.month):
        months.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda ym: _duka_month(sym, *ym), months))
    if sum(1 for x in res if x) < max(1, len(months) // 2):
        raise RuntimeError("fichiers mensuels Dukascopy indisponibles pour %s" % sym)
    for x in res:
        rows += x or []
    first = datetime.date(today.year, today.month, 1)
    cur = (today - first).days
    if cur > 0:
        try:
            recent = dukascopy(sym, 3600, cur)
        except Exception:
            recent = None                                # debut de mois / week-end : rien a ajouter
        if recent is not None and len(recent):
            last = max(r[0] for r in rows) if rows else 0
            rows += [tuple(r) for r in recent if r[0] > last]
    a = np.array(sorted({int(r[0]): r for r in rows}.values()), dtype=float)
    a = a[a[:, 0] >= int(datetime.datetime(start.year, start.month, start.day, tzinfo=datetime.timezone.utc).timestamp())]
    VOLUME[sym] = (a[:, 0].copy(), a[:, 5].copy())        # volume des ticks : order flow forex (a part)
    a[:, 5] = 0.0
    return a


def _duka_hour(sym, hour_dt):
    """Ticks d'une heure terminee (publies par Dukascopy apres la fin de l'heure) -> bougies 1 min (bid, volume bid)."""
    content = _duka_get(sym, hour_dt.date(), "%02dh_ticks.bi5" % hour_dt.hour, cache=False)
    if not content:
        return None
    try:
        raw = lzma.decompress(content, format=lzma.FORMAT_ALONE)
    except Exception:
        return None
    pt = DUKA_POINT.get(sym[3:], DUKA_POINT.get(sym[:3], 1e-5))
    t0 = int(hour_dt.timestamp())
    mins = {}
    for off in range(0, len(raw) - 19, 20):
        ms, _ask, bid, _av, bv = struct.unpack(">3i2f", raw[off:off + 20])
        t = t0 + (ms // 60000) * 60
        p = bid * pt
        m = mins.get(t)
        mins[t] = [p, p, p, p, bv] if m is None else [m[0], max(m[1], p), min(m[2], p), p, m[4] + bv]
    return [(t, *v) for t, v in sorted(mins.items())]


LIVE_CACHE = {}
VOLUME = {}                       # volume des ticks Dukascopy par paire (order flow forex), garde a part


def dukascopy_live(sym, step, days, recent=None):
    """Direct forex sur les memes prix que la recherche : jours termines (bougies 1 min) + heures terminees du jour
       (ticks). Tout est garde en memoire : seuls les nouveaux fichiers sont telecharges."""
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()
    need = []
    for i in range(days, 0, -1):
        d = today - datetime.timedelta(days=i)
        if d.weekday() != 5 and (sym, d) not in LIVE_CACHE:
            need.append(d)
    with cf.ThreadPoolExecutor(8) as ex:
        for d, res in zip(need, ex.map(lambda d: _duka_day(sym, d), need)):
            if res:
                LIVE_CACHE[(sym, d)] = res
            elif d >= today - datetime.timedelta(days=2):            # fichier du jour pas encore publie : par heures
                hrs = []
                for hh in range(24):
                    r = _duka_hour(sym, datetime.datetime(d.year, d.month, d.day, hh, tzinfo=datetime.timezone.utc))
                    hrs += r or []
                if hrs:
                    LIVE_CACHE[(sym, d, "h")] = hrs
    rows = []
    for i in range(days, 0, -1):
        d = today - datetime.timedelta(days=i)
        rows += LIVE_CACHE.get((sym, d)) or LIVE_CACHE.get((sym, d, "h")) or []
    for hh in range(now.hour):                                       # heures terminees d'aujourd'hui
        key = (sym, today, hh)
        if key not in LIVE_CACHE:
            r = _duka_hour(sym, datetime.datetime(today.year, today.month, today.day, hh, tzinfo=datetime.timezone.utc))
            if r:
                LIVE_CACHE[key] = r
        rows += LIVE_CACHE.get(key) or []
    if len(rows) < 1000:
        raise RuntimeError("Dukascopy direct %s : pas assez de donnees" % sym)
    last = max(r[0] for r in rows)
    duka_last = last
    rows += [r for r in (recent or []) if r[0] > last]      # minutes temps reel apres le dernier fichier publie
    a = _resample(np.array(rows, dtype=float), step)
    vol = a[:, 5].copy()
    vol[a[:, 0] + step > duka_last + 60] = np.nan           # bougie pas encore couverte par Dukascopy : volume inconnu
    VOLUME[sym] = (a[:, 0].copy(), vol)
    a[:, 5] = 0.0
    return a


TRUEFX = {"start": None, "min": {}, "session": None, "mode": "public", "paires": 0}
TFX_URL = "https://webrates.truefx.com/rates/connect.html"


def _truefx_session():
    """Compte TrueFX (secrets TRUEFX_USER / TRUEFX_PASS) : plus de paires. Jamais affiche ni enregistre."""
    u, p = os.environ.get("TRUEFX_USER", ""), os.environ.get("TRUEFX_PASS", "")
    if not u or not p:
        return None
    pairs = ",".join(s[:3] + "/" + s[3:] for s in os.environ.get("FX", "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,"
                     "NZDUSD,EURJPY,GBPJPY,EURGBP,AUDJPY,EURCHF,GBPCHF,CADJPY,CHFJPY,EURAUD,EURCAD,GBPAUD,AUDNZD").split(",")
                     if not s.startswith(("XAU", "XAG")))
    try:
        r = SESSION.get(TFX_URL, params={"u": u, "p": p, "q": "ozrates", "c": pairs, "f": "csv"}, timeout=10)
        sid = r.text.strip()
        ok = r.status_code == 200 and sid and "," not in sid and "\n" not in sid and "<" not in sid and len(sid) < 200
        if ok:
            t = SESSION.get(TFX_URL, params={"id": sid, "f": "csv", "s": "y"}, timeout=10)
            if t.status_code == 200 and "," in t.text:
                return sid
        # diagnostic sans aucun secret : code HTTP, taille et forme de la reponse
        low = sid.lower()
        hint = next((w for w in ("not authorized", "invalid", "error", "denied", "expired") if w in low), "")
        TRUEFX["mode"] = "erreur compte : HTTP %d, reponse %d caracteres%s%s" % (
            r.status_code, len(sid), ", %d ':'" % sid.count(":") if sid else "", (", mot-cle : " + hint) if hint else "")
    except Exception as ex:
        TRUEFX["mode"] = "erreur compte : %s" % type(ex).__name__
    return None


def poll_truefx():
    """Prix interbancaires en temps reel TrueFX -> bougies 1 min (bid) par paire. Avec le compte si les secrets
       sont presents (plus de paires), sinon le flux public (10 paires)."""
    if TRUEFX["session"] is None and not TRUEFX.get("tente") and os.environ.get("TRUEFX_USER"):
        TRUEFX["tente"] = True
        TRUEFX["session"] = _truefx_session()
        if TRUEFX["session"]:
            TRUEFX["mode"] = "compte"
    params = {"id": TRUEFX["session"], "f": "csv", "s": "y"} if TRUEFX["session"] else {"f": "csv"}
    try:
        r = SESSION.get(TFX_URL, params=params, timeout=8)
        if r.status_code != 200 or ("," not in r.text and TRUEFX["session"]):
            TRUEFX["session"] = None                       # session expiree : reconnexion au prochain tour
            TRUEFX["mode"] = "public"
            return 0
    except Exception:
        return 0
    n = 0
    now = time.time()
    for line in r.text.strip().splitlines():
        p = line.strip().split(",")
        if len(p) < 6 or "/" not in p[0]:
            continue
        try:
            sym, t, bid = p[0].replace("/", ""), int(p[1]) / 1000.0, float(p[2] + p[3])
        except ValueError:
            continue
        if TRUEFX["start"] is None:
            TRUEFX["start"] = now
        m = int(t // 60) * 60
        d = TRUEFX["min"].setdefault(sym, {})
        c = d.get(m)
        d[m] = [bid, bid, bid, bid] if c is None else [c[0], max(c[1], bid), min(c[2], bid), bid]
        for k in [k for k in d if k < now - 3 * 3600]:
            del d[k]
        n += 1
    TRUEFX["paires"] = max(TRUEFX["paires"], n) if n else TRUEFX["paires"]
    return n


def truefx_minutes(sym, step):
    """Minutes temps reel d'une paire, seulement a partir d'une bougie complete suivie depuis son debut."""
    if TRUEFX["start"] is None:
        return []
    first = (int(TRUEFX["start"]) // step + 1) * step
    return [(t, o, h, l, c, 0.0) for t, (o, h, l, c) in sorted(TRUEFX["min"].get(sym, {}).items()) if t >= first]


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
    """Historique long pour la recherche : Dukascopy (forex, 10 ans), Coinbase (crypto, 3 ans) ; Yahoo en secours."""
    step = TF[tf][2]
    try:
        if kind == "fx" and tf in FX_DAYS:
            v, d = FX_DAYS[tf]
            try:
                return dukascopy_hours(sym, int(os.environ.get(v, d)))       # 10 ans en ~120 fichiers par paire
            except Exception as ex:
                log.warning("Dukascopy mensuel %s : %s (fichiers journaliers)", sym, ex)
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



# ---- positionnement des gros acteurs et peur des options (sources publiques, sans inscription) ----
def _cached_get(name, url, keep=True, timeout=40):
    """Telechargement avec cache disque (DUKA_CACHE) pour les fichiers termines (annees passees)."""
    root = os.environ.get("DUKA_CACHE", "")
    path = os.path.join(root, "_sources", name) if root and keep else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    r = SESSION.get(url, timeout=timeout, headers={"User-Agent": "strategy-lab/1.0"})
    r.raise_for_status()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(r.content)
    return r.content


def dvol(cur="BTC", days=1200):
    """Indice de volatilite implicite Deribit (DVOL), bougies 1 h depuis 2021. Retourne (n, 2) : temps (debut de la
       bougie, s), cloture."""
    end = int(time.time() * 1000)
    base, step = 1617235200000, 40 * 86400 * 1000
    start = max(int((time.time() - days * 86400) * 1000), base)
    rows = {}
    t = base + (start - base) // step * step          # fenetres sur une grille fixe : le cache disque est reutilise
    while t < end:
        u = min(t + step, end)
        name = "dvol_%s_%d_%d.json" % (cur, t, u)
        try:
            raw = _cached_get(name, "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=%s"
                              "&resolution=3600&start_timestamp=%d&end_timestamp=%d" % (cur, t, u),
                              keep=u < end - 3 * 86400 * 1000)
            for x in json.loads(raw).get("result", {}).get("data", []):
                rows[int(x[0])] = float(x[4])
        except Exception as ex:
            log.warning("DVOL %s : %s", cur, ex)
        t = u
    return np.array(sorted((k / 1000.0, v) for k, v in rows.items()), dtype=float).reshape(-1, 2)


COT_NAMES = {"EURO FX": "EUR", "JAPANESE YEN": "JPY", "BRITISH POUND": "GBP", "BRITISH POUND STERLING": "GBP",
             "SWISS FRANC": "CHF", "CANADIAN DOLLAR": "CAD", "AUSTRALIAN DOLLAR": "AUD", "NZ DOLLAR": "NZD",
             "NEW ZEALAND DOLLAR": "NZD", "USD INDEX": "USD", "U.S. DOLLAR INDEX": "USD", "BITCOIN": "BTC"}


def cot(years=11):
    """Rapport COT de la CFTC (traders in financial futures) : positions des fonds a levier et des gerants.
       Retourne {devise: (n, 4) : disponibilite (s), net fonds / OI, net gerants / OI, net dealers / OI}.
       Disponibilite prudente : rapport du mardi utilise a partir du lundi suivant 00 h UTC."""
    import csv
    import zipfile
    this = datetime.date.today().year
    out = {}
    for y in range(this - years + 1, this + 1):
        try:
            raw = _cached_get("cot_fin_%d.zip" % y, "https://www.cftc.gov/files/dea/history/fut_fin_txt_%d.zip" % y,
                              keep=y < this)
            z = zipfile.ZipFile(io.BytesIO(raw))
            text = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
        except Exception as ex:
            log.warning("COT %d : %s", y, ex)
            continue
        for row in csv.DictReader(text):
            cur = COT_NAMES.get((row.get("Market_and_Exchange_Names") or "").split(" - ")[0].strip())
            if not cur:
                continue
            try:
                d = datetime.date.fromisoformat(row["Report_Date_as_YYYY-MM-DD"][:10])
                oi = float(row["Open_Interest_All"])
                f = lambda a, b: (float(row[a]) - float(row[b])) / oi if oi > 0 else np.nan  # noqa: E731
                vals = (f("Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
                        f("Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All"),
                        f("Dealer_Positions_Long_All", "Dealer_Positions_Short_All"))
            except (KeyError, ValueError):
                continue
            avail = datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc).timestamp() + 6 * 86400
            out.setdefault(cur, {})[avail] = vals
    return {k: np.array([(t,) + v[t] for t in sorted(v)], dtype=float) for k, v in out.items()}


def binance_flow(sym, days=1200):
    """Order flow crypto : bougies 1 h Binance (data-api.binance.vision, public, sans inscription) avec le volume des
       acheteurs agressifs (taker buy). Retourne (n, 3) : temps (debut de bougie, s), volume, volume acheteur."""
    pair = "%sUSDT" % sym
    end = int(time.time() * 1000)
    step = 1000 * 3600 * 1000
    base = 1500000000000
    t = base + (max(int((time.time() - days * 86400) * 1000), base) - base) // step * step
    rows = {}
    while t < end:
        u = min(t + step, end)
        try:
            raw = _cached_get("binance_%s_%d.json" % (pair, t),
                              "https://data-api.binance.vision/api/v3/klines?symbol=%s&interval=1h&startTime=%d"
                              "&endTime=%d&limit=1000" % (pair, t, u - 1), keep=u < end - 2 * 86400 * 1000)
            for k in json.loads(raw):
                rows[int(k[0]) // 1000] = (float(k[5]), float(k[9]))
        except Exception as ex:
            log.warning("Binance %s : %s", pair, ex)
        t = u
    return np.array([(k,) + v for k, v in sorted(rows.items())], dtype=float).reshape(-1, 3)
