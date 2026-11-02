"""Specialiste « flux des operateurs » : quand des acteurs sont OBLIGES d'acheter ou de vendre (banques, entreprises,
fonds), a heure fixe, independamment du prix. Idees publiques, reecrites sans code copie, toutes causales :
   - fixing de Londres (WM/Reuters, 16 h Londres) : ordres des clients des banques executes au fixing ;
   - jours Gotobi (5, 10, 15, 20, 25, fin de mois) et fixing de Tokyo (9 h 55 Tokyo) : achats de dollars des
     importateurs japonais ;
   - fin de mois / de trimestre : reequilibrage des fonds (contre la performance du mois) ;
   - NFP (emploi americain, 8 h 30 New York, premier vendredi du mois) : chasse de stops puis vraie direction ;
   - changement de jour bancaire (17 h New York) : trou de liquidite ;
   - divergence SMT (ICT) : l'actif fait un nouvel extreme mais pas sa reference correlee (EURUSD / BTC) ;
   - force de chaque devise calculee sur toutes les paires (forex) / force relative au panier (crypto) ;
   - gap du CME sur le bitcoin (cloture du vendredi contre prix du week-end) ; gap du lundi en forex."""
import datetime
import warnings

import numpy as np

NAN = np.nan
H = 3600
_DST_CACHE = {}


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _nth_sunday(y, m, nth):
    d = datetime.date(y, m, 1)
    first = d + datetime.timedelta(days=(6 - d.weekday()) % 7)
    if nth > 0:
        return first + datetime.timedelta(weeks=nth - 1)
    last = datetime.date(y + (m == 12), m % 12 + 1, 1) - datetime.timedelta(days=1)
    return last - datetime.timedelta(days=(last.weekday() + 1) % 7)


def _offsets(day):
    """(decalage New York, decalage Londres) en heures par rapport a UTC pour ce jour (heure d'ete incluse)."""
    if day in _DST_CACHE:
        return _DST_CACHE[day]
    d = datetime.date.fromordinal(int(day) + 719163)              # jour UTC depuis 1970 -> date
    y = d.year
    us = _nth_sunday(y, 3, 2) <= d < _nth_sunday(y, 11, 1)
    uk = _nth_sunday(y, 3, -1) <= d < _nth_sunday(y, 10, -1)
    _DST_CACHE[day] = (-4 if us else -5, 1 if uk else 0)
    return _DST_CACHE[day]


def _clock(ts):
    day = (ts // 86400).astype(np.int64)
    uniq, inv = np.unique(day, return_inverse=True)
    offs = np.array([_offsets(d) for d in uniq])[inv]
    return day, offs[:, 0], offs[:, 1]


def _event_window(ts, c, atr, ev_ts, before=6, after=6):
    """Heures signees jusqu'a l'evenement le plus proche (fenetre), et mouvement depuis `before` h avant lui."""
    n = len(c)
    rel, move = np.full(n, NAN), np.full(n, NAN)
    if not len(ev_ts):
        return rel, move
    ev_ts = np.sort(np.asarray(ev_ts, dtype=float))
    j = np.clip(np.searchsorted(ev_ts, ts), 0, len(ev_ts) - 1)
    jp = np.clip(j - 1, 0, len(ev_ts) - 1)
    for i in range(n):
        best = None
        for e in (ev_ts[j[i]], ev_ts[jp[i]]):
            d = (ts[i] - e) / H
            if -before <= d <= after and (best is None or abs(d) < abs(best[0])):
                best = (d, e)
        if best is None:
            continue
        rel[i] = best[0]
        k = np.searchsorted(ts, best[1] - before * H)
        if k < i and np.isfinite(atr[i]) and atr[i] > 0:
            move[i] = (c[i] - c[k]) / atr[i]
    return rel, move


def _days_range(ts):
    d0, d1 = int(ts[0] // 86400), int(ts[-1] // 86400)
    return [datetime.date.fromordinal(d + 719163) for d in range(d0, d1 + 2)]


def calendrier(ts, c, atr):
    out = {}
    days = _days_range(ts)
    # fixing de Londres : 16 h heure de Londres
    fix = [(datetime.datetime(d.year, d.month, d.day, 16) - datetime.timedelta(hours=_offsets(d.toordinal() - 719163)[1]))
           .replace(tzinfo=datetime.timezone.utc).timestamp() for d in days if d.weekday() < 5]
    out["heures_fixing_londres"], out["mouvement_avant_fixing"] = _event_window(ts, c, atr, fix)
    # Gotobi : 5, 10, 15, 20, 25 et dernier jour du mois (reporte au jour ouvre precedent) ; fixing de Tokyo 00 h 55 UTC
    got = set()
    for d in days:
        last = (datetime.date(d.year + (d.month == 12), d.month % 12 + 1, 1) - datetime.timedelta(days=1)).day
        if d.day in (5, 10, 15, 20, 25) or d.day == last:
            g = d
            while g.weekday() >= 5:
                g -= datetime.timedelta(days=1)
            got.add(g)
    tok = [datetime.datetime(g.year, g.month, g.day, 0, 55, tzinfo=datetime.timezone.utc).timestamp() for g in got]
    out["heures_fixing_tokyo_gotobi"], out["mouvement_avant_tokyo"] = _event_window(ts, c, atr, tok, 8, 4)
    dts = np.array([datetime.date.fromordinal(int(t // 86400) + 719163) for t in ts])
    out["jour_gotobi"] = np.array([1.0 if d in got else 0.0 for d in dts])
    # NFP : premier vendredi du mois, 8 h 30 New York
    nfp = []
    for d in days:
        if d.weekday() == 4 and d.day <= 7:
            ny = _offsets(d.toordinal() - 719163)[0]
            nfp.append(datetime.datetime(d.year, d.month, d.day, 8, 30, tzinfo=datetime.timezone.utc).timestamp() - ny * H)
    out["heures_nfp"], out["mouvement_nfp"] = _event_window(ts, c, atr, nfp, 6, 12)
    # fin de mois / trimestre : jours ouvres restants, performance du mois (les fonds reequilibrent a contre-sens)
    n = len(c)
    left, perf, quarter = np.full(n, NAN), np.full(n, NAN), np.zeros(n)
    month = np.array([d.year * 12 + d.month for d in dts])
    uniq, start = np.unique(month, return_index=True)
    for k in range(len(uniq)):
        s, e = start[k], (start[k + 1] if k + 1 < len(start) else n)
        d = dts[s]
        last = datetime.date(d.year + (d.month == 12), d.month % 12 + 1, 1) - datetime.timedelta(days=1)
        for i in range(s, e):
            cur = dts[i]
            left[i] = float(np.busday_count(cur, last + datetime.timedelta(days=1)) - (1 if np.is_busday(cur) else 0))
            if np.isfinite(atr[i]) and atr[i] > 0:
                perf[i] = (c[i] - c[s]) / atr[i]
            quarter[i] = 1.0 if d.month in (3, 6, 9, 12) else 0.0
    out["jours_ouvres_fin_mois"], out["performance_mois"] = left, perf
    out["fin_trimestre"] = np.where(left <= 3, quarter, 0.0)
    out["pression_reequilibrage"] = np.where(left <= 3, -np.sign(perf) * np.minimum(np.abs(perf), 20), 0.0)
    # changement de jour bancaire : 17 h New York
    day, ny, _uk = _clock(ts)
    hour_ny = ((ts / H + ny) % 24)
    out["heures_depuis_17h_new_york"] = (hour_ny - 17) % 24
    return out


def smt(c, bench, p=20):
    """Divergence SMT : nouvel extreme de cloture sur p bougies de l'actif sans celui de sa reference."""
    n = len(c)
    out = np.zeros(n)
    if bench is None or len(bench) != n:
        return {}
    for i in range(p, n):
        a_hi, a_lo = c[i] > c[i - p:i].max(), c[i] < c[i - p:i].min()
        b = bench[i - p:i + 1]
        if not np.all(np.isfinite(b)):
            continue
        b_hi, b_lo = b[-1] > b[:-1].max(), b[-1] < b[:-1].min()
        if a_hi and not b_hi:
            out[i] = -1.0                             # l'actif chasse les stops du haut seul : faiblesse cachee
        elif a_lo and not b_lo:
            out[i] = 1.0
        elif b_hi and not a_hi and c[i] < c[i - p:i].max():
            out[i] = -0.5                             # la reference monte, pas l'actif : actif faible
        elif b_lo and not a_lo and c[i] > c[i - p:i].min():
            out[i] = 0.5
    return {"divergence_smt": out}


def force(sym, kind, panier, n, w=(6, 24)):
    """Forex : force de chaque devise (moyenne des rendements de toutes les paires qui la contiennent).
       Crypto : force relative de l'actif par rapport a la mediane du panier."""
    if not panier or sym not in panier:
        return {}
    out = {}
    for p in w:
        rets = {}
        for s, x in panier.items():
            x = np.asarray(x, dtype=float)
            if len(x) != n:
                continue
            r = np.full(n, NAN)
            r[p:] = np.log(x[p:] / x[:-p])
            rets[s] = r
        if sym not in rets:
            continue
        if kind == "fx":
            cur = {}
            for s, r in rets.items():
                if len(s) != 6:
                    continue
                b, q = s[:3], s[3:]
                cur.setdefault(b, []).append(r)
                cur.setdefault(q, []).append(-r)
            strength = {k: np.nanmean(np.vstack(v), axis=0) for k, v in cur.items() if len(v) >= 2}
            disp = np.nanstd(np.vstack(list(strength.values())), axis=0) if strength else None
            b, q = sym[:3], sym[3:]
            if b in strength and q in strength and len(sym) == 6:
                out["force_base_%dh" % p] = _div(strength[b], disp)
                out["force_cotee_%dh" % p] = _div(strength[q], disp)
                out["ecart_force_%dh" % p] = _div(strength[b] - strength[q], disp)
                allv = np.vstack(list(strength.values()))
                out["rang_force_base_%dh" % p] = np.sum(allv < strength[b], axis=0).astype(float)
                out["rang_force_cotee_%dh" % p] = np.sum(allv < strength[q], axis=0).astype(float)
        else:
            allr = np.vstack(list(rets.values()))
            med, sd = np.nanmedian(allr, axis=0), np.nanstd(allr, axis=0)
            out["force_relative_panier_%dh" % p] = _div(rets[sym] - med, sd)
    return out


def gaps(ts, o, h, l, c, atr, kind):
    """Gap du lundi (forex : trou de cotation du week-end) et gap du CME (crypto : vendredi 21-22 h UTC -> dimanche
       22-23 h UTC). Distance au prix de cloture du vendredi tant que le gap n'est pas refermé."""
    n = len(c)
    dist, open_ = np.full(n, NAN), np.zeros(n)
    ref, filled, side = NAN, True, 0
    if kind == "fx":
        for i in range(1, n):
            if ts[i] - ts[i - 1] > 6 * H:                 # reouverture apres le week-end
                ref, filled = c[i - 1], False
                side = 1 if o[i] > ref else -1
            if not filled:
                if (side > 0 and l[i] <= ref) or (side < 0 and h[i] >= ref):
                    filled = True
                elif np.isfinite(atr[i]) and atr[i] > 0:
                    dist[i], open_[i] = (c[i] - ref) / atr[i], float(side)
        return {"gap_lundi_dist": dist, "gap_lundi_ouvert": open_}
    day, ny, _uk = _clock(ts)
    wd = ((ts // 86400 + 3) % 7).astype(int)              # 0 = lundi
    hour_ny = (ts / H + ny) % 24
    for i in range(1, n):
        if wd[i] == 4 and 16 <= hour_ny[i] < 17:          # derniere heure CME du vendredi (17 h New York = cloture)
            ref, filled, side = c[i], True, 0
        if np.isfinite(ref) and side == 0 and ((wd[i] == 6 and hour_ny[i] >= 18) or wd[i] == 0):
            side = 1 if c[i] > ref else -1                # reouverture du CME dimanche 18 h New York
            filled = False
        if not filled and side != 0:
            if (side > 0 and l[i] <= ref) or (side < 0 and h[i] >= ref):
                filled = True
            elif np.isfinite(atr[i]) and atr[i] > 0:
                dist[i], open_[i] = (c[i] - ref) / atr[i], float(side)
    return {"gap_cme_dist": dist, "gap_cme_ouvert": open_}


# ---- idees issues de la recherche approfondie (etudes publiques, voir eclaireur_integre.json) ----
# FOMC : jour de decision (2e jour) des reunions PROGRAMMEES uniquement (connues a l'avance), source federalreserve.gov
FOMC = ("2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14 "
        "2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13 "
        "2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19 "
        "2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11 "
        "2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16 "
        "2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15 "
        "2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14 "
        "2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13 "
        "2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18 "
        "2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 "
        "2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09 "
        "2027-01-27 2027-03-17 2027-04-28 2027-06-09 2027-07-28 2027-09-15 2027-10-27 2027-12-08").split()
# heures locales des places (devise -> fenetre en heures UTC hors heure d'ete ; Europe et New York ajustees)
LOCAL = {"JPY": "tokyo", "AUD": "tokyo", "NZD": "tokyo", "EUR": "europe", "CHF": "europe", "GBP": "europe",
         "USD": "new_york", "CAD": "new_york"}


def _usd_side(sym):
    """+1 si une hausse du dollar fait monter la paire (USDxxx), -1 si elle la fait baisser (xxxUSD), 0 sinon."""
    if not sym or len(sym) != 6:
        return 0
    return 1 if sym.startswith("USD") else (-1 if sym.endswith("USD") else 0)


def _ts(d, hh, mm=0):
    return datetime.datetime(d.year, d.month, d.day, hh, mm, tzinfo=datetime.timezone.utc).timestamp()


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    mo = (h + l_ - 7 * m + 114) // 31
    return datetime.date(y, mo, (h + l_ - 7 * m + 114) % 31 + 1)


def _holidays(y):
    """Jours feries des grandes places : {date: ensemble des devises dont la place est fermee}."""
    out = {}

    def add(d, *cur):
        out.setdefault(d, set()).update(cur)
    e = _easter(y)
    add(e - datetime.timedelta(days=2), "USD", "GBP", "EUR", "CHF", "AUD", "NZD", "CAD")      # vendredi saint
    add(e + datetime.timedelta(days=1), "GBP", "EUR", "CHF", "AUD", "NZD", "CAD")             # lundi de Paques
    for md in ((12, 24), (12, 25), (12, 26), (12, 31), (1, 1)):
        add(datetime.date(y, *md), "USD", "GBP", "EUR", "CHF", "JPY", "AUD", "NZD", "CAD")
    jul4 = datetime.date(y, 7, 4)
    add(jul4 + datetime.timedelta(days={5: -1, 6: 1}.get(jul4.weekday(), 0)), "USD")
    nov1 = datetime.date(y, 11, 1)
    add(nov1 + datetime.timedelta(days=(3 - nov1.weekday()) % 7 + 21), "USD")                  # Thanksgiving
    for md in ((4, 29), (5, 3), (5, 4), (5, 5)):
        add(datetime.date(y, *md), "JPY")                                                       # Golden Week
    may1 = datetime.date(y, 5, 1)
    add(may1 + datetime.timedelta(days=(7 - may1.weekday()) % 7), "GBP")                         # 1er lundi de mai
    for m_ in (5, 8):                                                                           # derniers lundis
        last = datetime.date(y + (m_ == 12), m_ % 12 + 1, 1) - datetime.timedelta(days=1)
        add(last - datetime.timedelta(days=last.weekday()), "GBP")
    return out


def recherche(sym, kind, ts, o, h, l, c, atr):
    n = len(c)
    out = {}
    days = _days_range(ts)
    dts = [datetime.date.fromordinal(int(t // 86400) + 719163) for t in ts]
    hour = (ts // H % 24).astype(int)
    _day, ny, uk = _clock(ts)                      # l'Europe continentale change d'heure aux memes dates que Londres
    usd = _usd_side(sym)
    # 1. « W » du dollar (Krohn, Mueller, Whelan) : fixing BCE 14 h 15 heure de Francfort
    bce = [_ts(d, 14, 15) - (1 + _offsets(d.toordinal() - 719163)[1]) * H for d in days if d.weekday() < 5]
    out["heures_fixing_bce"], out["mouvement_avant_bce"] = _event_window(ts, c, atr, bce, 7, 4)
    fra = (ts / H + 1 + uk) % 24                   # heure de Francfort
    ny_h = (ts / H + ny) % 24
    w = np.where((fra >= 7) & (fra < 14.25), 1.0, np.where((ny_h >= 11) & (ny_h < 17), -1.0, 0.0))
    if usd:
        out["biais_w_dollar"] = w * usd
    # 2. une devise baisse pendant ses propres heures (Breedon et Ranaldo)
    if kind == "fx" and sym and len(sym) == 6:
        def active(cur):
            z = LOCAL.get(cur)
            if z == "tokyo":
                return (hour >= 0) & (hour < 6)
            if z == "europe":
                return (fra >= 8) & (fra < 13)
            if z == "new_york":
                return (ny_h >= 8) & (ny_h < 16)
            return np.zeros(n, bool)
        out["biais_session_locale"] = active(sym[3:]).astype(float) - active(sym[:3]).astype(float)
    # 3. jour de la Fed (Mueller, Tahbaz-Salehi, Vedolin) : 14 h New York le jour de decision
    fomc = []
    for s in FOMC:
        d = datetime.date.fromisoformat(s)
        fomc.append(_ts(d, 14) - _offsets(d.toordinal() - 719163)[0] * H)
    out["heures_fomc"], out["mouvement_fomc"] = _event_window(ts, c, atr, fomc, 14, 10)
    fdays = {datetime.date.fromisoformat(s) for s in FOMC}
    jf = np.array([1.0 if d in fdays else 0.0 for d in dts])
    out["jour_fomc"] = jf
    if usd:
        out["biais_fomc_vendeur_dollar"] = np.where((jf == 1) & (out["heures_fomc"] <= 0), -usd, 0.0)
    # 4. or et argent : hausse de la nuit, baisse entre les fixings LBMA de Londres (10 h 30 et 15 h 00)
    if sym in ("XAUUSD", "XAGUSD"):
        lon = (ts / H + uk) % 24
        out["seance_or"] = np.where((lon >= 10.5) & (lon < 15), -1.0, np.where((lon >= 15) | (lon < 10.5), 1.0, 0.0))
        am = [_ts(d, 10, 30) - _offsets(d.toordinal() - 719163)[1] * H for d in days if d.weekday() < 5]
        pm = [_ts(d, 15) - _offsets(d.toordinal() - 719163)[1] * H for d in days if d.weekday() < 5]
        out["heures_fixing_lbma_matin"], _m = _event_window(ts, c, atr, am, 4, 4)
        out["heures_fixing_lbma_soir"], out["mouvement_avant_lbma_soir"] = _event_window(ts, c, atr, pm, 5, 4)
    # 5. momentum de la premiere heure de la journee UTC (Shen, Urquhart, Wang ; Elaut et al.)
    mom = np.full(n, NAN)
    day = (ts // 86400).astype(np.int64)
    uniq, start = np.unique(day, return_index=True)
    for k in range(len(uniq)):
        s, e = start[k], (start[k + 1] if k + 1 < len(start) else n)
        if hour[s] == 0 and e - s > 1 and np.isfinite(atr[s]) and atr[s] > 0:
            mom[s + 1:e] = (c[s] - o[s]) / atr[s]
    out["momentum_premiere_heure"] = mom
    out["heures_avant_minuit_utc"] = (24 - hour).astype(float)
    if kind == "cr":
        # echeance mensuelle CME (dernier vendredi, reglement BRR 15-16 h Londres), options Deribit (vendredi 8 h UTC),
        # fenetre de valeur liquidative des ETF (15-16 h New York), seance cash americaine (9 h 30 - 16 h New York)
        lastfri = {}
        for d in days:
            last = datetime.date(d.year + (d.month == 12), d.month % 12 + 1, 1) - datetime.timedelta(days=1)
            lastfri[(d.year, d.month)] = last - datetime.timedelta(days=(last.weekday() - 4) % 7)
        out["jours_avant_echeance_cme"] = np.array([float((lastfri[(d.year, d.month)] - d).days) for d in dts])
        lon = (ts / H + uk) % 24
        out["reglement_cme"] = np.array([1.0 if d == lastfri[(d.year, d.month)] else 0.0 for d in dts]) * \
            ((lon >= 15) & (lon < 16))
        fr8 = [_ts(d, 8) for d in days if d.weekday() == 4]
        rel, _m = _event_window(ts, c, atr, fr8, 48, 12)
        out["heures_echeance_deribit"] = rel
        typ = np.zeros(n)
        for i, d in enumerate(dts):
            nxt = d + datetime.timedelta(days=(4 - d.weekday()) % 7)
            if np.isfinite(rel[i]):
                typ[i] = 1.0 + (nxt == lastfri[(nxt.year, nxt.month)]) + (nxt == lastfri[(nxt.year, nxt.month)]
                                                                            and nxt.month in (3, 6, 9, 12))
        out["type_echeance_deribit"] = typ
        wd = ((ts // 86400 + 3) % 7).astype(int)
        out["fenetre_nav_etf"] = ((ny_h >= 15) & (ny_h < 16) & (wd < 5)).astype(float)
        out["seance_cash_usa"] = ((ny_h >= 9.5) & (ny_h < 16) & (wd < 5)).astype(float)
    # 6. jours feries des grandes places (liquidite faible, faux breakouts)
    hol = {}
    for y in sorted({d.year for d in days}):
        for d, cur in _holidays(y).items():
            hol.setdefault(d, set()).update(cur)
    out["places_fermees"] = np.array([float(len(hol.get(d, ()))) for d in dts])
    if kind == "fx" and sym and len(sym) == 6:
        out["ferie_devise_paire"] = np.array([float(sym[:3] in hol.get(d, ()) or sym[3:] in hol.get(d, ()))
                                              for d in dts])
    return out


def compute_flux(sym, kind, ts, o, h, l, c, atr, refs=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)
    refs = refs or {}
    F = calendrier(ts, c, atr)
    F.update(smt(c, refs.get("bench")))
    F.update(force(sym, kind, refs.get("panier"), len(c)))
    F.update(gaps(ts, o, h, l, c, atr, kind))
    F.update(recherche(sym, kind, ts, o, h, l, c, atr))
    return F
