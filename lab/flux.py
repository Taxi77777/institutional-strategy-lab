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


def compute_flux(sym, kind, ts, o, h, l, c, atr, refs=None):
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=__name__)
    refs = refs or {}
    F = calendrier(ts, c, atr)
    F.update(smt(c, refs.get("bench")))
    F.update(force(sym, kind, refs.get("panier"), len(c)))
    F.update(gaps(ts, o, h, l, c, atr, kind))
    return F
