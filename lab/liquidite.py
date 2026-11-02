"""Specialiste liquidite : ou sont les stops des autres traders, et quand le marche est liquide ou non.
   Sources publiques : niveaux de liquidite des desks (plus haut / bas de la veille, de la semaine, du mois),
   chasses de stops (balayage puis retour), zones de stops empiles au-dessus des sommets et sous les creux non
   touches ; mesures academiques de liquidite : ecart achat-vente estime de Corwin et Schultz (2012), mesure de Roll
   (1984), illiquidite d'Amihud (2002), activite par rapport a la meme heure habituelle ;
   liquidite des operateurs : plus haut / bas des sessions Asie, Londres, New York precedentes (ou les desks
   chassent les stops), ouvertures institutionnelles (minuit New York, jour, semaine, mois), niveaux de liquidation
   estimes des positions a levier (crypto 10x a 100x, forex 50x a 500x, facon carte des liquidations), pression
   acheteuse / vendeuse estimee bougie par bougie (delta cumule estime) et sa divergence avec le prix.
   Tout est causal : chaque valeur n'utilise que des bougies deja cloturees."""
import numpy as np

NAN = np.nan


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _periods(ts):
    day = (ts // 86400).astype(np.int64)
    week = (day + 3) // 7
    month = ts.astype("int64").astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    return {"veille": day, "semaine": week, "mois": month}


def _prev_levels(pid, h, l):
    """Plus haut / bas de la periode precedente terminee, et plus haut / bas de la periode en cours jusqu'ici."""
    n = len(h)
    ph, pl, ch, cl = (np.full(n, NAN) for _ in range(4))
    uniq, start = np.unique(pid, return_index=True)
    last = None
    for d in range(len(uniq)):
        s, e = start[d], (start[d + 1] if d + 1 < len(start) else n)
        ch[s:e], cl[s:e] = np.maximum.accumulate(h[s:e]), np.minimum.accumulate(l[s:e])
        if last is not None:
            ph[s:e], pl[s:e] = last
        last = (h[s:e].max(), l[s:e].min())
    return ph, pl, ch, cl


def niveaux(ts, h, l, c, atr):
    out = {}
    for nm, pid in _periods(ts).items():
        ph, pl, ch, cl = _prev_levels(pid, h, l)
        out["dist_haut_%s" % nm] = _div(c - ph, atr)
        out["dist_bas_%s" % nm] = _div(c - pl, atr)
        taken_h, taken_l = ch > ph, cl < pl
        out["haut_%s_pris" % nm] = np.where(np.isfinite(ph), taken_h.astype(float), NAN)
        out["bas_%s_pris" % nm] = np.where(np.isfinite(pl), taken_l.astype(float), NAN)
        # chasse de stops : le niveau a ete depasse dans la periode, mais le prix cloture de nouveau de l'autre cote
        out["chasse_%s" % nm] = np.where(taken_l & (c > pl) & ~taken_h, 1.0,
                                         np.where(taken_h & (c < ph) & ~taken_l, -1.0, 0.0))
    return out


def stops_empiles(h, l, c, atr, k=3, look=200):
    """Nombre de sommets (creux) de swing encore intacts a moins de 1 et 2 ATR au-dessus (en dessous) :
       les stops s'y accumulent et attirent le prix. Ecart a la plus proche reserve de liquidite intacte."""
    n = len(c)
    up1, dn1, up2, dn2, near_up, near_dn = (np.full(n, NAN) for _ in range(6))
    highs, lows = [], []                         # (indice, prix) des swings encore intacts
    for i in range(2 * k, n):
        j = i - k
        if h[j] == h[j - k:i + 1].max():
            highs.append((j, h[j]))
        if l[j] == l[j - k:i + 1].min():
            lows.append((j, l[j]))
        highs = [(a, p) for a, p in highs if p > h[i] and i - a <= look]   # depasse -> liquidite prise
        lows = [(a, p) for a, p in lows if p < l[i] and i - a <= look]
        a_ = atr[i]
        if not (np.isfinite(a_) and a_ > 0):
            continue
        du = np.array([p - c[i] for _, p in highs]) / a_
        dd = np.array([c[i] - p for _, p in lows]) / a_
        up1[i], up2[i] = float((du <= 1).sum()), float((du <= 2).sum())
        dn1[i], dn2[i] = float((dd <= 1).sum()), float((dd <= 2).sum())
        near_up[i] = du.min() if len(du) else NAN
        near_dn[i] = dd.min() if len(dd) else NAN
    return {"stops_au_dessus_1atr": up1, "stops_en_dessous_1atr": dn1, "stops_au_dessus_2atr": up2,
            "stops_en_dessous_2atr": dn2, "reserve_haute_proche": near_up, "reserve_basse_proche": near_dn,
            "desequilibre_stops": _div(up2 - dn2, up2 + dn2 + 1)}


def mesures(ts, h, l, c, v, w=24):
    n = len(c)
    out = {}
    # ecart achat-vente de Corwin-Schultz (a partir des plus hauts / bas de 2 bougies)
    lh = np.log(h / l) ** 2
    beta = lh + np.r_[NAN, lh[:-1]]
    hh, ll = np.maximum(h, np.r_[NAN, h[:-1]]), np.minimum(l, np.r_[NAN, l[:-1]])
    gamma = np.log(hh / ll) ** 2
    k = 3 - 2 * np.sqrt(2)
    with np.errstate(invalid="ignore"):
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        cs = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    cs = np.where(cs < 0, 0.0, cs)
    out["ecart_corwin_schultz"] = _roll_mean(cs, w)
    # mesure de Roll : covariance negative des rendements successifs -> ecart effectif
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    cov = np.full(n, NAN)
    for i in range(2 * w, n):
        a, b = r[i - 2 * w + 1:i + 1], r[i - 2 * w:i]
        cov[i] = np.mean((a - a.mean()) * (b - b.mean()))
    out["ecart_roll"] = np.where(np.isfinite(cov), 2 * np.sqrt(np.maximum(-cov, 0)), NAN)
    # activite par rapport a la meme heure des 20 derniers jours (liquidite habituelle de cette heure)
    hour = ((ts // 3600) % 24).astype(int)
    rg = h - l
    out["amplitude_vs_heure"] = _vs_same_hour(rg, hour)
    if np.nansum(v) > 0:
        out["volume_vs_heure"] = _vs_same_hour(v, hour)
        ami = _div(np.abs(r), v * c)
        m = _roll_mean(ami, w)
        out["illiquidite_amihud"] = _div(m, _roll_median(m, 500))
    return out


def _roll_mean(x, p):
    out = np.full(len(x), NAN)
    for i in range(p - 1, len(x)):
        seg = x[i - p + 1:i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= p // 2:
            out[i] = seg.mean()
    return out


def _roll_median(x, p):
    out = np.full(len(x), NAN)
    for i in range(50, len(x)):
        seg = x[max(0, i - p + 1):i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 50:
            out[i] = np.median(seg)
    return out


def _vs_same_hour(x, hour, days=20):
    """x divise par la moyenne de x a la meme heure sur les `days` jours precedents (sans la bougie actuelle)."""
    n = len(x)
    out = np.full(n, NAN)
    hist = {}
    for i in range(n):
        q = hist.setdefault(hour[i], [])
        if len(q) >= 5:
            m = np.mean(q)
            out[i] = x[i] / m if m > 0 else NAN
        if np.isfinite(x[i]):
            q.append(x[i])
            if len(q) > days:
                q.pop(0)
    return out


SESSIONS = {"asie": (0, 7), "londres": (7, 13), "new_york": (13, 21)}     # heures UTC


def sessions(ts, h, l, c, atr):
    """Plus haut / bas de la derniere session terminee de chaque place ; prise de ces stops puis retour."""
    n = len(c)
    hour = ((ts // 3600) % 24).astype(int)
    day = (ts // 86400).astype(np.int64)
    out = {}
    for nm, (a, b) in SESSIONS.items():
        dh, dl, sw = np.full(n, NAN), np.full(n, NAN), np.zeros(n)
        last, cur, cur_day = None, None, None
        run_h, run_l = -np.inf, np.inf                # extremes depuis la fin de la derniere session
        for i in range(n):
            inside = a <= hour[i] < b
            if inside:
                if cur is None or cur_day != day[i]:
                    cur, cur_day = [h[i], l[i]], day[i]
                else:
                    cur[0], cur[1] = max(cur[0], h[i]), min(cur[1], l[i])
            elif cur is not None:
                last, cur = tuple(cur), None
                run_h, run_l = -np.inf, np.inf
            if last is not None and not inside:
                run_h, run_l = max(run_h, h[i]), min(run_l, l[i])
                if np.isfinite(atr[i]) and atr[i] > 0:
                    dh[i], dl[i] = (c[i] - last[0]) / atr[i], (c[i] - last[1]) / atr[i]
                if run_l < last[1] and c[i] > last[1]:
                    sw[i] = 1.0
                elif run_h > last[0] and c[i] < last[0]:
                    sw[i] = -1.0
        out["dist_haut_%s" % nm], out["dist_bas_%s" % nm], out["chasse_%s" % nm] = dh, dl, sw
    return out


def ouvertures(ts, o, c, atr):
    """Prix d'ouverture de reference des desks : minuit New York (ICT), jour UTC, semaine, mois."""
    n = len(c)
    ny = ((ts - 5 * 3600) // 86400).astype(np.int64)
    per = _periods(ts)
    out = {}
    for nm, pid in (("minuit_new_york", ny), ("jour", per["veille"]), ("semaine", per["semaine"]),
                    ("mois", per["mois"])):
        op = np.full(n, NAN)
        uniq, start = np.unique(pid, return_index=True)
        for d in range(len(uniq)):
            s, e = start[d], (start[d + 1] if d + 1 < len(start) else n)
            op[s:e] = o[s]
        out["vs_ouverture_%s" % nm] = _div(c - op, atr)
    return out


def liquidations(h, l, c, atr, k=3, look=300, levs=(10, 25, 50, 100)):
    """Niveaux de liquidation / appel de marge estimes (crypto 10x-100x, forex 50x-500x). Les positions ouvertes aux derniers sommets / creux de swing avec un
       levier L sont liquidees a environ prix x (1 -/+ 1/L). Distance aux grappes les plus proches."""
    n = len(c)
    near_l, near_s, dens_l, dens_s = (np.full(n, NAN) for _ in range(4))
    ent = []                                      # (indice, prix) des points d'entree probables (swings)
    for i in range(2 * k, n):
        j = i - k
        if h[j] == h[j - k:i + 1].max() or l[j] == l[j - k:i + 1].min():
            ent.append((j, (h[j] + l[j]) / 2))
        ent = [(a, p) for a, p in ent if i - a <= look]
        if not ent or not (np.isfinite(atr[i]) and atr[i] > 0):
            continue
        px = np.array([p for _, p in ent])
        longs = np.concatenate([px * (1 - 1.0 / L) for L in levs])       # liquidations des acheteurs (sous le prix)
        shorts = np.concatenate([px * (1 + 1.0 / L) for L in levs])      # liquidations des vendeurs (au-dessus)
        lo, hi = longs[longs < c[i]], shorts[shorts > c[i]]
        near_l[i] = (c[i] - lo.max()) / atr[i] if len(lo) else NAN
        near_s[i] = (hi.min() - c[i]) / atr[i] if len(hi) else NAN
        dens_l[i] = float(((c[i] - lo) <= 2 * atr[i]).sum())
        dens_s[i] = float(((hi - c[i]) <= 2 * atr[i]).sum())
    return {"liquidations_acheteurs_proches": near_l, "liquidations_vendeurs_proches": near_s,
            "grappe_liquidations_dessous": dens_l, "grappe_liquidations_dessus": dens_s,
            "aimant_liquidations": _div(dens_s - dens_l, dens_s + dens_l + 1)}


def delta_estime(h, l, c, v, w=24):
    """Pression acheteuse / vendeuse estimee : volume x position de la cloture dans la bougie (sans donnees tick),
       cumul glissant et divergence avec le prix (le prix monte mais les vendeurs dominent : distribution)."""
    rg = h - l
    pos = np.where(rg > 0, (2 * c - h - l) / np.where(rg > 0, rg, 1), 0.0)
    d = pos * v
    cs = np.cumsum(np.r_[0.0, d])
    n = len(c)
    cum = np.full(n, NAN)
    cum[w - 1:] = cs[w:] - cs[:-w]
    vs = np.cumsum(np.r_[0.0, v])
    vol = np.full(n, NAN)
    vol[w - 1:] = vs[w:] - vs[:-w]
    ratio = _div(cum, vol)
    ret = np.full(n, NAN)
    ret[w:] = np.log(c[w:] / c[:-w])
    div = np.where(np.isfinite(ratio) & np.isfinite(ret), np.sign(ret) - np.sign(ratio), NAN) / 2
    return {"delta_estime": ratio, "divergence_delta": div}


def compute_liquidite(ts, h, l, c, v, atr, o=None, kind="fx"):
    F = niveaux(ts, h, l, c, atr)
    F.update(stops_empiles(h, l, c, atr))
    F.update(mesures(ts, h, l, c, v))
    F.update(sessions(ts, h, l, c, atr))
    if o is not None:
        F.update(ouvertures(ts, o, c, atr))
    F.update(liquidations(h, l, c, atr, levs=(10, 25, 50, 100) if kind == "cr" else (50, 100, 200, 500)))
    if np.nansum(v) > 0:
        F.update(delta_estime(h, l, c, v))
    return F
