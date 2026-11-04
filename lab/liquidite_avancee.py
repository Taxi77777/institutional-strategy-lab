"""Liquidite tres avancee (recherche approfondie du 4 novembre) — tout est causal, heures de New York avec heure d'ete :
   - Osler (Journal of Finance 2003, NY Fed 2005) : take-profit sur les chiffres ronds (rebond court), stops d'achat
     juste AU-DESSUS des 00 / 50 et stops de vente juste EN DESSOUS (cascade a la cassure, effet >= 2 h) ;
   - Abdi-Ranaldo (RFS 2017) : ecart achat-vente estime depuis les OHLC, decale d'une bougie (sinon fuite du futur),
     normalise par heure de la journee -> regime illiquide ;
   - trous de liquidite (BIS 2017, Banque d'Angleterre 2018, Morris-Shin 2004) : bougie geante pendant les heures
     creuses (21 h - minuit UTC) en regime illiquide ;
   - ICT (pratique, sans preuve academique — le labo tranche) : liquidite interne / externe (IRL / ERL), Power of 3
     et Judas swing + continuation de la cassure de Londres, CBDR / range asiatique et projections, fourchettes IPDA
     20 / 40 / 60 jours (premium / discount), NWOG (gap d'ouverture de la semaine) et son milieu, rejection blocks."""
import numpy as np

from lab.flux import _clock

NAN = np.nan
H = 3600


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _step(c):
    """Pas des chiffres ronds « 00 » selon le prix (EURUSD 0,01 ; USDJPY 1 ; or 10 ; BTC 1000...)."""
    m = np.nanmedian(c)
    if m < 10:
        return 0.01                                  # EURUSD 1,1000 / 1,1100
    if m < 1000:
        return 1.0                                   # USDJPY 150,00 / argent
    return 10 ** np.floor(np.log10(m)) / 10          # or 4000 / 4100, BTC 77000 / 78000


def osler(h, l, c, atr):
    n = len(c)
    st = _step(c)
    out = {}
    for nm, s in (("00", st), ("50", st / 2)):
        lvl_prev = np.floor(np.r_[c[0], c[:-1]] / s)
        lvl_now = np.floor(c / s)
        buf = 0.1 * np.nan_to_num(atr)
        up = (lvl_now > lvl_prev) & (c - lvl_now * s >= buf)          # cloture au-dessus d'un rond, avec tampon
        dn = (lvl_now < lvl_prev) & ((lvl_now + 1) * s - c >= buf)
        out["cassure_rond_%s" % nm] = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
        r_hi = np.ceil(c / s) * s
        r_lo = np.floor(c / s) * s
        rej_dn = (h >= r_hi) & (c < r_hi)                               # meche sur un rond, cloture en deca
        rej_up = (l <= r_lo) & (c > r_lo)
        out["rejet_rond_%s" % nm] = np.where(rej_up & ~rej_dn, 1.0, np.where(rej_dn & ~rej_up, -1.0, 0.0))
        # cote des stops (Osler) : zone de stops d'achat = juste au-dessus du rond (1 a 10 % du pas)
        frac = (c - r_lo) / s
        out["zone_stops_rond_%s" % nm] = np.where(frac <= 0.10, 1.0, np.where(frac >= 0.90, -1.0, 0.0))
    # continuation attendue >= 2 h apres une cassure de rond « 00 »
    cas = out["cassure_rond_00"]
    rec = np.zeros(n)
    for i in range(n):
        w = cas[max(0, i - 2):i + 1]
        nz = w[w != 0]
        rec[i] = nz[-1] if len(nz) else 0.0
    out["cassure_rond_recente"] = rec
    return out


def abdi_ranaldo(ts, h, l, c, w=24):
    """Ecart estime S (Abdi-Ranaldo), utilisable a la cloture suivante (decale d'une bougie), z-score par heure."""
    n = len(c)
    eta = (np.log(h) + np.log(l)) / 2
    lc = np.log(c)
    prod = np.full(n, NAN)
    prod[:-1] = (lc[:-1] - eta[:-1]) * (lc[:-1] - eta[1:])           # utilise la bougie t+1 -> on decale apres
    s2 = np.full(n, NAN)
    for i in range(w, n):
        seg = prod[i - w:i - 1]                                         # seulement des termes deja connus en i
        seg = seg[np.isfinite(seg)]
        if len(seg) >= w // 2:
            s2[i] = 4 * seg.mean()
    S = np.sqrt(np.maximum(s2, 0))
    hour = ((ts // H) % 24).astype(int)
    z = np.full(n, NAN)
    hist = {}
    for i in range(n):
        q = hist.setdefault(hour[i], [])
        if len(q) >= 10 and np.isfinite(S[i]):
            z[i] = (S[i] - np.mean(q)) / (np.std(q) + 1e-12)
        if np.isfinite(S[i]):
            q.append(S[i])
            if len(q) > 20:
                q.pop(0)
    return {"ecart_abdi_ranaldo": S, "illiquidite_heure_z": z}


def trou_liquidite(ts, h, l, c, z):
    n = len(c)
    hour = ((ts // H) % 24).astype(int)
    rng = h - l
    rel = np.full(n, NAN)
    hist = {}
    for i in range(n):
        q = hist.setdefault(hour[i], [])
        if len(q) >= 10:
            rel[i] = rng[i] / (np.mean(q) + 1e-12)
        q.append(rng[i])
        if len(q) > 20:
            q.pop(0)
    creuse = hour >= 21
    ev = (rel > 3) & (np.nan_to_num(z) > 2) & creuse
    mid = (h + l) / 2
    return {"trou_liquidite": np.where(ev, np.sign(c - mid) * -1.0, 0.0), "range_vs_heure": rel}


def ict_avance(ts, o, h, l, c, atr, k=3):
    n = len(c)
    out = {}
    day, ny, _uk = _clock(ts)
    ny_h = (ts / H + ny) % 24
    ny_day = ((ts + ny * H) // 86400).astype(np.int64)
    # --- IRL / ERL : range de travail = derniers swings confirmes
    sh = sl = NAN
    pos, ext = np.full(n, NAN), np.zeros(n)
    for i in range(2 * k, n):
        j = i - k
        if h[j] == h[j - k:i + 1].max():
            sh = h[j]
        if l[j] == l[j - k:i + 1].min():
            sl = l[j]
        if np.isfinite(sh) and np.isfinite(sl) and sh > sl:
            pos[i] = (c[i] - sl) / (sh - sl)
            if l[i] < sl and c[i] > sl:
                ext[i] = 1.0                    # balayage de la liquidite externe basse -> cible interne puis haute
            elif h[i] > sh and c[i] < sh:
                ext[i] = -1.0
    out["position_range_travail"] = pos
    out["balayage_erl"] = ext
    # --- Power of 3 journalier (heure de NY) : ouverture de minuit, manipulation 0 h - 5 h, Londres / Asie
    mo, asia_hi, asia_lo = np.full(n, NAN), np.full(n, NAN), np.full(n, NAN)
    judas, lon_brk = np.zeros(n), np.zeros(n)
    uniq, start = np.unique(ny_day, return_index=True)
    for d in range(len(uniq)):
        s, e = start[d], (start[d + 1] if d + 1 < len(start) else n)
        ahi, alo, man_lo, man_hi = -np.inf, np.inf, np.inf, -np.inf
        o_mid = o[s]
        for i in range(s, e):
            mo[i] = o_mid
            if ny_h[i] < 3:                                  # fin du range asiatique (minuit - 3 h NY)
                ahi, alo = max(ahi, h[i]), min(alo, l[i])
            elif np.isfinite(ahi):
                asia_hi[i], asia_lo[i] = ahi, alo
            if ny_h[i] < 5:
                man_lo, man_hi = min(man_lo, l[i]), max(man_hi, h[i])
            else:
                # Judas : manipulation sous l'ouverture de minuit puis retour au-dessus (biais haussier)
                if man_lo < o_mid and c[i] > o_mid and man_hi <= o_mid + (o_mid - man_lo):
                    judas[i] = 1.0
                elif man_hi > o_mid and c[i] < o_mid:
                    judas[i] = -1.0
            if 3 <= ny_h[i] < 8 and np.isfinite(ahi):         # cassure de Londres du range asiatique (continue 54 %)
                lon_brk[i] = 1.0 if c[i] > ahi else (-1.0 if c[i] < alo else 0.0)
    out["vs_ouverture_minuit_ny"] = _div(c - mo, atr)
    out["judas_swing"] = judas
    out["cassure_londres_asie"] = lon_brk
    # --- CBDR (14 h - 20 h NY, corps) et projections +-1 / 2 / 3 x
    cb_lo, cb_hi = np.full(n, NAN), np.full(n, NAN)
    cur_lo = cur_hi = None
    last = None
    for i in range(n):
        if 14 <= ny_h[i] < 20:
            bl, bh = min(o[i], c[i]), max(o[i], c[i])
            cur_lo = bl if cur_lo is None else min(cur_lo, bl)
            cur_hi = bh if cur_hi is None else max(cur_hi, bh)
        elif cur_lo is not None and ny_h[i] >= 20:
            last, cur_lo, cur_hi = (cur_lo, cur_hi), None, None
        if last is not None and not (14 <= ny_h[i] < 20):
            cb_lo[i], cb_hi[i] = last
    R = cb_hi - cb_lo
    proj = np.where(c > cb_hi, (c - cb_hi) / R, np.where(c < cb_lo, -(cb_lo - c) / R, 0.0))
    out["projection_cbdr"] = np.where(np.isfinite(R) & (R > 0), proj, NAN)
    out["taille_cbdr"] = _div(R, atr)
    # --- IPDA 20 / 40 / 60 jours (bougies 1 h -> 24 par jour de cotation)
    for dd in (20, 40, 60):
        w = dd * 24
        hi, lo = np.full(n, NAN), np.full(n, NAN)
        if n > w:
            from numpy.lib.stride_tricks import sliding_window_view as sw
            hi[w:] = sw(h[:-1], w).max(axis=1)[:n - w]
            lo[w:] = sw(l[:-1], w).min(axis=1)[:n - w]
        out["ipda_%dj_position" % dd] = _div(c - lo, hi - lo)          # < 0,5 discount, > 0,5 premium
        out["ipda_%dj_dist_haut" % dd] = _div(hi - c, atr)
        out["ipda_%dj_dist_bas" % dd] = _div(c - lo, atr)
    # --- NWOG : gap d'ouverture de la semaine (forex : trou du week-end) et son milieu
    gap_mid, gap_open = np.full(n, NAN), np.zeros(n)
    cur = None
    for i in range(1, n):
        if ts[i] - ts[i - 1] > 6 * H:
            a_, b_ = c[i - 1], o[i]
            cur = (min(a_, b_), max(a_, b_))
        if cur is not None:
            gap_mid[i] = (cur[0] + cur[1]) / 2
            gap_open[i] = 1.0 if l[i] > cur[1] else (-1.0 if h[i] < cur[0] else 0.0)
    out["vs_milieu_nwog"] = _div(c - gap_mid, atr)
    out["nwog_position"] = gap_open
    # --- rejection block : swing a longue meche, retour sur le corps extreme et rejet
    rb = np.zeros(n)
    zone = {1: None, -1: None}
    for i in range(2 * k, n):
        j = i - k
        body_lo, body_hi = min(o[j], c[j]), max(o[j], c[j])
        rngj = h[j] - l[j]
        if rngj > 0 and l[j] == l[j - k:i + 1].min() and (body_lo - l[j]) / rngj > 0.5:
            zone[1] = (l[j], body_lo)
        if rngj > 0 and h[j] == h[j - k:i + 1].max() and (h[j] - body_hi) / rngj > 0.5:
            zone[-1] = (body_hi, h[j])
        z1, z2 = zone[1], zone[-1]
        if z1 and l[i] <= z1[1] and c[i] > z1[1] and l[i] >= z1[0]:
            rb[i] = 1.0
        elif z2 and h[i] >= z2[0] and c[i] < z2[0] and h[i] <= z2[1]:
            rb[i] = -1.0
    out["rejection_block"] = rb
    return out


def compute_liquidite_avancee(ts, o, h, l, c, atr):
    F = osler(h, l, c, atr)
    ar = abdi_ranaldo(ts, h, l, c)
    F.update(ar)
    F.update(trou_liquidite(ts, h, l, c, ar["illiquidite_heure_z"]))
    F.update(ict_avance(ts, o, h, l, c, atr))
    return F
