"""Concepts des meilleurs specialistes mondiaux, reecrits a partir de leurs methodes publiques (aucun code copie) :
   - Ichimoku (Goichi Hosoda, Hidenobu Sasaki, Karen Peloille) : nombres de temps de Hosoda, objectifs de prix
     N / V / E, Kijun et Tenkan plates (equilibre), Chikou libre, torsion du nuage (Kumo twist) ;
   - Market / volume profile (Peter Steidlmayer, James Dalton) : POC de la veille non teste (« naked POC »),
     migration de la zone de valeur, regle des 80 %, cassure de l'equilibre initial (initial balance) ;
   - VWAP ancre (Brian Shannon) : depuis le debut de la semaine et depuis le dernier swing ;
   - Wyckoff : spring et upthrust (fausse cassure puis retour), effort contre resultat ;
   - journee de tendance (trend day) : seance qui s'etire, ouverture a un extreme, prix a l'autre.
   Tout est causal : chaque valeur n'utilise que des bougies deja cloturees."""
import numpy as np

NAN = np.nan
HOSODA = (9, 17, 26, 33, 42, 65, 76)


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _mid(h, l, p):
    n = len(h)
    out = np.full(n, NAN)
    for i in range(p - 1, n):
        out[i] = (h[i - p + 1:i + 1].max() + l[i - p + 1:i + 1].min()) / 2
    return out


def _swings(h, l, k=3):
    """Swings confirmes k bougies apres : listes (indice de confirmation, indice du swing, prix, type)."""
    ev = []
    for i in range(2 * k, len(h)):
        j = i - k
        if h[j] == h[j - k:i + 1].max():
            ev.append((i, j, h[j], 1))
        if l[j] == l[j - k:i + 1].min():
            ev.append((i, j, l[j], -1))
    return ev


def ichimoku_masters(h, l, c, atr):
    n = len(c)
    out = {}
    ten, kij, sb = _mid(h, l, 9), _mid(h, l, 26), _mid(h, l, 52)
    fut = np.sign((ten + kij) / 2 - sb)
    # Kijun / Tenkan plates : l'equilibre (aimant du prix selon Hosoda / Sasaki)
    for nm, line in (("kijun", kij), ("tenkan", ten)):
        flat = np.zeros(n)
        for i in range(1, n):
            flat[i] = flat[i - 1] + 1 if np.isfinite(line[i]) and line[i] == line[i - 1] else 0
        out["%s_plate" % nm] = flat
    out["dist_kijun_plate"] = np.where(out["kijun_plate"] >= 5, _div(c - kij, atr), NAN)
    # Chikou libre : la cloture actuelle comparee aux prix d'il y a ~26 bougies (fenetre passee uniquement)
    chk = np.zeros(n)
    for i in range(32, n):
        win_h, win_l = h[i - 31:i - 20].max(), l[i - 31:i - 20].min()
        chk[i] = 1.0 if c[i] > win_h else (-1.0 if c[i] < win_l else 0.0)
    out["chikou_libre"] = chk
    # torsion du nuage (Kumo twist) recente : le nuage futur change de couleur
    tw = np.zeros(n)
    for i in range(3, n):
        if np.isfinite(fut[i]) and np.isfinite(fut[i - 3]) and fut[i] != fut[i - 3] and fut[i] != 0:
            tw[i] = fut[i]
    out["kumo_twist"] = tw
    # theorie du temps de Hosoda + objectifs de prix N / V / E (theorie des vagues)
    ev = _swings(h, l)
    since_h, since_l = np.full(n, NAN), np.full(n, NAN)
    tN, tV, tE, prog = (np.full(n, NAN) for _ in range(4))
    last = {1: None, -1: None}
    seq = []                                     # derniers swings alternes (indice, prix, type)
    e = 0
    for i in range(n):
        while e < len(ev) and ev[e][0] <= i:
            _, j, px, typ = ev[e]
            last[typ] = j
            if seq and seq[-1][2] == typ:
                if (typ == 1 and px >= seq[-1][1]) or (typ == -1 and px <= seq[-1][1]):
                    seq[-1] = (j, px, typ)
            else:
                seq.append((j, px, typ))
                seq = seq[-3:]
            e += 1
        if last[1] is not None:
            since_h[i] = i - last[1]
        if last[-1] is not None:
            since_l[i] = i - last[-1]
        if len(seq) == 3 and np.isfinite(atr[i]) and atr[i] > 0:
            (_, a, ta), (_, b, _tb), (_, cc, _tc) = seq
            d = 1.0 if ta == -1 else -1.0         # A creux -> B sommet -> C creux : objectifs haussiers
            N, V, E = cc + (b - a), b + (b - cc), b + (b - a)
            tN[i], tV[i], tE[i] = (c[i] - N) / atr[i], (c[i] - V) / atr[i], (c[i] - E) / atr[i]
            span = N - cc
            prog[i] = (c[i] - cc) / span if span != 0 else NAN    # 0 = point C, 1 = objectif N atteint
            _ = d
    out["temps_depuis_sommet"], out["temps_depuis_creux"] = since_h, since_l
    near = np.full(n, NAN)
    for i in range(n):
        vals = [x for x in (since_h[i], since_l[i]) if np.isfinite(x)]
        if vals:
            near[i] = min(abs(v - k) for v in vals for k in HOSODA)
    out["temps_hosoda_ecart"] = near
    out["temps_hosoda_cle"] = np.where(np.isfinite(near), (near <= 1).astype(float), NAN)
    out["objectif_N"], out["objectif_V"], out["objectif_E"], out["progression_N"] = tN, tV, tE, prog
    return out


def profile_masters(o, h, l, c, v, atr, days):
    """Steidlmayer / Dalton sur la seance precedente et la seance en cours (jusqu'a la bougie actuelle)."""
    n = len(c)
    has_vol = np.nansum(v) > 0
    uniq, start = np.unique(days, return_index=True)
    prof = []                                    # (poc, vah, val) de chaque seance terminee
    naked, migr, rule80, ib_state, ib_size = (np.full(n, NAN) for _ in range(5))
    for d in range(len(uniq)):
        s, e = start[d], (start[d + 1] if d + 1 < len(start) else n)
        if prof:
            poc, vah, val = prof[-1]
            day_open = o[s]
            hi_so_far, lo_so_far = -np.inf, np.inf
            for i in range(s, e):
                hi_so_far, lo_so_far = max(hi_so_far, h[i]), min(lo_so_far, l[i])
                a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else NAN
                touched = lo_so_far <= poc <= hi_so_far
                naked[i] = NAN if touched else (c[i] - poc) / a
                if len(prof) >= 2:
                    migr[i] = np.sign(prof[-1][0] - prof[-2][0])
                inside = val <= c[i] <= vah
                rule80[i] = (-1.0 if day_open > vah else 1.0 if day_open < val else 0.0) if inside else 0.0
                if i - s >= 2:                   # equilibre initial : les 2 premieres bougies de la seance
                    ibh, ibl = h[s:s + 2].max(), l[s:s + 2].min()
                    ib_state[i] = 1.0 if c[i] > ibh else (-1.0 if c[i] < ibl else 0.0)
                    ib_size[i] = (ibh - ibl) / a
        px, w = c[s:e], v[s:e] if has_vol else np.ones(e - s)
        if len(px) >= 4:
            bins = np.linspace(px.min(), px.max() + 1e-12, 25)
            hist, edges = np.histogram(px, bins=bins, weights=w)
            k = int(np.argmax(hist))
            lo = hi = k
            tot, acc = hist.sum(), hist[k]
            while acc < 0.7 * tot and (lo > 0 or hi < len(hist) - 1):
                left = hist[lo - 1] if lo > 0 else -1
                right = hist[hi + 1] if hi < len(hist) - 1 else -1
                if right >= left:
                    hi += 1
                    acc += hist[hi]
                else:
                    lo -= 1
                    acc += hist[lo]
            mid = (edges[:-1] + edges[1:]) / 2
            prof.append((mid[k], edges[hi + 1], edges[lo]))
    return {"poc_veille_non_teste": naked, "migration_zone_valeur": migr, "regle_80": rule80,
            "equilibre_initial": ib_state, "taille_equilibre_initial": ib_size}


def anchored_vwap(ts, h, l, c, v, atr):
    """VWAP ancre (Brian Shannon) : depuis le debut de la semaine et depuis le dernier swing confirme."""
    n = len(c)
    w = v if np.nansum(v) > 0 else np.ones(n)
    tp = (h + l + c) / 3
    week = ((ts // 86400).astype(np.int64) + 3) // 7
    wk, sw = np.full(n, NAN), np.full(n, NAN)
    num = den = 0.0
    for i in range(n):
        if i == 0 or week[i] != week[i - 1]:
            num = den = 0.0
        num += tp[i] * w[i]
        den += w[i]
        wk[i] = num / den if den > 0 else NAN
    ev = _swings(h, l)
    e, anchor = 0, None
    cum_pv, cum_v = np.cumsum(tp * w), np.cumsum(w)
    for i in range(n):
        while e < len(ev) and ev[e][0] <= i:
            anchor = ev[e][1]
            e += 1
        if anchor is not None:
            pv = cum_pv[i] - (cum_pv[anchor - 1] if anchor > 0 else 0.0)
            vv = cum_v[i] - (cum_v[anchor - 1] if anchor > 0 else 0.0)
            sw[i] = pv / vv if vv > 0 else NAN
    return {"vwap_ancre_semaine": _div(c - wk, atr), "vwap_ancre_swing": _div(c - sw, atr)}


def wyckoff(h, l, c, v, p=20):
    """Spring (fausse cassure par le bas puis cloture au-dessus) / upthrust (l'inverse) ; effort contre resultat."""
    n = len(c)
    spring = np.zeros(n)
    for i in range(p, n):
        lo, hi = l[i - p:i].min(), h[i - p:i].max()
        if l[i] < lo and c[i] > lo:
            spring[i] = 1.0
        elif h[i] > hi and c[i] < hi:
            spring[i] = -1.0
    out = {"spring_upthrust": spring}
    if np.nansum(v) > 0:
        r = np.abs(np.diff(np.log(c), prepend=np.log(c[0])))
        zv, zr = np.full(n, NAN), np.full(n, NAN)
        for i in range(p, n):
            vv, rr = v[i - p:i], r[i - p:i]
            zv[i] = (v[i] - vv.mean()) / (vv.std() + 1e-12)
            zr[i] = (r[i] - rr.mean()) / (rr.std() + 1e-12)
        out["effort_resultat"] = zv - zr                 # beaucoup de volume, peu de mouvement : absorption
    return out


def trend_day(o, h, l, c, days, k=10):
    """Journee de tendance (idee publique QuantRocket « trend day ») : la seance en cours, jusqu'a la bougie actuelle,
       s'etire-t-elle plus que d'habitude, avec l'ouverture a un extreme et le prix a l'autre ?"""
    n = len(c)
    pos, amp, td = np.full(n, NAN), np.full(n, NAN), np.zeros(n)
    uniq, start = np.unique(days, return_index=True)
    ranges = []                                   # amplitudes des seances terminees
    for d in range(len(uniq)):
        s, e = start[d], (start[d + 1] if d + 1 < len(start) else n)
        hi, lo = -np.inf, np.inf
        ref = np.mean(ranges[-k:]) if len(ranges) >= 3 else NAN
        for i in range(s, e):
            hi, lo = max(hi, h[i]), min(lo, l[i])
            rg = hi - lo
            if rg <= 0:
                continue
            pos[i] = (c[i] - lo) / rg
            op = (o[s] - lo) / rg
            if np.isfinite(ref) and ref > 0:
                amp[i] = rg / ref
                if amp[i] >= 1.0:
                    td[i] = 1.0 if (op <= 0.15 and pos[i] >= 0.75) else (-1.0 if (op >= 0.85 and pos[i] <= 0.25) else 0.0)
        ranges.append(h[s:e].max() - l[s:e].min())
    return {"jour_position": pos, "jour_ampleur": amp, "jour_tendance": td}


def compute_masters(ts, o, h, l, c, v, atr, days):
    st = wyckoff(h, l, c, v)
    st.update(trend_day(o, h, l, c, days))
    return {"ichimoku": ichimoku_masters(h, l, c, atr), "profil": profile_masters(o, h, l, c, v, atr, days),
            "vwap": anchored_vwap(ts, h, l, c, v, atr), "structure": st}
