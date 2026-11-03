"""Specialiste « grands traders » : les regles publiees (livres, interviews, conferences) des meilleurs traders,
reecrites a partir de leurs descriptions publiques, sans code copie. Tout est causal (bougies cloturees seulement).
   - Les Tortues (Richard Dennis, William Eckhardt) : cassures 20 / 55 bougies, sortie sur 10 / 20 bougies ;
   - Linda Raschke : Holy Grail (ADX fort + retour sur l'EMA 20), Turtle Soup (fausse cassure), 80-20, « Anti » ;
   - Toby Crabel : bougies les plus etroites (NR4, NR7), bougie interieure, cassure de la bougie etroite ;
   - Larry Williams : cassure de volatilite (ouverture du jour + part de l'amplitude de la veille), « Oops » ;
   - Larry Connors : Connors RSI (RSI 3 + RSI de la serie + rang du rendement) ;
   - Alexander Elder : systeme d'impulsion, Elder Ray (force des acheteurs / vendeurs), Force Index ;
   - Tom DeMark : decompte TD Setup (9) et TD Countdown (13) ;
   - Bill Williams : Alligator, fractales, Awesome Oscillator, Accelerator ;
   - Victor Sperandeo : regle 1-2-3 et 2B (fausse cassure d'un sommet / creux de swing) ;
   - Mostafa Belkhayate : Timing (prix normalise autour du milieu des 5 dernieres bougies, zones +-4 / +-8) et centre de
     gravite (regression polynomiale + bandes au nombre d'or). L'original se redessine (repaint) : ici il est calcule
     uniquement sur le passe, comme il aurait ete vu en direct."""
import numpy as np

NAN = np.nan
TRADERS = ("Richard Dennis", "William Eckhardt", "Linda Raschke", "Toby Crabel", "Larry Williams",
           "Larry Connors", "Alexander Elder", "Tom DeMark", "Bill Williams", "Victor Sperandeo",
           "Paul Tudor Jones", "Ed Seykota", "Al Brooks", "Mark Minervini", "Stanley Druckenmiller", "Mostafa Belkhayate")


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _ema(x, p):
    out = np.full(len(x), NAN)
    a, s = 2.0 / (p + 1), NAN
    for i, v in enumerate(x):
        if np.isfinite(v):
            s = v if not np.isfinite(s) else s + a * (v - s)
        out[i] = s
    return out


def _wilder(x, p):
    out = np.full(len(x), NAN)
    s, k = NAN, 0
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        k += 1
        s = v if k == 1 else s + (v - s) / p
        if k >= p:
            out[i] = s
    return out


def _sma(x, p):
    out = np.full(len(x), NAN)
    cs = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    out[p - 1:] = (cs[p:] - cs[:-p]) / p
    return out


def _roll(x, p, fn):
    out = np.full(len(x), NAN)
    if len(x) >= p:
        w = np.lib.stride_tricks.sliding_window_view(x, p)
        out[p - 1:] = fn(w, axis=1)
    return out


def _prev(x, k=1):
    return np.r_[np.full(k, NAN), x[:-k]]


def _rsi(x, p):
    d = np.diff(x, prepend=x[0])
    up, dn = _wilder(np.maximum(d, 0), p), _wilder(np.maximum(-d, 0), p)
    return 100 - 100 / (1 + _div(up, dn))


def _adx(h, l, c, p=14):
    tr = np.maximum(h - l, np.maximum(abs(h - _prev(c)), abs(l - _prev(c))))
    upm, dnm = h - _prev(h), _prev(l) - l
    pdm = np.where((upm > dnm) & (upm > 0), upm, 0.0)
    ndm = np.where((dnm > upm) & (dnm > 0), dnm, 0.0)
    atr = _wilder(tr, p)
    pdi, ndi = 100 * _div(_wilder(pdm, p), atr), 100 * _div(_wilder(ndm, p), atr)
    dx = 100 * _div(abs(pdi - ndi), pdi + ndi)
    return _wilder(dx, p), pdi, ndi


def turtles(h, l, c):
    out = {}
    for ent, ex in ((20, 10), (55, 20)):
        hi, lo = _prev(_roll(h, ent, np.max)), _prev(_roll(l, ent, np.min))
        xh, xl = _prev(_roll(h, ex, np.max)), _prev(_roll(l, ex, np.min))
        pos = np.zeros(len(c))
        for i in range(1, len(c)):
            p = pos[i - 1]
            if p == 1 and c[i] < xl[i]:
                p = 0
            elif p == -1 and c[i] > xh[i]:
                p = 0
            if c[i] > hi[i]:
                p = 1
            elif c[i] < lo[i]:
                p = -1
            pos[i] = p
        out["tortue_%d" % ent] = pos
    return out


def raschke(o, h, l, c, atr):
    out = {}
    adx, pdi, ndi = _adx(h, l, c)
    e20 = _ema(c, 20)
    rising = adx > _prev(adx, 3)
    up, dn = (adx > 30) & rising & (pdi > ndi) & (l <= e20), (adx > 30) & rising & (ndi > pdi) & (h >= e20)
    out["holy_grail"] = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
    # Turtle Soup : nouveau plus bas 20 bougies, l'ancien plus bas date d'au moins 4 bougies, cloture revenue au-dessus
    n = len(c)
    soup = np.zeros(n)
    for i in range(21, n):
        wl, wh = l[i - 20:i], h[i - 20:i]
        jl, jh = int(np.argmin(wl)), int(np.argmax(wh))
        if l[i] < wl[jl] and c[i] > wl[jl] and 20 - jl >= 4:
            soup[i] = 1.0
        elif h[i] > wh[jh] and c[i] < wh[jh] and 20 - jh >= 4:
            soup[i] = -1.0
    out["turtle_soup"] = soup
    # 80-20 : la bougie precedente ouvre dans ses 20 % hauts et cloture dans ses 20 % bas, puis le prix revient
    rg = h - l
    po, pc = _div(_prev(o) - _prev(l), _prev(rg)), _div(_prev(c) - _prev(l), _prev(rg))
    b = (po >= 0.8) & (pc <= 0.2) & (l < _prev(l)) & (c > _prev(l))
    s = (po <= 0.2) & (pc >= 0.8) & (h > _prev(h)) & (c < _prev(h))
    out["regle_80_20"] = np.where(b, 1.0, np.where(s, -1.0, 0.0))
    # « Anti » : stochastique lente (16) orientee, la rapide (7) repart dans son sens apres un repli
    ll, hh = _roll(l, 7, np.min), _roll(h, 7, np.max)
    k = 100 * _div(c - ll, hh - ll)
    fast, slow = _sma(k, 10), _sma(_sma(k, 10), 16)
    sl_up = slow > _prev(slow)
    hook_up = (fast > _prev(fast)) & (_prev(fast) < _prev(fast, 2))
    hook_dn = (fast < _prev(fast)) & (_prev(fast) > _prev(fast, 2))
    out["anti_raschke"] = np.where(sl_up & hook_up, 1.0, np.where(~sl_up & hook_dn, -1.0, 0.0))
    out["adx_ema20_ecart"] = np.where(adx > 25, _div(c - e20, atr), NAN)
    return out


def crabel(h, l, c):
    rg = h - l
    out = {"nr4": (rg <= _roll(rg, 4, np.min)).astype(float), "nr7": (rg <= _roll(rg, 7, np.min)).astype(float)}
    inside = (h <= _prev(h)) & (l >= _prev(l))
    out["bougie_interieure"] = inside.astype(float)
    nr7p = _prev(out["nr7"]) == 1
    out["cassure_nr7"] = np.where(nr7p & (c > _prev(h)), 1.0, np.where(nr7p & (c < _prev(l)), -1.0, 0.0))
    return out


def williams_larry(o, h, l, c, days):
    n = len(c)
    brk, oops = np.full(n, NAN), np.zeros(n)
    uniq, start = np.unique(days, return_index=True)
    prev = None                                      # (haut, bas) de la seance precedente
    for d in range(len(uniq)):
        s, e = start[d], (start[d + 1] if d + 1 < len(start) else n)
        if prev is not None and prev[0] > prev[1]:
            ph, pl = prev
            for i in range(s, e):
                brk[i] = (c[i] - o[s]) / (ph - pl)    # > 0,6 : cassure de volatilite haussiere selon Williams
                if o[s] < pl and c[i] > pl:
                    oops[i] = 1.0
                elif o[s] > ph and c[i] < ph:
                    oops[i] = -1.0
        prev = (h[s:e].max(), l[s:e].min())
    return {"cassure_volatilite": brk, "oops_williams": oops}


def connors(c):
    n = len(c)
    streak = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    r1 = np.diff(c, prepend=c[0]) / np.r_[c[0], c[:-1]]
    rank = np.full(n, NAN)
    for i in range(100, n):
        rank[i] = 100 * np.mean(r1[i - 100:i] < r1[i])
    crsi = (_rsi(c, 3) + _rsi(streak, 2) + rank) / 3
    return {"connors_rsi": crsi, "serie_hausse_baisse": streak}


def elder(h, l, c, v, atr):
    e13 = _ema(c, 13)
    hist = (_ema(c, 12) - _ema(c, 26))
    hist = hist - _ema(hist, 9)
    up, dn = (e13 > _prev(e13)) & (hist > _prev(hist)), (e13 < _prev(e13)) & (hist < _prev(hist))
    out = {"impulsion_elder": np.where(up, 1.0, np.where(dn, -1.0, 0.0)),
           "elder_force_acheteurs": _div(h - e13, atr), "elder_force_vendeurs": _div(l - e13, atr)}
    if np.nansum(v) > 0:
        fi = _ema(np.diff(c, prepend=c[0]) * v, 13)
        out["force_index"] = _div(fi, _roll(np.abs(fi), 100, np.mean))
    return out


def demark(h, l, c):
    n = len(c)
    setup, cd = np.zeros(n), np.zeros(n)
    s = k = 0
    for i in range(4, n):
        if c[i] > c[i - 4]:
            s = s + 1 if s > 0 else 1
        elif c[i] < c[i - 4]:
            s = s - 1 if s < 0 else -1
        else:
            s = 0
        setup[i] = max(-9, min(9, s))
        if abs(s) == 9:                              # setup complet : debut du countdown (sens oppose attendu)
            k = 1 if s > 0 else -1
        elif k > 0 and i >= 2 and c[i] >= h[i - 2]:
            k += 1
        elif k < 0 and i >= 2 and c[i] <= l[i - 2]:
            k -= 1
        if abs(k) > 13:
            k = 0
        cd[i] = k
    return {"td_setup": setup, "td_countdown": cd}


def bill_williams(h, l, c, atr):
    med = (h + l) / 2
    jaw, teeth, lips = _prev(_wilder(med, 13), 8), _prev(_wilder(med, 8), 5), _prev(_wilder(med, 5), 3)
    up, dn = (lips > teeth) & (teeth > jaw) & (c > lips), (lips < teeth) & (teeth < jaw) & (c < lips)
    ao = _sma(med, 5) - _sma(med, 34)
    out = {"alligator": np.where(up, 1.0, np.where(dn, -1.0, 0.0)),
           "alligator_ouverture": _div(abs(lips - jaw), atr),
           "awesome_osc": _div(ao, atr), "accelerator_osc": _div(ao - _sma(ao, 5), atr)}
    n = len(c)
    frac, fh, fl = np.zeros(n), NAN, NAN
    for i in range(4, n):                            # fractale confirmee 2 bougies apres son centre
        j = i - 2
        if h[j] == h[j - 2:i + 1].max():
            fh = h[j]
        if l[j] == l[j - 2:i + 1].min():
            fl = l[j]
        frac[i] = 1.0 if np.isfinite(fh) and c[i] > fh else (-1.0 if np.isfinite(fl) and c[i] < fl else 0.0)
    out["cassure_fractale"] = frac
    return out


def sperandeo(h, l, c, k=3):
    """2B : le prix depasse le dernier sommet (creux) de swing puis cloture en dessous (au-dessus) en <= 3 bougies."""
    n = len(c)
    last_h, last_l, tb = NAN, NAN, np.zeros(n)
    hit_h = hit_l = -99
    for i in range(2 * k, n):
        j = i - k
        if h[j] == h[j - k:i + 1].max():
            last_h = h[j]
        if l[j] == l[j - k:i + 1].min():
            last_l = l[j]
        if np.isfinite(last_h) and h[i] > last_h:
            hit_h = i
        if np.isfinite(last_l) and l[i] < last_l:
            hit_l = i
        if i - hit_h <= 3 and c[i] < last_h:
            tb[i] = -1.0
        elif i - hit_l <= 3 and c[i] > last_l:
            tb[i] = 1.0
    return {"regle_2B": tb}


def belkhayate(h, l, c, win=120, deg=2):
    n = len(c)
    med = (h + l) / 2
    mid = _roll(med, 5, np.mean)
    scale = 0.2 * _roll(h - l, 5, np.mean)
    timing = (_div(h - mid, scale) + _div(l - mid, scale) + _div(c - mid, scale)) / 3
    out = {"belkhayate_timing": timing}
    cog, sd = np.full(n, NAN), np.full(n, NAN)
    if n >= win:
        x = np.arange(win, dtype=float) / win
        X = np.vander(x, deg + 1, increasing=True)            # 1, x, x^2
        P = np.linalg.pinv(X)                                  # (deg+1, win)
        W = np.lib.stride_tricks.sliding_window_view(c, win)  # fenetre passee uniquement
        coef = W @ P.T
        fit_last = coef @ X[-1]
        resid = W - coef @ X.T
        cog[win - 1:] = fit_last
        sd[win - 1:] = resid.std(axis=1)
    z = _div(c - cog, sd)
    out["belkhayate_centre_gravite"] = z                       # 0 = centre, +-1,618 / +-2,618 = bandes d'or
    out["belkhayate_pente_centre"] = _div(cog - _prev(cog, 5), sd)
    buy = (timing <= -4) & (timing >= -8) & (z <= -1.618)
    sell = (timing >= 4) & (timing <= 8) & (z >= 1.618)
    out["belkhayate_zone"] = np.where(buy, 1.0, np.where(sell, -1.0, 0.0))
    return out


def compute_traders(o, h, l, c, v, atr, days):
    F = {}
    F.update(turtles(h, l, c))
    F.update(raschke(o, h, l, c, atr))
    F.update(crabel(h, l, c))
    F.update(williams_larry(o, h, l, c, days))
    F.update(connors(c))
    F.update(elder(h, l, c, v, atr))
    F.update(demark(h, l, c))
    F.update(bill_williams(h, l, c, atr))
    F.update(sperandeo(h, l, c))
    F.update(belkhayate(h, l, c))
    return F
