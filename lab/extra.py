"""Specialistes ajoutes a partir de sources publiques reconnues (formules reecrites, aucun code copie) :
   - chandeliers : les 61 figures de chandeliers de TA-Lib ;
   - alpha158    : facteurs Alpha158 de Microsoft Qlib (version sans volume si l'actif n'en a pas) ;
   - wq101       : formules des « 101 Formulaic Alphas » (WorldQuant, 2015), version serie temporelle ;
   - smc         : Smart Money Concepts (swings confirmes, BOS / CHoCH, order blocks, FVG, liquidite).
   Tout est causal : chaque valeur n'utilise que les bougies deja cloturees."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

try:
    import talib
except Exception:                     # TA-Lib absent : les chandeliers sont simplement ignores
    talib = None

NAN = np.nan
WINDOWS = (5, 10, 20, 30, 60)


def _roll(x, p, fn):
    out = np.full(len(x), NAN)
    if len(x) >= p:
        out[p - 1:] = fn(swv(x, p), axis=1)
    return out


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _shift(x, k):
    out = np.full(len(x), NAN)
    out[k:] = x[:-k]
    return out


def _sma(x, p):
    return _roll(x, p, np.mean)


def _ts_rank(x, p):
    """Rang (0..1) de la valeur actuelle dans sa fenetre."""
    return _roll(np.nan_to_num(x), p, lambda w, axis: (w <= w[:, -1:]).mean(axis=axis))


def _corr(x, y, p):
    out = np.full(len(x), NAN)
    if len(x) < p:
        return out
    X, Y = swv(np.nan_to_num(x), p), swv(np.nan_to_num(y), p)
    xm, ym = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (xm * ym).sum(1) / np.sqrt((xm ** 2).sum(1) * (ym ** 2).sum(1))
    out[p - 1:] = r
    out[~np.isfinite(out)] = NAN
    return out


def _decay(x, p):
    w = np.arange(1, p + 1, dtype=float)
    w /= w.sum()
    return _roll(np.nan_to_num(x), p, lambda a, axis: (a * w).sum(axis=axis))


# ------------------------------------------------------------------ chandeliers (TA-Lib)
def candles(o, h, l, c):
    out = {}
    if talib is None:
        return out
    oo, hh, ll, cc = (np.ascontiguousarray(x, dtype=float) for x in (o, h, l, c))
    bull = np.zeros(len(c))
    bear = np.zeros(len(c))
    for fn in talib.get_function_groups()["Pattern Recognition"]:
        try:
            v = getattr(talib, fn)(oo, hh, ll, cc).astype(float) / 100.0
        except Exception:
            continue
        if not np.any(v):
            continue
        out[fn.lower()[3:]] = np.clip(v, -1, 1)
        bull += v > 0
        bear += v < 0
    out["haussieres_5"] = _roll(bull, 5, np.sum)
    out["baissieres_5"] = _roll(bear, 5, np.sum)
    out["solde_5"] = out["haussieres_5"] - out["baissieres_5"]
    return out


# ------------------------------------------------------------------ Alpha158 (Microsoft Qlib)
def alpha158(o, h, l, c, v, has_vol):
    out = {}
    rng = h - l + 1e-12
    out["kmid"] = _div(c - o, o)
    out["klen"] = _div(h - l, o)
    out["kmid2"] = (c - o) / rng
    out["kup"] = _div(h - np.maximum(o, c), o)
    out["kup2"] = (h - np.maximum(o, c)) / rng
    out["klow"] = _div(np.minimum(o, c) - l, o)
    out["klow2"] = (np.minimum(o, c) - l) / rng
    out["ksft"] = _div(2 * c - h - l, o)
    out["ksft2"] = (2 * c - h - l) / rng
    out["open0"], out["high0"], out["low0"] = _div(o, c), _div(h, c), _div(l, c)
    d = np.diff(c, prepend=c[0])
    up, dn = (d > 0).astype(float), (d < 0).astype(float)
    gain, loss, ad = np.clip(d, 0, None), np.clip(-d, 0, None), np.abs(d)
    lv = np.log(v + 1) if has_vol else None
    for p in WINDOWS:
        idx = np.arange(p, dtype=float)
        idx_c = idx - idx.mean()
        out["roc%d" % p] = _div(_shift(c, p), c)
        out["ma%d" % p] = _div(_sma(c, p), c)
        out["std%d" % p] = _div(_roll(c, p, np.std), c)
        slope = _roll(c, p, lambda w, axis: (w * idx_c).sum(axis=axis) / (idx_c ** 2).sum())
        out["beta%d" % p] = _div(slope, c)
        r2 = _corr(c, np.arange(len(c), dtype=float), p) ** 2
        out["rsqr%d" % p] = r2
        fit_last = _sma(c, p) + slope * idx_c[-1]
        out["resi%d" % p] = _div(c - fit_last, c)
        mx, mn = _roll(h, p, np.max), _roll(l, p, np.min)
        out["max%d" % p], out["min%d" % p] = _div(mx, c), _div(mn, c)
        out["qtlu%d" % p] = _div(_roll(c, p, lambda w, axis: np.quantile(w, 0.8, axis=axis)), c)
        out["qtld%d" % p] = _div(_roll(c, p, lambda w, axis: np.quantile(w, 0.2, axis=axis)), c)
        out["rank%d" % p] = _ts_rank(c, p)
        out["rsv%d" % p] = _div(c - mn, mx - mn + 1e-12)
        imax = _roll(h, p, lambda w, axis: np.argmax(w, axis=axis)) / p
        imin = _roll(l, p, lambda w, axis: np.argmin(w, axis=axis)) / p
        out["imax%d" % p], out["imin%d" % p], out["imxd%d" % p] = imax, imin, imax - imin
        out["cntp%d" % p], out["cntn%d" % p] = _sma(up, p), _sma(dn, p)
        out["cntd%d" % p] = out["cntp%d" % p] - out["cntn%d" % p]
        sa = _roll(ad, p, np.sum) + 1e-12
        out["sump%d" % p], out["sumn%d" % p] = _roll(gain, p, np.sum) / sa, _roll(loss, p, np.sum) / sa
        out["sumd%d" % p] = out["sump%d" % p] - out["sumn%d" % p]
        if has_vol:
            out["corr%d" % p] = _corr(c, lv, p)
            out["cord%d" % p] = _corr(_div(c, _shift(c, 1)), np.diff(lv, prepend=lv[0]), p)
            out["vma%d" % p] = _div(_sma(v, p), v + 1e-12)
            out["vstd%d" % p] = _div(_roll(v, p, np.std), v + 1e-12)
            vc = np.abs(_div(c, _shift(c, 1)) - 1) * v
            out["wvma%d" % p] = _div(_roll(np.nan_to_num(vc), p, np.std), _roll(np.nan_to_num(vc), p, np.mean))
            dv = np.diff(v, prepend=v[0])
            sv = _roll(np.abs(dv), p, np.sum) + 1e-12
            out["vsump%d" % p] = _roll(np.clip(dv, 0, None), p, np.sum) / sv
            out["vsumn%d" % p] = _roll(np.clip(-dv, 0, None), p, np.sum) / sv
            out["vsumd%d" % p] = out["vsump%d" % p] - out["vsumn%d" % p]
    return out


# ------------------------------------------------------------------ 101 Formulaic Alphas (WorldQuant)
def wq101(o, h, l, c, v, has_vol, atr):
    """Version serie temporelle : le « rank » transversal (entre actions) devient le rang dans le temps."""
    out = {}
    d1 = np.diff(c, prepend=c[0])
    ret = _div(d1, _shift(c, 1))
    vwap = (h + l + c) / 3.0
    mn5, mx5 = _roll(np.nan_to_num(d1), 5, np.min), _roll(np.nan_to_num(d1), 5, np.max)
    out["a009"] = np.where(mn5 > 0, d1, np.where(mx5 < 0, d1, -d1)) / (atr + 1e-12)
    mn4, mx4 = _roll(np.nan_to_num(d1), 4, np.min), _roll(np.nan_to_num(d1), 4, np.max)
    out["a010"] = _ts_rank(np.where(mn4 > 0, d1, np.where(mx4 < 0, d1, -d1)), 20)
    out["a004"] = -_ts_rank(l, 9)
    d7 = c - _shift(c, 7)
    out["a019"] = -np.sign(d7 + d7) * (1 + _ts_rank(_roll(np.nan_to_num(ret), 60, np.sum), 60))
    out["a023"] = np.where(_sma(h, 20) < h, -(h - _shift(h, 2)) / (atr + 1e-12), 0.0)
    cond = _div(_sma(c, 100) - _shift(_sma(c, 100), 100), _shift(c, 100)) <= 0.05
    out["a024"] = np.where(cond, -(c - _roll(c, 100, np.min)), -(c - _shift(c, 3))) / (atr + 1e-12)
    out["a033"] = _ts_rank(-(1 - _div(o, c)), 20)
    out["a038"] = -_ts_rank(c, 10) * _div(c, o)
    out["a041"] = _div(np.sqrt(h * l) - vwap, atr)
    out["a042"] = _div(vwap - c, vwap + c)
    s20 = _div(_shift(c, 20) - _shift(c, 10), np.full(len(c), 10.0))
    s10 = _div(_shift(c, 10) - c, np.full(len(c), 10.0))
    out["a046"] = _div(s20 - s10, atr)
    out["a049"] = np.where(s20 - s10 < -0.1 * atr, 1.0, -(c - _shift(c, 1)) / (atr + 1e-12))
    cl = c - l
    out["a053"] = -(_div((cl - (h - c)), cl + 1e-12) - _shift(_div((cl - (h - c)), cl + 1e-12), 9))
    out["a054"] = _div(-(l - c) * o ** 5, (l - h - 1e-12) * c ** 5)
    out["a101"] = (c - o) / (h - l + 0.001 * atr + 1e-12)
    out["a005_proxy"] = _ts_rank(o - _sma(vwap, 10), 10) * -np.abs(_ts_rank(c - vwap, 10))
    out["a034"] = _ts_rank((1 - _ts_rank(_div(_roll(ret, 2, np.std), _roll(ret, 5, np.std)), 20)) + (1 - _ts_rank(d1, 20)), 20)
    if has_vol:
        lv = np.log(v + 1)
        out["a002"] = -_corr(_ts_rank(np.diff(lv, n=2, prepend=[lv[0], lv[0]]), 20), _ts_rank(_div(c - o, o), 20), 6)
        out["a003"] = -_corr(_ts_rank(o, 10), _ts_rank(v, 10), 10)
        out["a006"] = -_corr(o, v, 10)
        out["a012"] = np.sign(np.diff(v, prepend=v[0])) * (-d1) / (atr + 1e-12)
        out["a013"] = -_ts_rank(_corr(_ts_rank(c, 20), _ts_rank(v, 20), 5), 20)
        out["a044"] = -_corr(h, _ts_rank(v, 20), 5)
    return out


# ------------------------------------------------------------------ Smart Money Concepts
def smc(o, h, l, c, atr, k=3):
    """Swings confirmes k bougies apres (fractales), structure (BOS / CHoCH), order blocks, FVG, liquidite."""
    n = len(c)
    trend = np.zeros(n)
    since_bos = np.full(n, NAN)
    choch = np.zeros(n)
    d_sh = np.full(n, NAN)
    d_sl = np.full(n, NAN)
    prem = np.full(n, NAN)
    ob_bull = np.full(n, NAN)
    ob_bear = np.full(n, NAN)
    in_ob = np.zeros(n)
    fvg_bull = np.full(n, NAN)
    fvg_bear = np.full(n, NAN)
    eq_hi = np.zeros(n)
    eq_lo = np.zeros(n)
    sweep = np.zeros(n)
    last_sh = last_sl = NAN
    t = 0
    bos_i = -1
    obu = obd = None            # (bas, haut) de la zone
    fu, fd = [], []             # FVG non combles : haussiers (bas, haut), baissiers
    highs, lows = [], []        # derniers swings pour les « equal highs / lows »
    for i in range(n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else NAN
        j = i - k
        if j - k >= 0:          # le swing en j est confirme quand k bougies ont suivi
            if h[j] == h[j - k:i + 1].max():
                last_sh = h[j]
                highs = (highs + [h[j]])[-5:]
            if l[j] == l[j - k:i + 1].min():
                last_sl = l[j]
                lows = (lows + [l[j]])[-5:]
        if np.isfinite(last_sh) and c[i] > last_sh:
            choch[i] = 1.0 if t < 0 else 0.0
            t, bos_i = 1, i
            dn_idx = [m for m in range(max(0, i - 20), i) if c[m] < o[m]]
            if dn_idx:
                m = dn_idx[-1]
                obu = (l[m], h[m])
            last_sh = NAN
        elif np.isfinite(last_sl) and c[i] < last_sl:
            choch[i] = -1.0 if t > 0 else 0.0
            t, bos_i = -1, i
            up_idx = [m for m in range(max(0, i - 20), i) if c[m] > o[m]]
            if up_idx:
                m = up_idx[-1]
                obd = (l[m], h[m])
            last_sl = NAN
        trend[i] = t
        if bos_i >= 0:
            since_bos[i] = i - bos_i
        if np.isfinite(a):
            if np.isfinite(last_sh):
                d_sh[i] = (last_sh - c[i]) / a
            if np.isfinite(last_sl):
                d_sl[i] = (c[i] - last_sl) / a
            if np.isfinite(last_sh) and np.isfinite(last_sl) and last_sh > last_sl:
                prem[i] = (c[i] - last_sl) / (last_sh - last_sl)
            if obu:
                ob_bull[i] = (c[i] - obu[1]) / a
                if obu[0] <= c[i] <= obu[1]:
                    in_ob[i] = 1.0
                if c[i] < obu[0]:
                    obu = None
            if obd:
                ob_bear[i] = (obd[0] - c[i]) / a
                if obd[0] <= c[i] <= obd[1]:
                    in_ob[i] = -1.0
                if c[i] > obd[1]:
                    obd = None
        if i >= 2:
            if l[i] > h[i - 2]:
                fu.append((h[i - 2], l[i]))
            if h[i] < l[i - 2]:
                fd.append((h[i], l[i - 2]))
        fu = [z for z in fu if l[i] > z[0]][-10:]
        fd = [z for z in fd if h[i] < z[1]][-10:]
        if np.isfinite(a):
            if fu:
                fvg_bull[i] = (c[i] - max(z[1] for z in fu)) / a
            if fd:
                fvg_bear[i] = (min(z[0] for z in fd) - c[i]) / a
            tol = 0.1 * a
            eq_hi[i] = sum(1 for x in highs if abs(x - max(highs)) <= tol) - 1 if highs else 0
            eq_lo[i] = sum(1 for x in lows if abs(x - min(lows)) <= tol) - 1 if lows else 0
            if highs and h[i] > max(highs) and c[i] < max(highs):
                sweep[i] = -1.0
            elif lows and l[i] < min(lows) and c[i] > min(lows):
                sweep[i] = 1.0
    return {"tendance": trend, "bougies_depuis_bos": since_bos, "choch": choch, "dist_swing_haut": d_sh,
            "dist_swing_bas": d_sl, "premium_discount": prem, "dist_ob_haussier": ob_bull, "dist_ob_baissier": ob_bear,
            "dans_ob": in_ob, "dist_fvg_haussier": fvg_bull, "dist_fvg_baissier": fvg_bear, "egaux_hauts": eq_hi,
            "egaux_bas": eq_lo, "balayage_liquidite": sweep}


def compute_extra(o, h, l, c, v, atr):
    has_vol = np.nansum(v) > 0
    return {"chandeliers": candles(o, h, l, c), "alpha158": alpha158(o, h, l, c, v, has_vol),
            "wq101": wq101(o, h, l, c, v, has_vol, atr), "smc": smc(o, h, l, c, atr)}
