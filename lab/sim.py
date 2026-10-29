"""Simulation des sorties : pour CHAQUE bougie et chaque modele de sortie, que se serait-il passe si on etait entre
a l'ouverture suivante ? Objectif / stop en multiples de l'ATR, duree maxi, execution prudente (si l'objectif et le stop
sont touches dans la meme bougie : perte ; gap au-dela du stop : execution a l'ouverture)."""
import numpy as np


def outcomes(arr, atr, exit_, days=None, cost_bps=0.0):
    """Retourne {1: (r, gagne, valide), -1: (...)} ; r en multiples du risque (stop = -1), frais deduits."""
    o, h, l, c = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    n = len(c)
    tp_k, sl_k, H = exit_["tp"], exit_["sl"], int(exit_["h"])
    out = {}
    entry = np.full(n, np.nan)
    entry[:-1] = o[1:]
    for dr in (1, -1):
        risk = sl_k * atr
        tp = entry + dr * tp_k * atr
        sl = entry - dr * risk
        r = np.full(n, np.nan)
        done = np.zeros(n, dtype=bool)
        last_close = np.full(n, np.nan)
        for k in range(1, H + 1):
            j = np.arange(n) + k
            ok = j < n
            jj = np.where(ok, j, n - 1)
            if days is not None:
                ok &= days[jj] == days[np.minimum(np.arange(n) + 1, n - 1)]    # actions : pas de nuit
            hk, lk, ok_ = h[jj], l[jj], ok & ~done
            ov = o[jj]
            hit_sl = ok_ & ((dr > 0) & (lk <= sl) | (dr < 0) & (hk >= sl))
            hit_tp = ok_ & ((dr > 0) & (hk >= tp) | (dr < 0) & (lk <= tp))
            fill_sl = np.where(dr > 0, np.minimum(ov, sl), np.maximum(ov, sl)) if k > 1 else sl
            r = np.where(hit_sl, (fill_sl - entry) * dr / risk, r)
            r = np.where(hit_tp & ~hit_sl, tp_k / sl_k, r)
            done |= hit_sl | hit_tp
            last_close = np.where(ok, c[jj], last_close)
        timeout = ~done & np.isfinite(last_close)
        r = np.where(timeout, (last_close - entry) * dr / risk, r)
        r = np.maximum(r, -1.5)
        cost = entry * cost_bps / 1e4 / risk
        r = r - cost
        valid = np.isfinite(r) & np.isfinite(atr) & (atr > 0)
        out[dr] = (np.where(valid, r, 0.0), valid & (r > 0), valid)
    return out
