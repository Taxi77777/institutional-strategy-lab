"""Specialiste Order flow (demande du trader), crypto ET forex :
   - crypto : VRAI delta = volume des acheteurs agressifs - volume des vendeurs agressifs (Binance, bougies 1 h) ;
   - forex : pas de bourse centrale, donc pas de vrai delta gratuit ; delta ESTIME a partir du volume des ticks
     Dukascopy et de la position de la cloture dans la bougie (methode classique « close location value »).
   Indicateurs : delta, delta cumule 24 h / 72 h, divergences delta / prix (globale et aux creux / sommets), absorption,
   desequilibres empiles, epuisement, volume relatif ; et la CHECKLIST D'ENTREE ORDER FLOW des pros :
   1) zone d'une unite superieure (plus bas / haut de la veille ou de la semaine, order block, FVG),
   2) le prix arrive dans la zone, 3) divergence du delta, 4) signal de declenchement (cassure du haut / bas de la
   bougie precedente avec un delta dans le bon sens). Tout est causal."""
import numpy as np

NAN = np.nan


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = NAN
    return r


def _rsum(x, p):
    out = np.full(len(x), NAN)
    if len(x) >= p:
        cs = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
        cnt = np.cumsum(np.r_[0, np.isfinite(x).astype(int)])
        s = cs[p:] - cs[:-p]
        k = cnt[p:] - cnt[:-p]
        out[p - 1:] = np.where(k >= p * 0.8, s, NAN)
    return out


def _prev(x, k=1):
    return np.r_[np.full(k, NAN), x[:-k]]


def compute_orderflow(kind, ts, o, h, l, c, atr, F, flow):
    """flow : {"vol": volume aligne, "buy": volume acheteur aligne (crypto) ou None (forex : delta estime)}."""
    n = len(c)
    if not flow or flow.get("vol") is None or len(flow["vol"]) != n:
        return {}
    vol = np.asarray(flow["vol"], dtype=float).copy()
    vol[~(vol > 0)] = NAN
    buy = flow.get("buy")
    if buy is not None and len(buy) == n:
        delta = 2 * np.asarray(buy, dtype=float) - vol              # vrai delta (acheteurs - vendeurs agressifs)
    else:
        rng = h - l
        clv = np.where(rng > 0, (2 * c - h - l) / np.where(rng > 0, rng, 1), 0.0)
        delta = clv * vol                                           # delta estime (forex)
    ratio = _div(delta, vol)
    out = {"delta": ratio}
    for p in (24, 72):
        out["delta_cumule_%dh" % p] = _div(_rsum(delta, p), _rsum(vol, p))
    ret24 = np.full(n, NAN)
    ret24[24:] = np.log(c[24:] / c[:-24])
    cvd = out["delta_cumule_24h"]
    out["divergence_delta_24h"] = np.where(np.isfinite(ret24) & np.isfinite(cvd), (np.sign(cvd) - np.sign(ret24)) / 2, NAN)
    # divergence aux creux / sommets : nouveau plus bas 20 bougies mais delta cumule plus haut qu'au plus bas precedent
    div_lo, div_hi = np.zeros(n), np.zeros(n)
    last_lo = last_hi = None
    for i in range(20, n):
        if not np.isfinite(cvd[i]):
            continue
        if l[i] < l[i - 20:i].min():
            if last_lo is not None and l[i] < last_lo[0] and cvd[i] > last_lo[1]:
                div_lo[i] = 1.0
            last_lo = (l[i], cvd[i])
        if h[i] > h[i - 20:i].max():
            if last_hi is not None and h[i] > last_hi[0] and cvd[i] < last_hi[1]:
                div_hi[i] = -1.0
            last_hi = (h[i], cvd[i])
    div = div_lo + div_hi
    out["divergence_creux_sommet"] = div
    recent = np.zeros(n)                                            # divergence dans les 5 dernieres bougies
    for i in range(n):
        w = div[max(0, i - 4):i + 1]
        recent[i] = 1.0 if (w > 0).any() else (-1.0 if (w < 0).any() else 0.0)
    out["divergence_recente"] = recent
    # volume relatif, absorption, desequilibres empiles, epuisement
    vz = np.full(n, NAN)
    for i in range(100, n):
        w = vol[i - 100:i]
        w = w[np.isfinite(w)]
        if len(w) > 50 and np.isfinite(vol[i]):
            vz[i] = (vol[i] - w.mean()) / (w.std() + 1e-12)
    out["volume_relatif"] = vz
    rng = h - l
    body = np.abs(c - o)
    small = np.isfinite(atr) & (rng < 0.8 * atr) & (body < 0.4 * np.where(rng > 0, rng, 1))
    out["absorption"] = np.where((vz > 1.5) & small, -np.sign(np.nan_to_num(ratio)), 0.0)   # agresseurs absorbes
    stack = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(ratio[i]) and abs(ratio[i]) > 0.15:
            s = np.sign(ratio[i])
            stack[i] = stack[i - 1] + s if np.sign(stack[i - 1]) == s else s
    out["desequilibre_empile"] = stack
    hh20 = np.r_[np.full(20, NAN), [h[i - 20:i].max() for i in range(20, n)]] if n > 20 else np.full(n, NAN)
    ll20 = np.r_[np.full(20, NAN), [l[i - 20:i].min() for i in range(20, n)]] if n > 20 else np.full(n, NAN)
    mid = (h + l) / 2
    out["epuisement"] = np.where((h > hh20) & (ratio > 0.3) & (c < mid), -1.0,
                                 np.where((l < ll20) & (ratio < -0.3) & (c > mid), 1.0, 0.0))
    # checklist d'entree order flow (4 etapes)
    def g(ag, nm):
        x = (F.get(ag) or {}).get(nm)
        return np.asarray(x, dtype=float) if x is not None and len(x) == n else np.full(n, NAN)
    near = lambda x: np.isfinite(x) & (np.abs(x) <= 0.5)
    zone_up = (near(g("liquidite", "dist_bas_veille")) | near(g("liquidite", "dist_bas_semaine"))
               | (g("smc", "dans_ob_haussier") == 1) | (g("smc", "dans_fvg_haussier") == 1)
               | (np.nan_to_num(g("smc", "ob_haussier_dist"), nan=9) <= 0.3))
    zone_dn = (near(g("liquidite", "dist_haut_veille")) | near(g("liquidite", "dist_haut_semaine"))
               | (g("smc", "dans_ob_baissier") == 1) | (g("smc", "dans_fvg_baissier") == 1)
               | (np.nan_to_num(g("smc", "ob_baissier_dist"), nan=9) <= 0.3))
    trig_up = (c > _prev(h)) & (ratio > 0)
    trig_dn = (c < _prev(l)) & (ratio < 0)
    div_up = (recent > 0) | (out["divergence_delta_24h"] > 0)
    div_dn = (recent < 0) | (out["divergence_delta_24h"] < 0)
    zone_recent_up = np.array([zone_up[max(0, i - 5):i + 1].any() for i in range(n)])
    zone_recent_dn = np.array([zone_dn[max(0, i - 5):i + 1].any() for i in range(n)])
    steps_up = zone_recent_up.astype(int) + (zone_recent_up & div_up).astype(int) + (zone_recent_up & div_up & trig_up).astype(int)
    steps_dn = zone_recent_dn.astype(int) + (zone_recent_dn & div_dn).astype(int) + (zone_recent_dn & div_dn & trig_dn).astype(int)
    out["checklist_etapes"] = np.where(steps_up >= steps_dn, steps_up, -steps_dn).astype(float)
    out["modele_order_flow"] = np.where(steps_up == 3, 1.0, np.where(steps_dn == 3, -1.0, 0.0))
    valid = np.isfinite(vol)
    for k in out:
        out[k] = np.where(valid, out[k], NAN)                       # volume inconnu : aucun signal d'order flow
    return out
