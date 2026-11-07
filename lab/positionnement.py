"""Specialiste positionnement : ce que font les gros acteurs et ce que craint le marche des options.
   - COT de la CFTC (chaque semaine depuis 1986) : position nette des fonds a levier, des gerants d'actifs et des
     dealers sur les contrats a terme de devises et de bitcoin, en part de l'interet ouvert ; extremes en percentile
     sur 3 ans (risque de retournement quand tout le monde est du meme cote) ;
   - DVOL de Deribit (volatilite implicite du bitcoin et de l'ether, depuis 2021) : niveau, ecart avec la
     volatilite reellement observee (prime de peur), variation sur 24 h.
   Tout est aligne sur la derniere valeur publiee (aucun regard vers le futur)."""
import numpy as np

NAN = np.nan


def _align(src_ts, val, ts):
    idx = np.searchsorted(src_ts, ts, side="right") - 1
    return np.where(idx >= 0, val[np.clip(idx, 0, None)], NAN)


def _pct(x, win=156):
    """Percentile de chaque valeur hebdomadaire parmi les `win` precedentes (0 a 1)."""
    out = np.full(len(x), NAN)
    for i in range(20, len(x)):
        past = x[max(0, i - win):i]
        past = past[np.isfinite(past)]
        if len(past) >= 20 and np.isfinite(x[i]):
            out[i] = np.mean(past < x[i])
    return out


def cot_features(sym, kind, ts, cot):
    if not cot:
        return {}
    out = {}

    def series(cur):
        a = cot.get(cur)
        if a is None or len(a) < 30:
            return None
        return a

    if kind == "fx" and sym and len(sym) == 6:
        b, q = series(sym[:3]), series(sym[3:])
        for nm, a in (("base", b), ("cotee", q)):
            if a is None:
                continue
            out["cot_fonds_%s" % nm] = _align(a[:, 0], a[:, 1], ts)
            out["cot_gerants_%s" % nm] = _align(a[:, 0], a[:, 2], ts)
            out["cot_percentile_fonds_%s" % nm] = _align(a[:, 0], _pct(a[:, 1]), ts)
            if a.shape[1] > 4:                      # petits traders (la foule, qui perd le plus souvent)
                out["cot_petits_%s" % nm] = _align(a[:, 0], a[:, 4], ts)
                out["cot_percentile_petits_%s" % nm] = _align(a[:, 0], _pct(a[:, 4]), ts)
        if "cot_fonds_base" in out and "cot_fonds_cotee" in out:
            out["cot_ecart_fonds"] = out["cot_fonds_base"] - out["cot_fonds_cotee"]
            out["cot_ecart_gerants"] = out["cot_gerants_base"] - out["cot_gerants_cotee"]
    elif kind == "cr":
        a = series("BTC")
        if a is not None:
            out["cot_bitcoin_fonds"] = _align(a[:, 0], a[:, 1], ts)
            out["cot_bitcoin_gerants"] = _align(a[:, 0], a[:, 2], ts)
            out["cot_bitcoin_percentile_fonds"] = _align(a[:, 0], _pct(a[:, 1]), ts)
    return out


def dvol_features(sym, ts, c, dv):
    if not dv:
        return {}
    out = {}
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    rv = np.full(len(c), NAN)
    for i in range(24, len(c)):
        rv[i] = np.std(r[i - 23:i + 1]) * np.sqrt(24 * 365) * 100
    for cur in ("BTC", "ETH"):
        a = dv.get(cur)
        if a is None or len(a) < 100 or (cur == "ETH" and sym != "ETH"):
            continue
        x = _align(a[:, 0], a[:, 1], ts)
        nm = "dvol_%s" % cur.lower()
        out[nm] = x
        out[nm + "_prime_peur"] = x - rv
        prev = _align(a[:, 0], a[:, 1], ts - 24 * 3600)
        out[nm + "_variation_24h"] = x - prev
        m = np.full(len(x), NAN)
        for i in range(720, len(x)):
            seg = x[i - 720:i]
            seg = seg[np.isfinite(seg)]
            if len(seg) > 300 and np.isfinite(x[i]):
                m[i] = (x[i] - seg.mean()) / (seg.std() + 1e-9)
        out[nm + "_zscore_30j"] = m
    return out


def compute_positionnement(sym, kind, ts, c, refs):
    refs = refs or {}
    F = cot_features(sym, kind, ts, refs.get("cot"))
    if kind == "cr":
        F.update(dvol_features(sym, ts, c, refs.get("dvol")))
    return F


def sources(kind, years=11):
    """Telecharge une fois par recherche les sources du positionnement (dict vide si indisponible)."""
    import os
    from lab import data
    if os.environ.get("POSITIONNEMENT", "1") == "0":
        return {}
    src = {}
    try:
        src["cot"] = data.cot(years=years)
    except Exception:
        src["cot"] = {}
    if kind == "cr":
        dv = {}
        for cur in ("BTC", "ETH"):
            try:
                dv[cur] = data.dvol(cur)
            except Exception:
                pass
        src["dvol"] = dv
    return src
