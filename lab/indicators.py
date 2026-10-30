"""Les 20 specialistes et leurs indicateurs (plus de 200 variantes, chacune declinee en 5 zones = plus de 1 000
conditions). Tout est calcule uniquement avec le passe (aucun regard vers le futur)."""
import datetime
from zoneinfo import ZoneInfo

import numpy as np
from lab import extra as EX
from numpy.lib.stride_tricks import sliding_window_view as swv

NY = ZoneInfo("America/New_York")
AGENTS = {
    "tendance": "Tendance (moyennes mobiles)", "ichimoku": "Ichimoku", "momentum": "Momentum (RSI, MACD, ROC)",
    "oscillateurs": "Oscillateurs (stochastique, CCI, Williams)", "volatilite": "Volatilité (ATR, compression)",
    "bandes": "Bandes (Bollinger, Donchian, Keltner)", "volume": "Volume (OBV, MFI, CMF)", "vwap": "VWAP",
    "profil": "Volume profile (POC, zone de valeur)", "fibonacci": "Fibonacci", "amd": "AMD et sessions",
    "structure": "Structure de marché", "bougies": "Chandeliers", "statisticien": "Statisticien",
    "mathematicien": "Mathématicien (Hurst, entropie)", "retour": "Retour à la moyenne", "temps": "Horaires",
    "intermarche": "Inter-marchés", "macro": "Macro et régime", "carnet": "Carnet d'ordres",
    "inventeur": "Inventeur (idées auto-générées)",
    "tradingview": "Favoris TradingView (WaveTrend, Squeeze, UT Bot, QQE…)",
    "multi_unites": "Multi-unités (tendance 1 h, 4 h, jour)",
    "chandeliers": "Chandeliers TA-Lib (61 figures)",
    "alpha158": "Alpha158 (Microsoft Qlib)",
    "wq101": "101 Alphas (WorldQuant)",
    "smc": "Smart Money Concepts (BOS, CHoCH, order blocks, FVG)",
    "derives": "Dérivés crypto (taux de financement des perpétuels)",
    "chiffres_ronds": "Chiffres ronds (niveaux psychologiques)",
    "weekend": "Volatilité du week-end crypto",
}
HTF = ((3600, "1h"), (14400, "4h"), (86400, "jour"))
NAN = np.nan
# grilles de parametres : DENSE=1 (defaut) -> plus de 600 variantes d'indicateurs
G = {
    "ma": (3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 60, 75, 89, 100, 120, 150, 200, 250, 300),
    "cross": ((3, 10), (5, 13), (5, 20), (8, 21), (9, 30), (10, 50), (12, 26), (13, 48), (20, 50), (20, 100),
              (21, 55), (30, 90), (50, 100), (50, 200), (100, 200)),
    "ichi": ((9, 26, 52), (7, 22, 44), (10, 30, 60), (12, 36, 72), (20, 60, 120), (6, 18, 36)),
    "rsi": (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 21, 25, 28, 30, 35, 42, 50),
    "roc": (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60),
    "macd": ((12, 26, 9), (5, 35, 5), (8, 17, 9), (3, 10, 16), (6, 13, 5), (19, 39, 9), (24, 52, 18), (10, 20, 7)),
    "stoch": (3, 5, 7, 9, 12, 14, 18, 21, 28, 35),
    "will": (5, 7, 10, 14, 20, 28, 40),
    "cci": (7, 10, 14, 20, 30, 40, 50),
    "atr": (3, 5, 7, 10, 14, 20, 28, 40),
    "ratio": ((3, 30), (5, 50), (7, 70), (10, 100), (14, 100), (20, 200)),
    "bb": (10, 14, 20, 30, 40, 50, 75, 100),
    "bbk": (1.0, 1.5, 2.0, 2.5, 3.0),
    "don": (5, 10, 14, 20, 30, 40, 55, 75, 100, 150, 200),
    "kelt": ((10, 1.0), (10, 2.0), (20, 1.5), (20, 2.0), (20, 2.5), (50, 2.0)),
    "vol": (5, 10, 14, 20, 30, 50, 100),
    "z": (5, 10, 14, 20, 30, 40, 50, 75, 100, 150, 200),
    "rank": (10, 20, 30, 50, 75, 100, 150, 200),
    "skew": (20, 30, 50, 100),
    "eff": (5, 10, 14, 20, 30, 50, 75, 100),
    "vr": (2, 3, 4, 6, 8),
    "dev": (3, 5, 7, 10, 14, 20),
    "div": (10, 14, 20, 30, 50),
}


# ------------------------------------------------------------------ outils vectorises
def shift(x, k):
    out = np.full_like(x, NAN, dtype=float)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def sma(x, p):
    out = np.full(len(x), NAN)
    if len(x) >= p:
        c = np.cumsum(np.nan_to_num(x, nan=0.0))
        out[p - 1:] = (c[p - 1:] - np.concatenate(([0.0], c[:-p]))) / p
    return out


def ema(x, p):
    out = np.empty(len(x))
    a = 2.0 / (p + 1)
    e = x[0]
    for i, v in enumerate(x):
        e = v * a + e * (1 - a) if np.isfinite(v) else e
        out[i] = e
    out[:min(p, len(x))] = NAN
    return out


def roll(x, p, fn):
    out = np.full(len(x), NAN)
    if len(x) >= p:
        out[p - 1:] = fn(swv(x, p), axis=1)
    return out


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r[~np.isfinite(r)] = NAN
    return r


def rsi(c, p):
    d = np.diff(c, prepend=c[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = ema(up, 2 * p - 1), ema(dn, 2 * p - 1)
    return 100 - 100 / (1 + safe_div(au, ad))


def true_range(h, l, c):
    pc = shift(c, 1)
    return np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)


def day_index(ts, kind):
    """Numero de seance : jour UTC (forex, crypto) ou jour New York (actions)."""
    if kind == "stocks":
        days = np.array([datetime.datetime.fromtimestamp(t, NY).toordinal() for t in ts])
    else:
        days = (ts // 86400).astype(int)
    return days


def per_day(days, x, fn):
    """Valeur de la seance precedente (fn appliquee a chaque jour), alignee sur chaque bougie."""
    out = np.full(len(x), NAN)
    uniq, start = np.unique(days, return_index=True)
    vals = [fn(x[start[i]:(start[i + 1] if i + 1 < len(start) else len(x))]) for i in range(len(uniq))]
    for i in range(1, len(uniq)):
        s, e = start[i], (start[i + 1] if i + 1 < len(start) else len(x))
        out[s:e] = vals[i - 1]
    return out


def intraday_cum(days, x, fn):
    """Cumul depuis le debut de la seance (inclus la bougie courante, deja cloturee)."""
    out = np.full(len(x), NAN)
    uniq, start = np.unique(days, return_index=True)
    for i in range(len(uniq)):
        s, e = start[i], (start[i + 1] if i + 1 < len(start) else len(x))
        out[s:e] = fn(x[s:e])
    return out


def volume_profile(c, v, days):
    """POC, VAH, VAL de la seance precedente (70 % du volume ; nombre de bougies si pas de volume)."""
    poc, vah, val = (np.full(len(c), NAN) for _ in range(3))
    uniq, start = np.unique(days, return_index=True)
    prev = None
    for i in range(len(uniq)):
        s, e = start[i], (start[i + 1] if i + 1 < len(start) else len(c))
        if prev:
            poc[s:e], vah[s:e], val[s:e] = prev
        px, w = c[s:e], v[s:e]
        if len(px) >= 4:
            w = w if np.nansum(w) > 0 else np.ones(len(px))
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
            prev = (mid[k], edges[hi + 1], edges[lo])
    return poc, vah, val


# ------------------------------------------------------------------ les 20 specialistes
def compute(arr, kind, refs=None, macro=None):
    """arr : (n, 6) temps, o, h, l, c, v. refs : {"bench": cloture alignee de l'actif de reference}.
       macro : {"spy_up": +1/-1 aligne, "vix": niveau aligne}. Retourne {agent: {nom: tableau}}."""
    ts, o, h, l, c, v = (arr[:, i] for i in range(6))
    n = len(c)
    F = {a: {} for a in AGENTS}
    atr14 = ema(true_range(h, l, c), 27)
    ret = np.diff(np.log(c), prepend=np.log(c[0]))
    days = day_index(ts, kind)
    has_vol = np.nansum(v) > 0

    # 1 tendance
    for p in G["ma"]:
        F["tendance"]["sma%d" % p] = safe_div(c, sma(c, p)) - 1
        e = ema(c, p)
        F["tendance"]["ema%d" % p] = safe_div(c, e) - 1
        F["tendance"]["pente_ema%d" % p] = safe_div(e, shift(e, 5)) - 1
    for a, b in G["cross"]:
        F["tendance"]["croise_%d_%d" % (a, b)] = safe_div(ema(c, a), ema(c, b)) - 1

    # 2 ichimoku
    for t_, k_, s_ in G["ichi"]:
        mid = lambda p: (roll(h, p, np.max) + roll(l, p, np.min)) / 2  # noqa: E731
        ten, kij = mid(t_), mid(k_)
        sa, sb = shift((ten + kij) / 2, k_), shift(mid(s_), k_)
        top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
        tag = "%d_%d" % (t_, k_)
        F["ichimoku"]["vs_tenkan_" + tag] = safe_div(c - ten, atr14)
        F["ichimoku"]["vs_kijun_" + tag] = safe_div(c - kij, atr14)
        F["ichimoku"]["tenkan_kijun_" + tag] = safe_div(ten - kij, atr14)
        F["ichimoku"]["au_dessus_nuage_" + tag] = safe_div(c - top, atr14)
        F["ichimoku"]["sous_nuage_" + tag] = safe_div(c - bot, atr14)
        F["ichimoku"]["epaisseur_nuage_" + tag] = safe_div(top - bot, atr14)
        F["ichimoku"]["chikou_" + tag] = safe_div(c, shift(c, k_)) - 1

    # 3 momentum
    for p in G["rsi"]:
        F["momentum"]["rsi%d" % p] = rsi(c, p)
    for p in G["roc"]:
        F["momentum"]["roc%d" % p] = safe_div(c, shift(c, p)) - 1
    for f, s, g in G["macd"]:
        m = ema(c, f) - ema(c, s)
        sg = ema(np.nan_to_num(m), g)
        F["momentum"]["macd_%d_%d" % (f, s)] = safe_div(m, c)
        F["momentum"]["macd_signal_%d_%d" % (f, s)] = safe_div(m - sg, c)
        F["momentum"]["macd_hist_pente_%d_%d" % (f, s)] = safe_div((m - sg) - shift(m - sg, 3), c)

    # divergences prix / indicateur (idee du trader) : nouveau plus haut du prix sans nouveau plus haut de l'indicateur
    def divergence(ind, p):
        hi_p, lo_p = roll(c, p, np.max), roll(c, p, np.min)
        hi_i, lo_i = roll(np.nan_to_num(ind, nan=0.0), p, np.max), roll(np.nan_to_num(ind, nan=0.0), p, np.min)
        span = np.where(hi_i - lo_i > 0, hi_i - lo_i, NAN)
        bear = (c >= hi_p) & (ind < hi_i - 0.15 * span)
        bull = (c <= lo_p) & (ind > lo_i + 0.15 * span)
        return np.where(bear, -1.0, np.where(bull, 1.0, 0.0))
    for p in G["div"]:
        for rp in (5, 9, 14, 21):
            F["momentum"]["divergence_rsi%d_%d" % (rp, p)] = divergence(F["momentum"]["rsi%d" % rp], p)
        F["momentum"]["divergence_macd_%d" % p] = divergence(F["momentum"]["macd_12_26"], p)
        F["momentum"]["divergence_roc10_%d" % p] = divergence(F["momentum"]["roc10"], p)

    # 4 oscillateurs
    for p in G["stoch"]:
        hh, ll = roll(h, p, np.max), roll(l, p, np.min)
        k = 100 * safe_div(c - ll, hh - ll)
        F["oscillateurs"]["stoch_k%d" % p] = k
        F["oscillateurs"]["stoch_d%d" % p] = sma(np.nan_to_num(k, nan=50), 3)
    for p in G["will"]:
        F["oscillateurs"]["williams%d" % p] = -100 * safe_div(roll(h, p, np.max) - c, roll(h, p, np.max) - roll(l, p, np.min))
    tp = (h + l + c) / 3
    for p in G["cci"]:
        md = roll(tp, p, lambda w, axis: np.mean(np.abs(w - w.mean(axis=1, keepdims=True)), axis=axis))
        F["oscillateurs"]["cci%d" % p] = safe_div(tp - sma(tp, p), 0.015 * md)

    # 5 volatilite
    tr = true_range(h, l, c)
    for p in G["atr"]:
        F["volatilite"]["atr_pct%d" % p] = safe_div(ema(tr, 2 * p - 1), c)
    for a, b in G["ratio"]:
        F["volatilite"]["atr_ratio_%d_%d" % (a, b)] = safe_div(sma(tr, a), sma(tr, b))
    for p in (10, 20, 50):
        bw = safe_div(4 * roll(c, p, np.std), sma(c, p))
        F["volatilite"]["bb_largeur%d" % p] = bw
        F["volatilite"]["bb_largeur_rang%d" % p] = roll(bw, 100, lambda w, axis: (w[:, -1:] >= w).mean(axis=axis))
    sd20 = roll(c, 20, np.std)
    F["volatilite"]["squeeze"] = ((4 * sd20) < (3 * atr14)).astype(float)    # Bollinger dans Keltner
    F["volatilite"]["sortie_squeeze"] = F["volatilite"]["squeeze"] - shift(F["volatilite"]["squeeze"], 1)
    F["volatilite"]["parkinson_ratio"] = safe_div(sma(np.log(h / l) ** 2, 10), sma(np.log(h / l) ** 2, 100))

    # 6 bandes
    for p in G["bb"]:
        m, sd = sma(c, p), roll(c, p, np.std)
        for kk in G["bbk"]:
            F["bandes"]["bb_pctb_%d_%.1f" % (p, kk)] = safe_div(c - (m - kk * sd), 2 * kk * sd)
    for p in G["don"]:
        hh, ll = roll(h, p, np.max), roll(l, p, np.min)
        F["bandes"]["donchian%d" % p] = safe_div(c - ll, hh - ll)
    for p, kk in G["kelt"]:
        F["bandes"]["keltner_%d_%.1f" % (p, kk)] = safe_div(c - ema(c, p), kk * atr14)

    # 7 volume
    if has_vol:
        for p in G["vol"]:
            F["volume"]["vol_relatif%d" % p] = safe_div(v, sma(v, p))
        obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)
        F["volume"]["obv_pente"] = safe_div(obv - sma(obv, 20), sma(v, 20) * 20)
        mfv = safe_div((c - l) - (h - c), h - l) * v
        for p in G["vol"]:
            F["volume"]["cmf%d" % p] = safe_div(sma(np.nan_to_num(mfv), p), sma(v, p))
        rmf = tp * v
        pos = np.where(np.diff(tp, prepend=tp[0]) > 0, rmf, 0.0)
        neg = np.where(np.diff(tp, prepend=tp[0]) < 0, rmf, 0.0)
        for p in (5, 7, 10, 14, 21, 28):
            F["volume"]["mfi%d" % p] = 100 - 100 / (1 + safe_div(sma(pos, p), sma(neg, p)))
        F["volume"]["vol_zscore"] = safe_div(v - sma(v, 50), roll(v, 50, np.std))
        for p in G["div"]:
            F["volume"]["divergence_obv_%d" % p] = divergence(obv, p)
            F["volume"]["divergence_mfi_%d" % p] = divergence(F["volume"]["mfi14"], p)

    # 8 VWAP (seance)
    w = v if has_vol else np.ones(n)
    cpv = intraday_cum(days, tp * w, np.cumsum)
    cv = intraday_cum(days, w, np.cumsum)
    vw = safe_div(cpv, cv)
    F["vwap"]["dist_atr"] = safe_div(c - vw, atr14)
    F["vwap"]["dist_pct"] = safe_div(c, vw) - 1
    dev = intraday_cum(days, (c - vw) ** 2 * w, np.cumsum)
    F["vwap"]["zscore"] = safe_div(c - vw, np.sqrt(safe_div(dev, cv)))
    for p in (5, 10, 20, 30, 50, 100):
        F["vwap"]["vwap_glissant%d" % p] = safe_div(c, safe_div(sma(tp * w, p), sma(w, p))) - 1

    # 9 volume profile (seance precedente)
    poc, vah, val = volume_profile(c, w, days)
    F["profil"]["dist_poc"] = safe_div(c - poc, atr14)
    F["profil"]["dist_vah"] = safe_div(c - vah, atr14)
    F["profil"]["dist_val"] = safe_div(c - val, atr14)
    F["profil"]["position_zone_valeur"] = safe_div(c - val, vah - val)

    # 10 fibonacci (range de la veille et dernier swing de 50 bougies)
    ph = per_day(days, h, np.max)
    pl = per_day(days, l, np.min)
    pos_d = safe_div(c - pl, ph - pl)
    F["fibonacci"]["position_veille"] = pos_d
    for lv in (0.236, 0.382, 0.5, 0.618, 0.786):
        F["fibonacci"]["dist_%d_veille" % int(lv * 1000)] = safe_div(c - (pl + lv * (ph - pl)), atr14)
    sh, sl_ = roll(h, 50, np.max), roll(l, 50, np.min)
    F["fibonacci"]["position_swing50"] = safe_div(c - sl_, sh - sl_)
    F["fibonacci"]["dist_618_swing50"] = safe_div(c - (sl_ + 0.618 * (sh - sl_)), atr14)

    # 11 AMD et sessions (range asiatique 0 h - 7 h UTC : accumulation, balayage, distribution)
    hour = (ts % 86400) // 3600
    asia = hour < 7
    ah = intraday_cum(days, np.where(asia, h, -np.inf), np.maximum.accumulate)
    al = intraday_cum(days, np.where(asia, l, np.inf), np.minimum.accumulate)
    ah[~np.isfinite(ah)] = NAN
    al[~np.isfinite(al)] = NAN
    F["amd"]["vs_haut_asie"] = safe_div(c - ah, atr14)
    F["amd"]["vs_bas_asie"] = safe_div(c - al, atr14)
    hi_day = intraday_cum(days, h, np.maximum.accumulate)
    lo_day = intraday_cum(days, l, np.minimum.accumulate)
    F["amd"]["balayage_haut"] = np.where(~asia & (hi_day > ah) & (c < ah), 1.0, 0.0)
    F["amd"]["balayage_bas"] = np.where(~asia & (lo_day < al) & (c > al), 1.0, 0.0)
    F["amd"]["largeur_asie"] = safe_div(ah - al, atr14)
    F["amd"]["position_journee"] = safe_div(c - lo_day, hi_day - lo_day)

    # 12 structure
    F["structure"]["vs_haut_veille"] = safe_div(c - ph, atr14)
    F["structure"]["vs_bas_veille"] = safe_div(c - pl, atr14)
    pc_day = per_day(days, c, lambda x: x[-1])
    op_day = intraday_cum(days, o, lambda x: np.full(len(x), x[0]))
    F["structure"]["gap"] = safe_div(op_day, pc_day) - 1
    pivot = (ph + pl + pc_day) / 3
    F["structure"]["vs_pivot"] = safe_div(c - pivot, atr14)
    F["structure"]["vs_r1"] = safe_div(c - (2 * pivot - pl), atr14)
    F["structure"]["vs_s1"] = safe_div(c - (2 * pivot - ph), atr14)
    F["structure"]["plus_hauts_10"] = roll((h > shift(h, 1)).astype(float), 10, np.sum)
    F["structure"]["plus_bas_10"] = roll((l < shift(l, 1)).astype(float), 10, np.sum)

    # 13 chandeliers
    rng = np.maximum(h - l, 1e-12)
    F["bougies"]["corps"] = (c - o) / rng
    F["bougies"]["meche_haute"] = (h - np.maximum(o, c)) / rng
    F["bougies"]["meche_basse"] = (np.minimum(o, c) - l) / rng
    po, pc_ = shift(o, 1), shift(c, 1)
    F["bougies"]["englobante"] = np.where((c > o) & (pc_ < po) & (c >= po) & (o <= pc_), 1.0,
                                          np.where((c < o) & (pc_ > po) & (c <= po) & (o >= pc_), -1.0, 0.0))
    F["bougies"]["interieure"] = ((h < shift(h, 1)) & (l > shift(l, 1))).astype(float)
    up = (c > shift(c, 1)).astype(float)
    F["bougies"]["hausses_5"] = roll(up, 5, np.sum)
    F["bougies"]["taille_vs_atr"] = safe_div(h - l, atr14)

    # 14 statisticien
    for p in G["z"]:
        F["statisticien"]["zscore%d" % p] = safe_div(c - sma(c, p), roll(c, p, np.std))
    for p in G["skew"]:
        m = roll(ret, p, np.mean)
        s = roll(ret, p, np.std)
        F["statisticien"]["asymetrie%d" % p] = roll(ret, p, lambda w, axis: np.mean((w - w.mean(axis=1, keepdims=True)) ** 3, axis=axis)) / np.where(s > 0, s ** 3, NAN)
        F["statisticien"]["sharpe_local%d" % p] = safe_div(m, s)
    for p in G["rank"]:
        F["statisticien"]["rang%d" % p] = roll(c, p, lambda w, axis: (w[:, -1:] >= w).mean(axis=axis))

    # 15 mathematicien
    for q in G["vr"]:
        rq = np.log(c) - np.log(shift(c, q))
        F["mathematicien"]["ratio_variance%d" % q] = safe_div(roll(rq, 64, np.var), q * roll(ret, 64, np.var))
    F["mathematicien"]["autocorr1"] = roll(ret, 50, lambda w, axis: np.array([np.corrcoef(x[1:], x[:-1])[0, 1] if x.std() > 0 else 0 for x in w]))
    for p in G["eff"]:
        F["mathematicien"]["efficacite%d" % p] = safe_div(np.abs(c - shift(c, p)), roll(np.abs(np.diff(c, prepend=c[0])), p, np.sum))
    sgn = (ret > 0).astype(float)
    pr = roll(sgn, 20, np.mean)
    F["mathematicien"]["entropie20"] = -(pr * np.log(np.clip(pr, 1e-9, 1)) + (1 - pr) * np.log(np.clip(1 - pr, 1e-9, 1)))
    rs = roll(ret, 64, lambda w, axis: np.array([(np.ptp(np.cumsum(x - x.mean())) / x.std()) if x.std() > 0 else NAN for x in w]))
    F["mathematicien"]["hurst"] = np.log(rs) / np.log(64)

    # 16 retour a la moyenne
    for p in G["dev"]:
        F["retour"]["ecart_std%d" % p] = safe_div(c - sma(c, p), roll(c, p, np.std))
    dn = (c < shift(c, 1)).astype(float)
    F["retour"]["baisses_consecutives"] = intraday_streak(dn)
    F["retour"]["hausses_consecutives"] = intraday_streak(up)
    F["retour"]["sous_bb_basse"] = (c < sma(c, 20) - 2 * roll(c, 20, np.std)).astype(float) - (c > sma(c, 20) + 2 * roll(c, 20, np.std)).astype(float)

    # 17 horaires
    hh_ = hour if kind != "stocks" else np.array([datetime.datetime.fromtimestamp(t, NY).hour for t in ts])
    F["temps"]["heure"] = hh_.astype(float)
    F["temps"]["jour_semaine"] = ((ts // 86400 + 3) % 7).astype(float)
    F["temps"]["minute_seance"] = (ts - intraday_cum(days, ts, lambda x: np.full(len(x), x[0]))) / 60.0

    # 18 inter-marches
    if refs and refs.get("bench") is not None and len(refs["bench"]) == n:
        b = refs["bench"]
        for p in (5, 20):
            F["intermarche"]["force_relative%d" % p] = (safe_div(c, shift(c, p)) - 1) - (safe_div(b, shift(b, p)) - 1)
        rb = np.diff(np.log(b), prepend=np.log(b[0]))
        cor = np.full(n, NAN)
        for i in range(49, n):
            x_, y_ = ret[i - 49:i + 1], rb[i - 49:i + 1]
            if x_.std() > 0 and y_.std() > 0:
                cor[i] = np.corrcoef(x_, y_)[0, 1]
        F["intermarche"]["correlation50"] = cor
        for lag in (1, 3, 6):                       # avance entre marches : le meneur (BTC, EURUSD, SPY) bouge d'abord
            F["intermarche"]["meneur_ret%d" % lag] = safe_div(b, shift(b, lag)) - 1
            F["intermarche"]["retard_vs_meneur%d" % lag] = (safe_div(b, shift(b, lag)) - 1) - (safe_div(c, shift(c, lag)) - 1)

    # 19 macro et regime
    if macro:
        for k_, arr_ in macro.items():
            if arr_ is not None and len(arr_) == n:
                F["macro"][k_] = arr_
    dly = per_day(days, c, lambda x: x[-1])
    F["macro"]["tendance_jour_20"] = safe_div(c, sma_days(days, dly, 20)) - 1

    # ---------------------------------------------------------- extensions (plus d'indicateurs « inimaginables »)
    def wma(x, p):
        w = np.arange(1, p + 1, dtype=float)
        return roll(np.nan_to_num(x), p, lambda a, axis: (a * w).sum(axis=axis) / w.sum())

    def linreg(x, p):
        t_ = np.arange(p, dtype=float)
        tm, tv = t_.mean(), ((t_ - t_.mean()) ** 2).sum()
        slope = roll(x, p, lambda a, axis: ((a - a.mean(axis=1, keepdims=True)) * (t_ - tm)).sum(axis=axis) / tv)
        r2 = roll(x, p, lambda a, axis: np.nan_to_num(np.array([np.corrcoef(r_, t_)[0, 1] ** 2 if r_.std() > 0 else 0 for r_ in a])))
        return slope, r2
    for p in (6, 8, 10, 13, 16, 20, 30, 40, 50, 100):
        hull = wma(2 * wma(c, p // 2) - wma(c, p), max(2, int(p ** 0.5)))
        F["tendance"]["hull%d" % p] = safe_div(c, hull) - 1
    for p in (5, 10, 14, 20, 30, 40, 50, 75, 100, 150):
        sl_, r2_ = linreg(c, p)
        F["tendance"]["regression_pente%d" % p] = safe_div(sl_ * p, c)
        F["statisticien"]["regression_r2_%d" % p] = r2_
    for p in (5, 7, 10, 14, 20, 25, 35, 50):
        up_ = roll(h, p + 1, lambda a, axis: np.argmax(a, axis=axis)) / p
        dn_ = roll(l, p + 1, lambda a, axis: np.argmin(a, axis=axis)) / p
        F["tendance"]["aroon%d" % p] = up_ - dn_
    for p, m_ in ((7, 2.0), (10, 3.0), (14, 2.0), (20, 3.0), (10, 1.5), (5, 1.0), (14, 3.0), (30, 2.5), (50, 3.0)):
        mid_ = (h + l) / 2
        a_ = ema(true_range(h, l, c), 2 * p - 1)
        F["tendance"]["supertrend_%d_%.1f" % (p, m_)] = safe_div(c - (mid_ - m_ * a_), 2 * m_ * a_)
    for p in (14, 21, 28):
        vmp, vmm = np.abs(h - shift(l, 1)), np.abs(l - shift(h, 1))
        F["tendance"]["vortex%d" % p] = safe_div(roll(vmp, p, np.nansum) - roll(vmm, p, np.nansum), roll(tr, p, np.nansum))
    for p in (5, 9, 12, 15, 20, 30):
        e3 = ema(ema(ema(c, p), p), p)
        F["momentum"]["trix%d" % p] = safe_div(e3, shift(e3, 1)) - 1
    for p in (5, 9, 14, 20, 25, 30, 40, 50):
        d_ = np.diff(c, prepend=c[0])
        su, sd_ = roll(np.clip(d_, 0, None), p, np.sum), roll(np.clip(-d_, 0, None), p, np.sum)
        F["momentum"]["cmo%d" % p] = 100 * safe_div(su - sd_, su + sd_)
    for p in (13, 26):
        e_ = ema(c, p)
        F["momentum"]["elder_force_haussiere%d" % p] = safe_div(h - e_, atr14)
        F["momentum"]["elder_force_baissiere%d" % p] = safe_div(l - e_, atr14)
    for p in (5, 10, 14, 20, 30, 40):
        F["momentum"]["dpo%d" % p] = safe_div(c - shift(sma(c, p), p // 2 + 1), atr14)
    bp = c - np.fmin(l, shift(c, 1))
    for a, b, d in ((7, 14, 28), (5, 10, 20)):
        avg = lambda p: safe_div(roll(bp, p, np.nansum), roll(tr, p, np.nansum))  # noqa: E731
        F["oscillateurs"]["ultimate_%d" % a] = 100 * (4 * avg(a) + 2 * avg(b) + avg(d)) / 7
    for p in (20, 30, 50, 75, 100):
        F["statisticien"]["aplatissement%d" % p] = roll(ret, p, lambda w, axis: np.mean((w - w.mean(axis=1, keepdims=True)) ** 4, axis=axis)) / np.where(roll(ret, p, np.var) > 0, roll(ret, p, np.var) ** 2, NAN)
    for lag in (2, 3, 5, 10):
        F["mathematicien"]["autocorr%d" % lag] = roll(ret, 60, lambda w, axis, lag=lag: np.array([np.corrcoef(x[lag:], x[:-lag])[0, 1] if x.std() > 0 else 0 for x in w]))
    for p in (7, 14, 20, 28, 50):
        ulcer = np.sqrt(roll(((c - roll(c, p, np.max)) / c) ** 2, p, np.nanmean))
        F["volatilite"]["ulcer%d" % p] = ulcer
    for kk in (1.0, 2.0, 3.0):
        F["vwap"]["bande_%.0f_sigma" % kk] = F["vwap"]["zscore"] - kk
    for p in (20, 100, 200):
        sh2, sl2 = roll(h, p, np.max), roll(l, p, np.min)
        F["fibonacci"]["position_swing%d" % p] = safe_div(c - sl2, sh2 - sl2)
        F["fibonacci"]["dist_618_swing%d" % p] = safe_div(c - (sl2 + 0.618 * (sh2 - sl2)), atr14)
    for ext in (1.272, 1.618):
        F["fibonacci"]["extension_%d" % int(ext * 1000)] = safe_div(c - (pl + ext * (ph - pl)), atr14)
    body, rng_ = np.abs(c - o), np.maximum(h - l, 1e-12)
    F["bougies"]["marteau"] = ((np.minimum(o, c) - l) > 2 * body).astype(float) * (body / rng_ < 0.35)
    F["bougies"]["etoile_filante"] = ((h - np.maximum(o, c)) > 2 * body).astype(float) * (body / rng_ < 0.35)
    F["bougies"]["doji"] = (body / rng_ < 0.1).astype(float)
    F["bougies"]["trois_soldats"] = roll(((c > o) & (c > shift(c, 1))).astype(float), 3, np.sum) - roll(((c < o) & (c < shift(c, 1))).astype(float), 3, np.sum)
    F["bougies"]["harami"] = ((h < shift(h, 1)) & (l > shift(l, 1)) & (np.sign(c - o) != np.sign(shift(c, 1) - shift(o, 1)))).astype(float)
    for nd in (2, 5):
        hh_d = per_day(days, h, np.max)
        F["structure"]["vs_haut_%dj" % nd] = safe_div(c - roll(hh_d, max(1, nd * 4), np.nanmax), atr14)
        F["structure"]["vs_bas_%dj" % nd] = safe_div(c - roll(per_day(days, l, np.min), max(1, nd * 4), np.nanmin), atr14)
    for p in (2, 3, 5, 7, 10):
        F["retour"]["rsi_du_rsi%d" % p] = rsi(np.nan_to_num(F["momentum"]["rsi14"], nan=50.0), p)

    # 22 favoris de la communaute TradingView (formules publiques, reecrites sans regarder le futur)
    tv = F.setdefault("tradingview", {})
    hlc3 = (h + l + c) / 3
    for n1, n2 in ((10, 21), (9, 12)):                            # WaveTrend (LazyBear / VuManChu Cipher B)
        esa = ema(hlc3, n1)
        dd = ema(np.abs(hlc3 - np.nan_to_num(esa)), n1)
        wt1 = ema(np.nan_to_num(safe_div(hlc3 - esa, 0.015 * dd)), n2)
        wt2 = sma(np.nan_to_num(wt1), 4)
        tv["wavetrend_%d_%d" % (n1, n2)] = wt1
        tv["wavetrend_croise_%d_%d" % (n1, n2)] = wt1 - wt2
    for p in (20, 30):                                            # Squeeze Momentum (LazyBear)
        mid_ = ((roll(h, p, np.max) + roll(l, p, np.min)) / 2 + sma(c, p)) / 2
        sl_, _r2 = linreg(np.nan_to_num(c - mid_), p)
        tv["squeeze_momentum%d" % p] = safe_div((c - mid_) + sl_ * (p - 1) / 2, atr14)
    for key, ap in ((1.0, 10), (2.0, 10), (1.0, 3)):              # UT Bot (stop suiveur ATR)
        a_ = key * ema(true_range(h, l, c), 2 * ap - 1)
        stop = np.full(n, NAN)
        for i in range(1, n):
            if not np.isfinite(a_[i]):
                continue
            prev = stop[i - 1] if np.isfinite(stop[i - 1]) else c[i] - a_[i]
            if c[i] > prev and c[i - 1] > prev:
                stop[i] = max(prev, c[i] - a_[i])
            elif c[i] < prev and c[i - 1] < prev:
                stop[i] = min(prev, c[i] + a_[i])
            else:
                stop[i] = c[i] - a_[i] if c[i] > prev else c[i] + a_[i]
        tv["ut_bot_%.0f_%d" % (key, ap)] = safe_div(c - stop, atr14)
    for p, m_ in ((22, 3.0), (14, 2.0)):                          # Chandelier Exit
        a_ = ema(true_range(h, l, c), 2 * p - 1)
        tv["chandelier_long_%d" % p] = safe_div(c - (roll(h, p, np.max) - m_ * a_), atr14)
        tv["chandelier_short_%d" % p] = safe_div((roll(l, p, np.min) + m_ * a_) - c, atr14)
    for rp, sf in ((14, 5), (6, 3)):                              # QQE
        rs_ = ema(np.nan_to_num(rsi(c, rp), nan=50.0), sf)
        atr_rsi = ema(np.abs(np.diff(rs_, prepend=rs_[0])), 2 * rp - 1) * 4.236
        trail = np.full(n, NAN)
        for i in range(1, n):
            if not (np.isfinite(rs_[i]) and np.isfinite(atr_rsi[i])):
                continue
            pv = trail[i - 1] if np.isfinite(trail[i - 1]) else rs_[i]
            if rs_[i] > pv and rs_[i - 1] > pv:
                trail[i] = max(pv, rs_[i] - atr_rsi[i])
            elif rs_[i] < pv and rs_[i - 1] < pv:
                trail[i] = min(pv, rs_[i] + atr_rsi[i])
            else:
                trail[i] = rs_[i] - atr_rsi[i] if rs_[i] > pv else rs_[i] + atr_rsi[i]
        tv["qqe_%d" % rp] = rs_ - trail
        tv["qqe_niveau_%d" % rp] = rs_ - 50
    for p, m_ in ((100, 3.0), (50, 2.0)):                         # Range Filter
        rng_f = ema(np.abs(np.diff(c, prepend=c[0])), p) * m_
        filt = np.full(n, NAN)
        for i in range(n):
            if not np.isfinite(rng_f[i]):
                continue
            pv = filt[i - 1] if i and np.isfinite(filt[i - 1]) else c[i]
            filt[i] = max(pv, c[i] - rng_f[i]) if c[i] > pv else min(pv, c[i] + rng_f[i])
        tv["range_filter_%d" % p] = safe_div(c - filt, atr14)
        tv["range_filter_pente_%d" % p] = np.sign(filt - shift(filt, 1))
    for p in (10, 20):                                            # SSL Channel
        up_, dn_ = sma(h, p), sma(l, p)
        tv["ssl_%d" % p] = safe_div(c - (up_ + dn_) / 2, up_ - dn_)
    for p in (9, 20):                                             # Fisher Transform
        hh, ll = roll(hlc3, p, np.max), roll(hlc3, p, np.min)
        x_ = np.clip(2 * safe_div(hlc3 - ll, hh - ll) - 1, -0.999, 0.999)
        x_ = ema(np.nan_to_num(x_), 3)
        tv["fisher_%d" % p] = 0.5 * np.log(safe_div(1 + x_, 1 - x_))
    for f_, s_, cy in ((23, 50, 10), (12, 26, 9)):                # Schaff Trend Cycle
        m = ema(c, f_) - ema(c, s_)
        lo_, hi_ = roll(np.nan_to_num(m), cy, np.min), roll(np.nan_to_num(m), cy, np.max)
        k1 = ema(np.nan_to_num(100 * safe_div(m - lo_, hi_ - lo_), nan=50.0), 3)
        lo2, hi2 = roll(k1, cy, np.min), roll(k1, cy, np.max)
        tv["schaff_%d_%d" % (f_, s_)] = ema(np.nan_to_num(100 * safe_div(k1 - lo2, hi2 - lo2), nan=50.0), 3)
    for q_ in (0.01, 0.1):                                        # filtre de Kalman (niveau)
        est, pe, out_k = c[0], 1.0, np.empty(n)
        for i in range(n):
            pe += q_
            kg = pe / (pe + 1.0)
            est = est + kg * (c[i] - est)
            pe *= (1 - kg)
            out_k[i] = est
        tv["kalman_%g" % q_] = safe_div(c - out_k, atr14)
    bull_gap = (l > shift(h, 2))                                   # Fair Value Gaps (SMC)
    bear_gap = (h < shift(l, 2))
    tv["fvg_haussiers_20"] = roll(bull_gap.astype(float), 20, np.sum)
    tv["fvg_baissiers_20"] = roll(bear_gap.astype(float), 20, np.sum)
    tv["fvg_dernier"] = np.where(bull_gap, 1.0, np.where(bear_gap, -1.0, 0.0))
    for bw in (8, 16):                                            # Nadaraya-Watson (causal, ne repeint pas)
        wts = np.exp(-(np.arange(50)[::-1] ** 2) / (2 * bw * bw))
        nw = roll(c, 50, lambda w, axis, wts=wts: (w * wts).sum(axis=axis) / wts.sum())
        mae = roll(np.abs(c - np.nan_to_num(nw)), 50, np.mean) * 3
        tv["nadaraya_watson_%d" % bw] = safe_div(c - nw, mae)

    # 23 multi-unites : la tendance des unites superieures (1 h, 4 h, jour), comme les desks institutionnels
    F["multi_unites"] = higher_tf(ts, o, h, l, c)

    # 24-27 sources publiques reconnues : chandeliers TA-Lib, Alpha158 (Qlib), 101 Alphas (WorldQuant), SMC
    F.update(EX.compute_extra(o, h, l, c, v, atr14, ts))
    if refs and refs.get("funding"):                # crypto : financement des perpetuels (aligne, deja connu)
        F["derives"] = {k: np.asarray(x, dtype=float) for k, x in refs["funding"].items() if len(x) == n}

    # 20 carnet d'ordres : uniquement en direct (pas d'historique gratuit) -> vote de confirmation dans council.py
    return F, atr14


def higher_tf(ts, o, h, l, c):
    """Indicateurs calcules sur les bougies 1 h / 4 h / jour, alignes sans regarder l'avenir : une bougie superieure
       n'est utilisee qu'une fois entierement cloturee."""
    out = {}
    if len(ts) < 3:
        return out
    step = float(np.median(np.diff(ts)))
    for H, lab in HTF:
        if H <= step * 1.5:
            continue
        g = (ts // H).astype(np.int64)
        uniq, start = np.unique(g, return_index=True)
        end = np.append(start[1:], len(ts))
        ho = o[start]
        hc = c[end - 1]
        hh = np.maximum.reduceat(h, start)
        hl = np.minimum.reduceat(l, start)
        k = np.searchsorted((uniq + 1) * H, ts + step, side="right") - 1   # derniere bougie superieure cloturee
        ok = k >= 0
        kk = np.clip(k, 0, None)

        def al(x):
            r = x[kk].astype(float)
            r[~ok] = NAN
            return r
        hatr = ema(true_range(hh, hl, hc), 14)
        long_p = (10, 20) if lab == "jour" else (20, 50)
        for p in long_p:
            e = ema(hc, p)
            out["tendance_%s_ema%d" % (lab, p)] = al(safe_div(hc, e) - 1)
            out["prix_vs_ema%d_%s" % (p, lab)] = safe_div(c - al(e), al(hatr))          # cloture actuelle (connue)
        e20 = ema(hc, long_p[0])
        out["pente_%s" % lab] = al(safe_div(e20, shift(e20, 3)) - 1)
        out["rsi14_%s" % lab] = al(rsi(hc, 14))
        n_ = 10 if lab == "jour" else 20
        mx, mn = roll(hh, n_, np.max), roll(hl, n_, np.min)
        out["position_%s_%d" % (lab, n_)] = al(safe_div(hc - mn, mx - mn))
        out["bougie_%s" % lab] = al(safe_div(hc - ho, hatr))
    return out


def intraday_streak(x):
    out = np.zeros(len(x))
    run = 0.0
    for i, val in enumerate(x):
        run = run + 1 if val > 0 else 0.0
        out[i] = run
    return out


def sma_days(days, dly, p):
    """Moyenne des p dernieres clotures journalieres connues (alignee)."""
    out = np.full(len(dly), NAN)
    uniq, start = np.unique(days, return_index=True)
    closes = [dly[s] for s in start]
    for i in range(len(uniq)):
        if i >= p:
            s, e = start[i], (start[i + 1] if i + 1 < len(start) else len(dly))
            out[s:e] = np.nanmean(closes[i - p + 1:i + 1])
    return out


def count(F):
    return sum(len(v) for v in F.values())
