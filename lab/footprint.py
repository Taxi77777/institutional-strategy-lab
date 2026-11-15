"""Test FOOTPRINT (order flow par niveau de prix), demande du trader (15 nov.). Crypto seulement : le forex n'a pas de
marche central, donc pas de footprint gratuit. Donnees : toutes les transactions Binance (data.binance.vision,
aggTrades, public, sans inscription), avec le sens de l'agresseur. Pour chaque bougie, volume acheteur et vendeur a
chaque niveau de prix (pas de 0,05 %).
Lectures testees (entree a l'ouverture suivante, stop au-dela de l'extreme de la bougie + 0,1 ATR, moitie au premier
objectif puis stop a l'entree, objectif 2 fois le risque, regle stricte) :
  - absorption : la bougie chasse le plus bas des 8 precedentes, beaucoup de ventes agressives dans le quart bas,
    mais cloture dans la moitie haute -> vendeurs absorbes : achat (inverse pour la vente) ;
  - desequilibres : >= 3 niveaux consecutifs ou les acheteurs (vendeurs) dominent 3 fois -> continuation ;
  - divergence : nouveau plus haut (bas) des 8 bougies mais delta contraire -> retournement ;
  - epuisement : nouveau plus haut (bas) avec presque aucun volume aux 2 niveaux extremes et cloture opposee.
Test honnete : 60 % anciens pour choisir, 40 % recents (jamais vus).   python -m lab.footprint cr 15M"""
import datetime
import io
import json
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")
COINS = os.environ.get("FP_COINS", "BTC,ETH,SOL,XRP,DOGE,ADA,LINK,AVAX,LTC,SUI").split(",")
DAYS = int(os.environ.get("FP_DAYS", "120"))
URL = "https://data.binance.vision/data/spot/daily/aggTrades/{p}/{p}-aggTrades-{d}.zip"
BIN = 0.0005
COLS = ["t", "o", "h", "l", "c", "vol", "delta", "abs_bas", "abs_haut", "stack_buy", "stack_sell", "ext_haut", "ext_bas"]


def bar_features(t, px, q, buy, step):
    """Une journee de transactions -> lignes de bougies avec les mesures footprint."""
    out = []
    bar = (t // step).astype(np.int64)
    w = px[0] * BIN
    lvl = np.floor(px / w).astype(np.int64)
    order = np.argsort(bar, kind="stable")
    bar, px, q, buy, lvl = bar[order], px[order], q[order], buy[order], lvl[order]
    cuts = np.flatnonzero(np.diff(bar)) + 1
    for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(bar)]):
        p, qq, bb, lv = px[s:e], q[s:e], buy[s:e], lvl[s:e]
        o, c, h, l = p[0], p[-1], p.max(), p.min()
        vol = qq.sum()
        if vol <= 0:
            continue
        bv = (qq * bb).sum()
        lo, hi = lv.min(), lv.max()
        n = hi - lo + 1
        B = np.bincount(lv - lo, weights=qq * bb, minlength=n)
        S = np.bincount(lv - lo, weights=qq * (1 - bb), minlength=n)
        q1 = max(1, int(round(n / 4)))
        abs_bas = S[:q1].sum() / vol
        abs_haut = B[-q1:].sum() / vol
        # desequilibres diagonaux : acheteurs au niveau k contre vendeurs au niveau k-1 (et inverse)
        ib = (B[1:] >= 3 * np.maximum(S[:-1], 1e-12)) & (B[1:] > 0) if n > 1 else np.array([], bool)
        isell = (S[:-1] >= 3 * np.maximum(B[1:], 1e-12)) & (S[:-1] > 0) if n > 1 else np.array([], bool)

        def run(x):
            m = r = 0
            for v in x:
                r = r + 1 if v else 0
                m = max(m, r)
            return m
        k2 = min(2, n)
        out.append([bar[s] * step, o, h, l, c, vol, (2 * bv - vol) / vol, abs_bas, abs_haut, run(ib), run(isell),
                    (B[-k2:].sum() + S[-k2:].sum()) / vol, (B[:k2].sum() + S[:k2].sum()) / vol])
    return out


def day_bars(sym, d, steps):
    import pandas as pd
    from lab.data import SESSION
    p = "%sUSDT" % sym
    try:
        r = SESSION.get(URL.format(p=p, d=d), timeout=120)
        if r.status_code != 200:
            return sym, d, None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, usecols=[1, 2, 5, 6])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        t = df[5].to_numpy(np.float64)
        t = t / (1e6 if t[0] > 1e14 else 1e3)
        buy = (~df[6].astype(bool)).to_numpy(np.float64)          # isBuyerMaker faux = acheteur agressif
        px, q = df[1].to_numpy(np.float64), df[2].to_numpy(np.float64)
        return sym, d, {st: bar_features(t, px, q, buy, st) for st in steps}
    except Exception as ex:
        log.warning("%s %s : %s", sym, d, ex)
        return sym, d, None


def setups(F, atr, model, seuil):
    t, o, h, l, c = F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4]
    col = {k: F[:, i] for i, k in enumerate(COLS)}
    n, out = len(F), []
    for i in range(9, n - 1):
        A = atr[i]
        if not np.isfinite(A) or A <= 0 or t[i + 1] - t[i] > 2 * (t[1] - t[0]):
            continue
        rngb = h[i] - l[i]
        if rngb <= 0:
            continue
        pos = (c[i] - l[i]) / rngb
        new_lo, new_hi = l[i] < l[i - 8:i].min(), h[i] > h[i - 8:i].max()
        d = 0
        if model == "absorption":
            if new_lo and col["abs_bas"][i] >= seuil and col["delta"][i] < 0 and pos > 0.5:
                d = 1
            elif new_hi and col["abs_haut"][i] >= seuil and col["delta"][i] > 0 and pos < 0.5:
                d = -1
        elif model == "desequilibres":
            if col["stack_buy"][i] >= seuil and c[i] > o[i] and col["delta"][i] > 0.1:
                d = 1
            elif col["stack_sell"][i] >= seuil and c[i] < o[i] and col["delta"][i] < -0.1:
                d = -1
        elif model == "divergence":
            if new_hi and col["delta"][i] < -seuil:
                d = -1
            elif new_lo and col["delta"][i] > seuil:
                d = 1
        elif model == "epuisement":
            if new_hi and col["ext_haut"][i] <= seuil and pos < 0.5:
                d = -1
            elif new_lo and col["ext_bas"][i] <= seuil and pos > 0.5:
                d = 1
        if not d:
            continue
        e = o[i + 1]
        sl = (l[i] - 0.1 * A) if d == 1 else (h[i] + 0.1 * A)
        risk = (e - sl) * d
        if risk < 0.2 * A:
            continue
        out.append({"i": i + 1, "dir": d, "entry": float(e), "stop": float(sl), "target": float(e + d * 2 * risk)})
    return out


SEUILS = {"absorption": (0.25, 0.35), "desequilibres": (3, 4), "divergence": (0.05, 0.15), "epuisement": (0.03, 0.01)}
VARIANTES = [{"model": m, "seuil": s, "tp1": p} for m, ss in SEUILS.items() for s in ss for p in (0.5, 1.0)]


def run(kind, tf="15M"):
    if kind != "cr":
        print("Footprint : crypto seulement (pas de marche central en forex)", flush=True)
        return
    steps = {"15M": 900, "1H": 3600}
    today = datetime.datetime.utcnow().date()
    days = [(today - datetime.timedelta(days=k)).isoformat() for k in range(DAYS, 0, -1)]
    data = {s: {st: [] for st in steps.values()} for s in COINS}
    t0 = time.time()
    jobs = [(s, d) for s in COINS for d in days]
    with ThreadPoolExecutor(6) as ex:
        for k, (sym, d, res) in enumerate(ex.map(lambda x: day_bars(x[0], x[1], tuple(steps.values())), jobs)):
            if res:
                for st, rows in res.items():
                    data[sym][st].extend(rows)
            if k % 100 == 0:
                print("TELECHARGE", k, "/", len(jobs), "fichiers en", round(time.time() - t0), "s", flush=True)
    out = {"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "jours": DAYS, "cryptos": COINS, "unites": {}}
    for tfn, st in steps.items():
        bars = {}
        for s in COINS:
            F = np.array(sorted(data[s][st]), dtype=float)
            if len(F) > 500:
                bars[s] = (F, _atr(np.c_[F[:, :5], F[:, 5]]))
        if not bars:
            continue
        t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
        cut = np.quantile(t_all, 0.6)
        dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
        print("UNITE", tfn, "| cryptos", len(bars), "| bougies", len(t_all), "| du",
              time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure", time.strftime("%d/%m/%Y", time.gmtime(cut)),
              flush=True)
        res = []
        for v in VARIANTES:
            tr, te = [], []
            for s, (F, atr) in bars.items():
                a = np.c_[F[:, :5], F[:, 5]]
                busy = -1
                for stp in setups(F, atr, v["model"], v["seuil"]):
                    if stp["i"] <= busy:
                        continue
                    p = play(a, stp, "cr", max_bars=48, tp1_r=v["tp1"])
                    if p is None:
                        continue
                    r, win, end = p
                    busy = end
                    (te if a[stp["i"], 0] >= cut else tr).append((r, win))

            def stats(x, days_):
                if not x:
                    return {"trades": 0}
                r = np.array([y[0] for y in x])
                return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                        "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                        "par_jour": round(len(x) / max(1, days_), 2)}
            res.append({"variante": v, "apprentissage": stats(tr, dtr), "test": stats(te, dte)})
            print("VARIANTE", tfn, json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
                  json.dumps(res[-1]["test"]), flush=True)
        best = max(res, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9)
                   if x["apprentissage"].get("trades", 0) >= 50 else -9)
        print("CHOISIE", tfn, "(sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
              json.dumps(best["test"]), flush=True)
        out["unites"][tfn] = {"choisie": best["variante"], "variantes": res}
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/footprint_cr.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "15M")
