"""Meta-etiquetage (meta-labeling, Lopez de Prado) des setups de liquidite (demande du trader, 8 nov.).
1. Modeles primaires : tous les setups de liquidite deja construits (prise de liquidite + cassure + order block,
   order block seul, confluence Fibonacci, profil de volume, Market Profile poor high / Initial Balance).
2. Pour chaque setup : contexte connu AVANT l'entree (heure, jour, session, modele, sens, risque et objectif en ATR,
   regime de volatilite, tendance 1 / 4 / 20 jours, position dans la journee d'hier, distance aux zones de
   liquidite, volume relatif, bougie precedente, nombre de modeles d'accord au meme moment).
3. Une IA (gradient boosting) apprend sur les 60 % de trades les plus anciens lesquels finissent gagnants ; le seuil
   de confiance est choisi sur les 20 % suivants ; verdict sur les 20 % les plus recents (jamais vus), un trade a la
   fois par actif, frais deduits, regle stricte (objectif jamais compte sur la bougie d'entree).
   python -m lab.meta fx 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab import banque, fibo, ob_retour, tpo, vprofil
from lab.liq2liq import _atr, pools

log = logging.getLogger("lab")
MODELES = ["banque", "ob", "fibo", "vp_balayage", "vp_80", "tpo_poor", "tpo_ib_faux"]
TP1 = 0.5


def primary(a, vol, atr, P, kind):
    kz = kind == "fx"
    out = []
    for nm, st in (("banque", banque.setups(a, P, atr, "bord", False, "toutes", kz=kz)),
                   ("ob", ob_retour.setups(a, atr, 1.0, "bord", "2R", kz=kz)),
                   ("fibo", fibo.setups(a, atr, {N: fibo.pivots(a[:, 2], a[:, 3], N) for N in fibo.SCALES},
                                        "or", 0.25, "origine", kz=kz)),
                   ("vp_balayage", vprofil.setups(a, vol, atr, "balayage", 1, "bord", kz=kz)),
                   ("vp_80", vprofil.setups(a, vol, atr, "80", 2, "bord", kz=kz)),
                   ("tpo_poor", tpo.setups(a, atr, "poor", 2, kz=kz)),
                   ("tpo_ib_faux", tpo.setups(a, atr, "ib_faux", 4, kz=kz))):
        for s in st:
            s["modele"] = MODELES.index(nm)
            out.append(s)
    return out


def features(a, vol, atr, P, s, conf, aid):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    m, d, e = s["i"], s["dir"], s["entry"]
    j = m - 1                                              # derniere bougie fermee avant l'entree
    A = atr[j]
    risk = (e - s["stop"]) * d

    def mom(n):
        return d * (c[j] - c[j - n]) / A if j - n >= 0 else np.nan

    def dist(nm):
        x = P[nm][j]
        return d * (x - e) / A if np.isfinite(x) else np.nan
    pdh, pdl = P["pdh"][j], P["pdl"][j]
    pos = (e - pdl) / (pdh - pdl) if np.isfinite(pdh) and pdh > pdl else np.nan
    if d == -1 and np.isfinite(pos):
        pos = 1 - pos
    vr = vol[j] / np.median(vol[max(0, j - 96):j]) if j > 10 and np.median(vol[max(0, j - 96):j]) > 0 else np.nan
    med_atr = np.nanmedian(atr[max(0, j - 480):j]) if j > 50 else np.nan
    return [s["modele"], d, (ts[m] % 86400) // 3600, ((ts[m] // 86400) + 3) % 7, aid, risk / A,
            (s["target"] - e) * d / risk, A / med_atr if med_atr else np.nan, mom(24), mom(96), mom(480), pos,
            dist("pdh"), dist("pdl"), dist("ash"), dist("asl"), dist("pwh"), dist("pwl"), vr,
            (h[j] - l[j]) / A, d * (c[j] - o[j]) / A, conf]


NOMS = ["modele", "sens", "heure", "jour", "actif", "risque_atr", "objectif_r", "regime_vol", "tendance_24", "tendance_96",
        "tendance_480", "position_veille", "dist_haut_veille", "dist_bas_veille", "dist_haut_asie", "dist_bas_asie",
        "dist_haut_semaine", "dist_bas_semaine", "volume_relatif", "taille_bougie", "corps_bougie", "modeles_accord"]


def run(kind, tf="1H"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from lab import data
    from lab.config import UNIVERSE
    rows = []
    for aid, sym in enumerate(UNIVERSE[kind]):
        try:
            a = data.long_bars(sym, kind, tf)
            if len(a) <= 1500:
                continue
            vol = a[:, 5].copy()
            vt = getattr(data, "VOLUME", {}).get(sym)
            if kind == "fx" and vt is not None:
                mp = dict(zip(vt[0].astype(int), vt[1]))
                vol = np.array([mp.get(int(t), np.nan) for t in a[:, 0]])
            if not np.isfinite(vol).any() or np.nansum(vol) <= 0:
                vol = np.ones(len(a))
            vol = np.where(np.isfinite(vol), vol, np.nanmedian(vol))
            atr, P = _atr(a), pools(a)
            st = [s for s in primary(a, vol, atr, P, kind) if s["i"] >= 500]
            cnt = {}
            for s in st:
                for dm in (-1, 0, 1):
                    cnt.setdefault((s["i"] + dm, s["dir"]), set()).add(s["modele"])
            seen = set()
            for s in sorted(st, key=lambda x: (x["i"], x["modele"])):
                if (s["i"], s["dir"]) in seen:
                    continue                                    # meme bougie, meme sens : un seul trade
                seen.add((s["i"], s["dir"]))
                p = ob_retour.play(a, s, kind, tp1_r=TP1)
                if p is None:
                    continue
                r, win, end = p
                rows.append((a[s["i"], 0], a[end, 0], aid, r,
                             features(a, vol, atr, P, s, len(cnt[(s["i"], s["dir"])]), aid)))
        except Exception as ex:
            log.warning("%s : %s", sym, ex)
    rows.sort(key=lambda x: x[0])
    T = np.array([r[0] for r in rows])
    E = np.array([r[1] for r in rows])
    AID = np.array([r[2] for r in rows])
    R = np.array([r[3] for r in rows])
    X = np.array([r[4] for r in rows], dtype=float)
    y = (R > 0).astype(int)
    q1, q2 = np.quantile(T, [0.6, 0.8])
    tr = (T < q1) & (E < q1)                                 # embargo : le resultat doit etre connu avant q1
    va = (T >= q1) & (T < q2) & (E < q2)
    te = T >= q2
    print("SETUPS", len(R), "| apprentissage", int(tr.sum()), "| validation", int(va.sum()), "| test", int(te.sum()),
          "| reussite brute (tous setups, test)", round(float(y[te].mean()), 3), "gain moyen", round(float(R[te].mean()), 3),
          flush=True)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=80,
                                         l2_regularization=1.0, categorical_features=[0, 4], random_state=0)
    clf.fit(X[tr], y[tr])
    pr = clf.predict_proba(X)[:, 1]
    days_va, days_te = (q2 - q1) / 86400, (T.max() - q2) / 86400

    def select(mask, th):
        """Trades retenus (confiance >= seuil), un a la fois par actif."""
        busy, keep = {}, []
        for k in np.where(mask & (pr >= th))[0]:
            if T[k] < busy.get(AID[k], -1):
                continue
            busy[AID[k]] = E[k]
            keep.append(k)
        return np.array(keep, dtype=int)

    def stats(k, days):
        if len(k) == 0:
            return {"trades": 0}
        return {"trades": int(len(k)), "reussite": round(float((R[k] > 0).mean()), 3),
                "gain_moyen_R": round(float(R[k].mean()), 3), "total_R": round(float(R[k].sum()), 1),
                "par_jour": round(len(k) / max(1, days), 2)}
    table = []
    for th in np.round(np.arange(0.5, 0.96, 0.05), 2):
        sv, st_ = stats(select(va, th), days_va), stats(select(te, th), days_te)
        table.append({"seuil": float(th), "validation": sv, "test": st_})
        print("SEUIL", th, "| VALIDATION", json.dumps(sv), "| TEST", json.dumps(st_), flush=True)
    ok = [t for t in table if t["validation"].get("trades", 0) >= 30]
    best = max(ok, key=lambda t: t["validation"]["gain_moyen_R"]) if ok else table[0]
    print("CHOISI (sur la validation seulement) seuil", best["seuil"], "-> TEST (jamais vu)", json.dumps(best["test"]),
          flush=True)
    try:
        from sklearn.inspection import permutation_importance
        sub = np.where(va)[0][:20000]
        imp = permutation_importance(clf, X[sub], y[sub], n_repeats=3, random_state=0, scoring="roc_auc")
        top = sorted(zip(NOMS, imp.importances_mean), key=lambda x: -x[1])[:8]
        print("IMPORTANT", json.dumps([(n, round(float(v), 4)) for n, v in top]), flush=True)
        from sklearn.metrics import roc_auc_score
        print("AUC validation", round(float(roc_auc_score(y[va], pr[va])), 3), "test",
              round(float(roc_auc_score(y[te], pr[te])), 3), flush=True)
    except Exception as ex:
        log.warning("importance : %s", ex)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/meta_%s_%s.json" % (kind, tf), "w") as f:
        json.dump({"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                   "setups": int(len(R)), "seuil_choisi": best["seuil"], "seuils": table}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
