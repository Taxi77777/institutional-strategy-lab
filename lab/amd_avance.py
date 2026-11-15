"""AMD avance (demande du trader, 15 nov., apres recherche) : les elements ICT absents du premier test AMD.
Tout en HEURE DE NEW YORK (changement d'heure compris).
A. PO3 du jour :
   - ouverture vraie du jour = 00 h New York ; Asie = 19 h - 00 h New York ;
   - manipulation : killzone de Londres (2 h - 5 h) ou de New York (7 h - 10 h) : le prix chasse UN cote de l'Asie
     ET passe du « mauvais » cote de l'ouverture de minuit (achat seulement si la chasse est passee sous minuit) ;
   - option SMT : la paire jumelle (EURUSD/GBPUSD, AUDUSD/NZDUSD, BTC/ETH, altcoin/BTC) ne chasse PAS son Asie ;
   - entree : cassure du dernier sommet (MSS) puis retour dans le FVG de l'impulsion (ou entree au marche) ;
   - stop : au-dela de l'extreme de la manipulation (+0,1 ATR) ; objectif : autre bord de l'Asie, ou 1 / 2
     deviations du mouvement de manipulation ; moitie a mi-chemin puis stop a l'entree ; sortie a 12 h (Londres)
     ou 16 h New York.
B. CRT (Candle Range Theory) : bougie 4 h (blocs New York 1 h, 5 h, 9 h...) ou jour (minuit New York) ; la bougie
   suivante chasse le bas (haut) puis cloture DANS la fourchette -> achat (vente) a l'ouverture suivante ; stop
   au-dela de la meche ; moitie au milieu de la fourchette, reste a l'autre extreme ; duree max 1 ou 2 bougies.
Regle stricte (objectif jamais compte sur la bougie d'entree). Test honnete : 60 % anciens pour choisir, 40 %
recents (jamais vus).   python -m lab.amd_avance fx"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.elder3 import play2
from lab.liq2liq import _atr

log = logging.getLogger("lab")
JUMELLE = {"EURUSD": "GBPUSD", "GBPUSD": "EURUSD", "AUDUSD": "NZDUSD", "NZDUSD": "AUDUSD", "BTC": "ETH", "ETH": "BTC"}


def ny(ts):
    import pandas as pd
    t = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
    hour = (t.hour + t.minute / 60.0).to_numpy()
    day = np.asarray(t.tz_localize(None).to_numpy(), dtype="datetime64[D]").astype(np.int64)   # jour New York
    return hour, day


def po3(a, atr, hr, dy, part=None, kz="londres", smt=False, entree="fvg", cible="asie"):
    ts, o, h, l, c = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    # journee de session : 19 h (veille) -> 16 h ; on la rattache au jour civil New York suivant si heure >= 19 h
    sess = dy + (hr >= 19).astype(np.int64)
    idx = {}
    for i, d in enumerate(sess):
        idx.setdefault(d, []).append(i)
    pidx = None
    if part is not None:
        pa, phr, pdy = part
        psess = pdy + (phr >= 19).astype(np.int64)
        pidx = {}
        for i, d in enumerate(psess):
            pidx.setdefault(d, []).append(i)
    k0, k1 = (2, 5) if kz == "londres" else (7, 10)
    fin = 12 if kz == "londres" else 16
    out = []
    for d, rows in idx.items():
        asie = [i for i in rows if hr[i] >= 19]
        jour = [i for i in rows if hr[i] < 19]
        if len(asie) < 8 or len(jour) < 16:
            continue
        ah, al = h[asie].max(), l[asie].min()
        m0 = next((i for i in jour if hr[i] >= 0), None)
        if m0 is None:
            continue
        mo = o[m0]                                            # ouverture vraie du jour (minuit New York)
        kzr = [i for i in jour if k0 <= hr[i] < k1]
        endr = [i for i in jour if hr[i] < fin]
        if not kzr or not endr:
            continue
        end_i = endr[-1]
        side, s = 0, None
        for i in kzr:
            up, dn = h[i] > ah, l[i] < al
            if up and dn:
                break
            if dn and l[i] < mo:
                side, s = 1, i
                break
            if up and h[i] > mo:
                side, s = -1, i
                break
        if not side:
            continue
        if smt and pidx is not None:
            prow = pidx.get(d)
            if not prow:
                continue
            pasie = [i for i in prow if phr[i] >= 19]
            if len(pasie) < 8:
                continue
            t_lo, t_hi = ts[kzr[0]], ts[s]
            pw = [i for i in prow if t_lo <= pa[i, 0] <= t_hi]
            if not pw:
                continue
            if side == 1 and pa[pw, 3].min() < pa[pasie, 3].min():
                continue                                      # la jumelle a aussi chasse : pas de SMT
            if side == -1 and pa[pw, 2].max() > pa[pasie, 2].max():
                continue
        A = atr[s]
        if not np.isfinite(A) or A <= 0 or s < 4:
            continue
        ref = h[s - 3:s].max() if side == 1 else l[s - 3:s].min()   # dernier sommet (creux) avant la chasse
        ext = l[s] if side == 1 else h[s]
        for j in range(s, min(s + 13, end_i)):
            ext = min(ext, l[j]) if side == 1 else max(ext, h[j])
            if not ((c[j] > ref) if side == 1 else (c[j] < ref)):
                continue
            leg_hi = h[s:j + 1].max() if side == 1 else l[s:j + 1].min()
            if entree == "fvg":
                fv = None
                for k in range(s + 2, j + 1):
                    if side == 1 and l[k] > h[k - 2]:
                        fv = l[k]
                    if side == -1 and h[k] < l[k - 2]:
                        fv = h[k]
                if fv is None:
                    break
                e, start = fv, j + 1
            else:
                e, start = o[j + 1], j + 1
            sl = ext - side * 0.1 * A
            risk = (e - sl) * side
            if cible == "asie":
                tg = ah if side == 1 else al
            else:
                n_dev = 1.0 if cible == "dev1" else 2.0
                tg = leg_hi + side * n_dev * abs(leg_hi - ext)
            if risk < 0.2 * A or (tg - e) * side < risk:
                break
            for m in range(start, end_i + 1):
                if entree != "fvg" or ((l[m] <= e) if side == 1 else (h[m] >= e)):
                    out.append({"i": m, "dir": side, "entry": float(e), "stop": float(sl),
                                "tp1": float(e + 0.5 * (tg - e)), "target": float(tg), "fin": int(end_i)})
                    break
                if (h[m] >= tg) if side == 1 else (l[m] <= tg):
                    break
            break
    out.sort(key=lambda x: x["i"])
    return out


def crt(a, atr, hr, dy, htf="4H", force=False, duree=1):
    """a : bougies 1H. Bougies de grande unite construites en heure de New York."""
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    if htf == "4H":
        blk = (hr.astype(int) - 1) // 4                     # blocs 1-5, 5-9, 9-13, 13-17, 17-21, 21-1
        key = dy * 10 + np.where(hr < 1, -1, blk)
    else:
        key = dy
    grp = []
    cur, start = None, 0
    for i, k in enumerate(key):
        if k != cur:
            if cur is not None:
                grp.append((start, i - 1))
            cur, start = k, i
    grp.append((start, len(key) - 1))
    nb = 4 if htf == "4H" else 24
    out = []
    for g in range(1, len(grp) - 1):
        (a0, a1), (b0, b1) = grp[g - 1], grp[g]
        if a1 - a0 + 1 < nb * 0.75 or b1 - b0 + 1 < nb * 0.75 or b1 + 1 >= len(a):
            continue
        rh, rl = h[a0:a1 + 1].max(), l[a0:a1 + 1].min()
        mh, ml, mc = h[b0:b1 + 1].max(), l[b0:b1 + 1].min(), c[b1]
        mid = (rh + rl) / 2
        A = atr[b1]
        if not np.isfinite(A) or A <= 0 or rh <= rl:
            continue
        for d in (1, -1):
            if d == 1 and not (ml < rl and rl < mc < rh and mh <= rh):
                continue
            if d == -1 and not (mh > rh and rl < mc < rh and ml >= rl):
                continue
            if force and ((mc < mid) if d == 1 else (mc > mid)):
                continue                                      # cloture au-dela du milieu exigee
            e = o[b1 + 1]
            sl = (ml - 0.1 * A) if d == 1 else (mh + 0.1 * A)
            tg = rh if d == 1 else rl
            t1 = mid if (mid - e) * d > 0 else e + 0.5 * (tg - e)   # deja au-dela du milieu : mi-chemin
            risk = (e - sl) * d
            if risk < 0.2 * A or (t1 - e) * d <= 0:              # CRT : pas de minimum de gain/risque (regle ICT)
                continue
            out.append({"i": b1 + 1, "dir": d, "entry": float(e), "stop": float(sl), "tp1": float(t1),
                        "target": float(tg), "fin": int(min(len(a) - 1, b1 + duree * nb))})
    return out


VAR_PO3 = [{"famille": "po3", "kz": k, "smt": s, "entree": e, "cible": t} for k in ("londres", "ny")
           for s in (False, True) for e in ("fvg", "marche") for t in ("asie", "dev1", "dev2")]
VAR_CRT = [{"famille": "crt", "htf": f, "force": fo, "duree": du} for f in ("4H", "1D") for fo in (False, True)
           for du in (1, 2)]


def run(kind, tf=""):
    from lab import data
    from lab.config import UNIVERSE
    B = {"15M": {}, "1H": {}}
    for s in UNIVERSE[kind]:
        for u in B:
            try:
                a = data.long_bars(s, kind, u)
                if len(a) > 1500:
                    hr, dy = ny(a[:, 0])
                    B[u][s] = (a, _atr(a), hr, dy)
            except Exception as ex:
                log.warning("%s %s : %s", s, u, ex)
    print("ACTIFS", kind, {u: len(v) for u, v in B.items()}, flush=True)

    def partner(s):
        p = JUMELLE.get(s) or ("BTC" if kind == "cr" else None)
        return p if p in B["15M"] and p != s else None
    res = []
    for fam, VAR, u in (("po3", VAR_PO3, "15M"), ("crt", VAR_CRT, "1H")):
        bars = B[u]
        if not bars:
            continue
        t_all = np.concatenate([b[0][:, 0] for b in bars.values()])
        cut = np.quantile(t_all, 0.6)
        dtr, dte = (cut - t_all.min()) / 86400, (t_all.max() - cut) / 86400
        for v in VAR:
            tr, te = [], []
            for s, (a, atr, hr, dy) in bars.items():
                if fam == "po3":
                    p = partner(s)
                    if v["smt"] and not p:
                        continue
                    pp = None
                    if p:
                        pa, _, phr, pdy = B["15M"][p]
                        pp = (pa, phr, pdy)
                    st = po3(a, atr, hr, dy, pp, v["kz"], v["smt"], v["entree"], v["cible"])
                else:
                    st = crt(a, atr, hr, dy, v["htf"], v["force"], v["duree"])
                busy = -1
                for x in st:
                    if x["i"] <= busy:
                        continue
                    r_ = play2(a, x, kind, max_bars=max(1, x["fin"] - x["i"] + 1))
                    if r_ is None:
                        continue
                    r, win, end = r_
                    busy = end
                    (te if a[x["i"], 0] >= cut else tr).append((r, r > 0, x["dir"]))

            def stats(x, days):
                if not x:
                    return {"trades": 0}
                r = np.array([y[0] for y in x])
                return {"trades": len(x), "reussite": round(float(np.mean([y[1] for y in x])), 3),
                        "gain_moyen_R": round(float(r.mean()), 3), "total_R": round(float(r.sum()), 1),
                        "par_jour": round(len(x) / max(1, days), 2),
                        "achats_R": round(float(sum(y[0] for y in x if y[2] == 1)), 1),
                        "ventes_R": round(float(sum(y[0] for y in x if y[2] == -1)), 1)}
            res.append({"variante": v, "apprentissage": stats(tr, dtr), "test": stats(te, dte)})
            print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
                  json.dumps(res[-1]["test"]), flush=True)
    for fam in ("po3", "crt"):
        R = [x for x in res if x["variante"]["famille"] == fam]
        if R:
            best = max(R, key=lambda x: x["apprentissage"].get("gain_moyen_R", -9)
                       if x["apprentissage"].get("trades", 0) >= 50 else -9)
            print("CHOISIE", fam, "(sur l'apprentissage seulement)", json.dumps(best["variante"]), "-> TEST (jamais vu)",
                  json.dumps(best["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_avance_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f,
                  indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1])
