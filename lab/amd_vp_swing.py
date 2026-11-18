"""AMD swing gagnant (lab/tendance_amd.py) + profil de volume (demande du trader, 18 nov. : AMD et volume profil).
Base : cassure 20 jours dans le sens de la cloture d'il y a 90 jours ; le lendemain, Asie 0-7 h UTC, manipulation
7-12 h UTC, reprise -> entree (sinon 16 h UTC) ; stop 4 ATR ; moitie au TP1 puis stop a l'entree ; stop suiveur
journalier. Le profil de volume (ticks Dukascopy / volume crypto reparti sur la fourchette de chaque bougie 1 h) sert a :
  filtre « val »  : la manipulation doit toucher la zone de valeur d'hier (VAL pour un achat, VAH pour une vente) ;
  filtre « hvn »  : la manipulation doit toucher le noeud de plus fort volume des 5 derniers jours sous le prix ;
  filtre « poc »  : l'entree doit etre du bon cote du POC d'hier (au-dessus pour un achat) = acceptation de la valeur ;
  TP1 « poc_nu » : la moitie est prise au POC nu (jamais retouche, 10 jours) le plus proche dans le sens du trade,
                   s'il est entre 0,3 et 2 ATR ; sinon 0,5 ATR.
Frais deduits. Choix sur les 60 % anciens, verdict sur les 40 % recents jamais vus.   python -m lab.amd_vp_swing cr 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.amd_vp import profile
from lab.liq2liq import COST
from lab.tendance_amd import daily

log = logging.getLogger("lab")


def trades(a, vol, kind, filtre="aucun", tp1="atr", k=4.0, t1=0.5, n=20, fond=90):
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr = daily(a)
    is_close = np.zeros(len(a), bool)
    is_close[en] = True
    profs, naked = {}, []

    def prof_day(i):
        if i not in profs:
            rr = np.arange(st[i], en[i] + 1)
            profs[i] = profile(h[rr], l[rr], vol[rr])
        return profs[i]
    out, i = [], max(n, fond, 21) + 1
    last_naked_day = -1
    while i < len(D) - 2:
        # mise a jour des POC nus jusqu'au jour i
        for dd in range(max(last_naked_day + 1, i - 12), i + 1):
            rr = np.arange(st[dd], en[dd] + 1)
            naked = [(p, x) for p, x in naked if not (l[rr].min() <= p <= h[rr].max()) and dd - x <= 10]
            P = prof_day(dd)
            if P is not None:
                naked.append((P["poc"], dd))
        last_naked_day = i
        d = 0
        if np.isfinite(atr[i]) and atr[i] > 0:
            if D[i, 4] > D[i - n:i, 2].max() and D[i, 4] > D[i - fond, 4]:
                d = 1
            elif D[i, 4] < D[i - n:i, 3].min() and D[i, 4] < D[i - fond, 4]:
                d = -1
        if not d:
            i += 1
            continue
        A = atr[i]
        P = prof_day(i)
        rows = range(st[i + 1], en[i + 1] + 1)
        asie = [r for r in rows if hr[r] < 7]
        m, ext = None, None
        if len(asie) >= 4:
            al, ah = l[asie].min(), h[asie].max()
            manip = False
            for r in rows:
                if hr[r] < 7:
                    continue
                if not manip and hr[r] < 12 and ((l[r] < al) if d == 1 else (h[r] > ah)):
                    manip = True
                if manip:
                    w_ = l[r] if d == 1 else h[r]
                    ext = w_ if ext is None else (min(ext, w_) if d == 1 else max(ext, w_))
                if manip and ((c[r] > al) if d == 1 else (c[r] < ah)) and r + 1 < len(a):
                    m = r + 1
                    break
                if hr[r] >= 16:
                    break
        amd = m is not None
        if m is None:
            m = next((r for r in rows if hr[r] >= 16), None)
        if m is None or m >= len(a) - 1:
            i += 1
            continue
        # filtres profil de volume
        ok = True
        if filtre == "val":
            ok = amd and P is not None and ((ext <= P["val"]) if d == 1 else (ext >= P["vah"]))
        elif filtre == "hvn":
            rr5 = np.arange(st[max(0, i - 4)], en[i] + 1)
            C5 = profile(h[rr5], l[rr5], vol[rr5])
            if amd and C5 is not None:
                mm = C5["mids"] < o[m] if d == 1 else C5["mids"] > o[m]
                if mm.any():
                    kk = np.flatnonzero(mm)[C5["prof"][mm].argmax()]
                    lvl = C5["mids"][kk]
                    ok = (ext <= lvl) if d == 1 else (ext >= lvl)
                else:
                    ok = False
            else:
                ok = False
        elif filtre == "poc":
            ok = P is not None and ((o[m] > P["poc"]) if d == 1 else (o[m] < P["poc"]))
        if not ok:
            i += 1
            continue
        e = o[m]
        risk = k * A
        tg1 = e + d * t1 * A
        if tp1 == "poc_nu":
            cands = [p for p, _ in naked if 0.3 * A <= (p - e) * d <= 2 * A]
            if cands:
                tg1 = min(cands, key=lambda p: abs(p - e))
        stop, half, best = e - d * risk, False, e
        x, j = None, m
        while j < len(a):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                x = min(o[j], stop) if (d == 1 and j > m) else (max(o[j], stop) if j > m else stop)
                break
            if not half and ((h[j] >= tg1) if d == 1 else (l[j] <= tg1)):
                half = True
                stop = max(stop, e) if d == 1 else min(stop, e)
            if is_close[j]:
                best = max(best, c[j]) if d == 1 else min(best, c[j])
                stop = max(stop, best - risk) if d == 1 else min(stop, best + risk)
            j += 1
        if x is None:
            break
        g1 = abs(tg1 - e)
        r = (0.5 * g1 / risk + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
        out.append((a[m, 0], a[j, 0], d, float(r - COST[kind] * e / risk)))
        i = int(np.searchsorted(D[:, 0], a[j, 0] // 86400)) + 1
    return out


VARIANTES = [{"filtre": f, "tp1": t, "t1": t1} for f in ("aucun", "val", "hvn", "poc") for t in ("atr", "poc_nu")
             for t1 in (0.5, 1.0)]


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars, vols = {}, {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                if s not in data.VOLUME:
                    continue
                tv, vv = data.VOLUME[s]
                mp = dict(zip(tv.astype(np.int64), vv))
                v = np.array([mp.get(int(t), np.nan) for t in a[:, 0]], float)
                keep = ((a[:, 0] // 86400 + 3) % 7) < 5
                a, v = a[keep], v[keep]
            else:
                if a.shape[1] <= 5:
                    continue
                v = a[:, 5].astype(float)
            if len(a) > 24 * 150 and np.nansum(v) > 0:
                bars[s], vols[s] = a[:, :5], v
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "coupure", time.strftime("%d/%m/%Y", time.gmtime(cut)), flush=True)

    def stats(x):
        if not x:
            return {"trades": 0}
        x = sorted(x, key=lambda t: t[1])
        r = np.array([t[3] for t in x])
        eq = np.cumsum(r)
        yrs = max(0.25, (x[-1][1] - x[0][0]) / 86400 / 365)
        w, lo = r[r > 0], r[r <= 0]
        return {"trades": len(r), "reussite": round(float((r > 0).mean()), 3), "gain_moyen_R": round(float(r.mean()), 3),
                "gain_moyen_gagnant": round(float(w.mean()), 3) if len(w) else 0,
                "perte_moyenne": round(float(lo.mean()), 3) if len(lo) else 0, "R_par_an": round(float(r.sum()) / yrs, 1),
                "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
                "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, vols[s], kind, **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    base = next(x for x in res if x["variante"] == {"filtre": "aucun", "tp1": "atr", "t1": 0.5})
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 100 and x["apprentissage"].get("t", 0) >= 2
          and x["apprentissage"].get("gain_moyen_R", 0) > base["apprentissage"].get("gain_moyen_R", 0)]
    print("BASE (sans profil) | APPRENTISSAGE", json.dumps(base["apprentissage"]), "| TEST", json.dumps(base["test"]), flush=True)
    print("VARIANTES meilleures que la base (apprentissage, t >= 2) :", len(ok), "/", len(res), flush=True)
    for x in sorted(ok, key=lambda x: -x["apprentissage"]["gain_moyen_R"])[:6]:
        print("   RETENUE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]), "| TEST",
              json.dumps(x["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_vp_swing_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
