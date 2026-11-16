"""AMD a plusieurs echelles (demande du trader, 16 nov. : AMD ICT au resultat exceptionnel, intraday / court terme).
Bougies 1 h, jours UTC, sens = tendance journaliere quand elle est demandee (cloture > SMA 20 j et > cloture d'il y a
90 j pour un achat, l'inverse pour une vente), toujours connue AVANT la seance traitee. Trois modeles :
  HEBDO   AMD hebdomadaire : lundi = accumulation ; mardi (ou mardi-mercredi) le prix passe sous le bas du lundi
          (achat) d'au moins m ATR jour, puis une cloture 1 h revient au-dessus -> entree a l'ouverture suivante ;
          stop sous la meche ; objectif 1R, 2R ou « sem » (plus haut de la semaine precedente, au moins 1R, moitie a
          1R puis stop a l'entree) ; tout est ferme le vendredi a 20 h UTC au plus tard.
  DOUBLE  Double AMD : Asie 0-7 h = accumulation, Londres 7-12 h balaie un cote de l'Asie et revient dedans
          (1re confirmation, sens oppose au balayage) ; puis New York 12-16 h balaie le cote correspondant de
          Londres (bas de Londres pour un achat) et revient dedans (2e confirmation, meme sens) -> entree.
          Niveau balaye a New York : bas de Londres (« londres ») ou de nouveau bas de l'Asie (« asie »).
          Stop structurel sous la meche de New York.
          Objectif 1R, 2R ou « fin » (on garde jusqu'a 20 h UTC).
  REGIME  AMD journalier Asie -> Londres filtre : Asie compressee (taille < x ATR jour), ATR 20 j / ATR 100 j dans le
          regime voulu (« bas » < 0,9, « haut » > 1,1), pente de la SMA 20 sur 5 jours > s ATR (sens = pente,
          cloture du bon cote de la SMA). Stop structurel derriere la meche, objectifs 1R / 2R / « liq » (plus haut de
          la veille, au moins 1R, moitie a 1R puis stop a l'entree) ; ferme a 20 h UTC.
Stop teste avant l'objectif sur chaque bougie ; frais COST deduits ; stop absurde (< 0,05 ATR ou trop loin) ignore.
Test honnete : coupure temporelle 60 / 40 ; choix sur l'apprentissage seulement (t >= 2, >= 150 trades).
python -m lab.amd_echelles cr 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

from lab.liq2liq import COST
from lab.tendance_amd import daily

log = logging.getLogger("lab")
BUF = 0.05


def _prep(a):
    D, st, en, atr = daily(a)
    pc = np.r_[D[0, 4], D[:-1, 4]]
    tr = np.maximum(D[:, 2] - D[:, 3], np.maximum(abs(D[:, 2] - pc), abs(D[:, 3] - pc)))
    atr100 = np.full(len(D), np.nan)
    cs = np.cumsum(tr)
    atr100[100:] = (cs[100:] - cs[:-100]) / 100
    sma = np.full(len(D), np.nan)
    cc = np.cumsum(D[:, 4])
    sma[19:] = (cc[19:] - np.r_[0, cc[:-20]]) / 20
    tend = np.zeros(len(D), int)
    for i in range(90, len(D)):
        if D[i, 4] > sma[i] and D[i, 4] > D[i - 90, 4]:
            tend[i] = 1
        elif D[i, 4] < sma[i] and D[i, 4] < D[i - 90, 4]:
            tend[i] = -1
    return D, st, en, atr, atr100, sma, tend


def _gerer(a, kind, k_in, t_end, d, e, stop, tg, half_at):
    """Gestion bougie par bougie jusqu'a t_end (exclu) ; stop teste avant l'objectif. Renvoie (t_in, t_out, d, R)."""
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    risk = (e - stop) * d
    half, x, last, j = False, None, k_in, k_in
    while j < len(a) and a[j, 0] < t_end:
        last = j
        if (l[j] <= stop) if d == 1 else (h[j] >= stop):
            x = stop if j == k_in else (min(o[j], stop) if d == 1 else max(o[j], stop))
            break
        if half_at is not None and not half and ((h[j] >= half_at) if d == 1 else (l[j] <= half_at)):
            half, stop = True, e
        if tg is not None and ((h[j] >= tg) if d == 1 else (l[j] <= tg)):
            x = tg
            break
        j += 1
    if x is None:
        x = c[last]
    rr = (0.5 + 0.5 * (x - e) * d / risk) if half else (x - e) * d / risk
    return (a[k_in, 0], a[last, 0], d, float(rr - COST[kind] * e / risk))


def _cible(cible, d, e, risk, niveau):
    if cible == "1R":
        return e + d * risk, None
    if cible == "2R":
        return e + d * 2 * risk, None
    if cible == "fin":
        return None, None
    tg = niveau
    if tg is None or (tg - e) * d < risk:
        tg = e + d * risk
    return tg, e + d * risk


def _balayage(a, hr, rows, h_a, h_b, d, lo_, hi_, m_abs=0.0):
    """Le prix passe sous lo_ (achat) / sur hi_ (vente) entre h_a et h_b puis une cloture 1 h revient dedans.
    Renvoie (indice d'entree, extreme de la meche) ou (None, None)."""
    l, h, c = a[:, 3], a[:, 2], a[:, 4]
    ext = None
    for r in rows:
        if hr[r] < h_a:
            continue
        if hr[r] >= h_b and ext is None:
            break
        if hr[r] >= h_b + 1:
            break
        w = l[r] if d == 1 else h[r]
        if ext is None and ((w < lo_ - m_abs) if d == 1 else (w > hi_ + m_abs)):
            ext = w
        elif ext is not None:
            ext = min(ext, w) if d == 1 else max(ext, w)
        if ext is not None and ((c[r] > lo_) if d == 1 else (c[r] < hi_)):
            return r + 1, ext
    return None, None


def trades_hebdo(a, kind, P, cible="2R", m=0.0, fen="mar"):
    o, h, l = a[:, 1], a[:, 2], a[:, 3]
    D, st, en, atr, atr100, sma, tend = P
    out = []
    for i in range(91, len(D) - 1):
        if (int(D[i, 0]) + 3) % 7 != 0:                      # lundi
            continue
        d, A = tend[i], atr[i]
        if not d or not (np.isfinite(A) and A > 0):
            continue
        mon = int(D[i, 0])
        jours = [i + k for k in (1, 2) if i + k < len(D) and int(D[i + k, 0]) - mon == k]
        if fen == "mar":
            jours = jours[:1]
        if not jours or int(D[jours[0], 0]) - mon != 1:
            continue
        rows = [r for j in jours for r in range(st[j], en[j] + 1)]
        hr_fake = np.zeros(len(a), int)                        # toute la fenetre est valable
        k_in, ext = _balayage(a, hr_fake, rows, 0, 24, d, D[i, 3], D[i, 2], m * A)
        if k_in is None or k_in >= len(a):
            continue
        t_end = (mon + 4) * 86400 + 20 * 3600
        if a[k_in, 0] >= t_end:
            continue
        e = o[k_in]
        stop = ext - d * BUF * A
        risk = (e - stop) * d
        if risk <= 0.05 * A or risk > 2.0 * A:
            continue
        s0, s1 = np.searchsorted(D[:, 0], mon - 7), i
        pw = (D[s0:s1, 2].max() if d == 1 else D[s0:s1, 3].min()) if s1 > s0 else None
        tg, half_at = _cible(cible, d, e, risk, pw)
        out.append(_gerer(a, kind, k_in, t_end, d, e, stop, tg, half_at))
    return out


def trades_double(a, kind, P, biais="aucun", niveau="londres", cible="2R"):
    o, h, l = a[:, 1], a[:, 2], a[:, 3]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr, atr100, sma, tend = P
    out = []
    for i in range(91, len(D) - 1):
        A = atr[i]
        if not (np.isfinite(A) and A > 0):
            continue
        rows = list(range(st[i + 1], en[i + 1] + 1))
        if len(rows) < 16:
            continue
        asie = [r for r in rows if hr[r] < 7]
        if len(asie) < 4:
            continue
        al, ah = l[asie].min(), h[asie].max()
        sens = (tend[i],) if biais == "tendance" else (1, -1)
        best = None
        for d in sens:
            if not d:
                continue
            k1, e1 = _balayage(a, hr, rows, 7, 12, d, al, ah)
            if k1 is not None and (best is None or k1 < best[1]):
                best = (d, k1, e1)
        if best is None:
            continue
        d, k1, e1 = best
        lond = [r for r in rows if 7 <= hr[r] < 12]
        ll, lh = (l[lond].min(), h[lond].max()) if niveau == "londres" else (al, ah)
        rows2 = [r for r in rows if r >= k1]
        k_in, ext = _balayage(a, hr, rows2, 12, 16, d, ll, lh)
        if k_in is None or k_in >= len(a) or hr[k_in] >= 20 or k_in > rows[-1]:
            continue
        e = o[k_in]
        stop = ext - d * BUF * A
        risk = (e - stop) * d
        if risk <= 0.05 * A or risk > 1.5 * A:
            continue
        tg, half_at = _cible(cible, d, e, risk, None)
        out.append(_gerer(a, kind, k_in, D[i + 1, 0] * 86400 + 20 * 3600, d, e, stop, tg, half_at))
    return out


def trades_regime(a, kind, P, x=0.4, regime="bas", s=0.3, cible="2R"):
    o, h, l = a[:, 1], a[:, 2], a[:, 3]
    hr = ((a[:, 0] % 86400) // 3600).astype(int)
    D, st, en, atr, atr100, sma, tend = P
    out = []
    for i in range(101, len(D) - 1):
        A, B = atr[i], atr100[i]
        if not (np.isfinite(A) and A > 0 and np.isfinite(B) and B > 0):
            continue
        q = A / B
        if (regime == "bas" and q >= 0.9) or (regime == "haut" and q <= 1.1):
            continue
        pente = (sma[i] - sma[i - 5]) / A
        if pente > s and D[i, 4] > sma[i]:
            d = 1
        elif pente < -s and D[i, 4] < sma[i]:
            d = -1
        else:
            continue
        rows = list(range(st[i + 1], en[i + 1] + 1))
        if len(rows) < 16:
            continue
        asie = [r for r in rows if hr[r] < 7]
        if len(asie) < 4:
            continue
        al, ah = l[asie].min(), h[asie].max()
        if ah - al >= x * A:
            continue
        k_in, ext = _balayage(a, hr, rows, 7, 12, d, al, ah)
        if k_in is None or k_in >= len(a) or hr[k_in] >= 20 or k_in > rows[-1]:
            continue
        e = o[k_in]
        stop = ext - d * BUF * A
        risk = (e - stop) * d
        if risk <= 0.05 * A or risk > 1.5 * A:
            continue
        tg, half_at = _cible(cible, d, e, risk, D[i, 2] if d == 1 else D[i, 3])
        out.append(_gerer(a, kind, k_in, D[i + 1, 0] * 86400 + 20 * 3600, d, e, stop, tg, half_at))
    return out


MODELES = {"HEBDO": trades_hebdo, "DOUBLE": trades_double, "REGIME": trades_regime}
VARIANTES = ([{"modele": "HEBDO", "cible": t, "m": m, "fen": f} for t in ("1R", "2R", "sem") for m in (0.0, 0.25)
              for f in ("mar", "mar-mer")]
             + [{"modele": "DOUBLE", "biais": b, "niveau": nv, "cible": t} for b in ("aucun", "tendance")
                for nv in ("londres", "asie") for t in ("1R", "2R", "fin")]
             + [{"modele": "REGIME", "x": x, "regime": rg, "s": s, "cible": t} for x in (0.3, 0.5) for rg in ("bas", "haut")
                for s in (0.2, 0.5) for t in ("1R", "2R", "liq")])


def trades(a, kind, modele="HEBDO", P=None, **kw):
    return MODELES[modele](a, kind, P if P is not None else _prep(a), **kw)


def stats(x):
    if not x:
        return {"trades": 0}
    x = sorted(x, key=lambda t: t[1])
    r = np.array([t[3] for t in x])
    eq = np.cumsum(r)
    yrs = max(0.25, (x[-1][1] - x[0][0]) / 86400 / 365)
    w, lo = r[r > 0], r[r <= 0]
    return {"trades": len(r), "par_jour": round(len(r) / yrs / 365, 2), "reussite": round(float((r > 0).mean()), 3),
            "gain_moyen_R": round(float(r.mean()), 3), "gain_moyen_gagnant": round(float(w.mean()), 3) if len(w) else 0,
            "perte_moyenne": round(float(lo.mean()), 3) if len(lo) else 0, "R_par_an": round(float(r.sum()) / yrs, 1),
            "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 and r.std() > 0 else 0,
            "pire_baisse_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1)}


def evaluer(bars, kind, cut):
    preps = {s: _prep(a) for s, a in bars.items()}
    res = []
    for v in VARIANTES:
        tr, te = [], []
        for s, a in bars.items():
            for t in trades(a, kind, P=preps[s], **v):
                (te if t[0] >= cut else tr).append(t)
        res.append({"variante": v, "apprentissage": stats(tr), "test": stats(te)})
        print("VARIANTE", json.dumps(v), "| APPRENTISSAGE", json.dumps(res[-1]["apprentissage"]), "| TEST",
              json.dumps(res[-1]["test"]), flush=True)
    ok = [x for x in res if x["apprentissage"].get("trades", 0) >= 150 and x["apprentissage"].get("t", 0) >= 2]
    print("VARIANTES t >= 2 sur l'apprentissage :", len(ok), "/", len(res), flush=True)
    for x in sorted(ok, key=lambda x: -x["apprentissage"]["t"])[:8]:
        print("   RETENUE", json.dumps(x["variante"]), "| APPRENTISSAGE", json.dumps(x["apprentissage"]), "| TEST",
              json.dumps(x["test"]), flush=True)
    if not ok:
        print("   RETENUE aucune (aucune variante t >= 2 et >= 150 trades sur l'apprentissage)", flush=True)
    return res


def run(kind, tf="1H"):
    from lab import data
    from lab.config import UNIVERSE
    bars = {}
    for s in UNIVERSE[kind]:
        try:
            a = data.long_bars(s, kind, "1H")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) < 5]
            if len(a) > 24 * 200:
                bars[s] = a[:, :5]
        except Exception as ex:
            log.warning("%s : %s", s, ex)
    t_all = np.concatenate([a[:, 0] for a in bars.values()])
    cut = np.quantile(t_all, 0.6)
    print("ACTIFS", kind, len(bars), "du", time.strftime("%d/%m/%Y", time.gmtime(t_all.min())), "coupure",
          time.strftime("%d/%m/%Y", time.gmtime(cut)), "au", time.strftime("%d/%m/%Y", time.gmtime(t_all.max())), flush=True)
    res = evaluer(bars, kind, cut)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/amd_echelles_%s.json" % kind, "w") as f:
        json.dump({"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "variantes": res}, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
