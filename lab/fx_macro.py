"""Forex : les 3 primes documentees que les fonds exploitent en devises (Asness, Moskowitz & Pedersen 2013 ;
Lustig & Verdelhan ; Menkhoff et al. 2012), plus le COT. Demande du trader (16 nov. soir) : « concentre-toi forex ».
Univers : EUR, GBP, JPY, CHF, CAD, AUD, NZD contre USD (prix Dukascopy). Chaque mois (premier jour ouvre), on classe
les 7 devises et on ACHETE les 2 meilleures / VEND les 2 moins bonnes (position neutre en dollar), tenue 1 mois :
  carry    : taux d'interet a 3 mois du pays - taux US (OCDE via FRED, decale d'un mois) ; le portefeuille touche aussi
             le differentiel de taux (le swap) ;
  momentum : rendement des 3 ou 12 derniers mois ;
  valeur   : taux de change reel (prix x CPI etranger / CPI US, CPI decale de 2 mois) sous sa moyenne 5 ans = pas cher ;
  cot      : positionnement net des fonds a levier (CFTC) en ecart-type sur 3 ans, a contre-courant ou dans le sens ;
  combo    : moyenne des rangs carry + momentum 12 mois + valeur (le « portefeuille des fonds »).
Verdict sur les 40 % recents ; le sens du COT est choisi sur les 60 % anciens.   python -m lab.fx_macro fx 1H"""
import io
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
CUR = {"EUR": ("EURUSD", 1, "EZ"), "GBP": ("GBPUSD", 1, "GB"), "JPY": ("USDJPY", -1, "JP"), "CHF": ("USDCHF", -1, "CH"),
       "CAD": ("USDCAD", -1, "CA"), "AUD": ("AUDUSD", 1, "AU"), "NZD": ("NZDUSD", 1, "NZ")}
CPI_ID = {"US": "CPALTT01USM661N", "EZ": "CP0000EZ19M086NEST", "GB": "CPALTT01GBM661N", "JP": "CPALTT01JPM661N",
          "CH": "CPALTT01CHM661N", "CA": "CPALTT01CAM661N", "AU": "CPALTT01AUQ661N", "NZ": "CPALTT01NZQ661N"}


def fred(series):
    from lab import data
    r = data.SESSION.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series}, timeout=40)
    r.raise_for_status()
    out = []
    for line in r.text.splitlines()[1:]:
        p = line.split(",")
        if len(p) < 2 or p[1] in ("", "."):
            continue
        try:
            y, m = int(p[0][:4]), int(p[0][5:7])
            out.append((y * 12 + m - 1, float(p[1])))
        except ValueError:
            continue
    return dict(out)


def run(kind, tf="1H"):
    from lab import data
    if kind != "fx":
        print("forex seulement")
        return
    # ---- prix de fin de mois (valeur de la devise en dollars)
    px = {}
    for c, (sym, sg, _) in CUR.items():
        a = data.long_bars(sym, "fx", "1H")
        mon = np.array([time.gmtime(t).tm_year * 12 + time.gmtime(t).tm_mon - 1 for t in a[:, 0]])
        last = np.r_[mon[1:] != mon[:-1], True]
        px[c] = {int(m): (a[k, 4] if sg > 0 else 1.0 / a[k, 4]) for k, m in zip(np.flatnonzero(last), mon[last])}
    months = sorted(set.intersection(*[set(v) for v in px.values()]))
    print("MOIS", len(months), "du", months[0] // 12, months[0] % 12 + 1, "au", months[-1] // 12, months[-1] % 12 + 1, flush=True)
    # ---- taux et CPI
    rates, cpi = {}, {}
    for cc in ["US"] + [v[2] for v in CUR.values()]:
        for sid in ("IR3TIB01%sM156N" % cc, "IRSTCI01%sM156N" % cc):
            try:
                rates[cc] = fred(sid)
                if rates[cc]:
                    print("TAUX", cc, sid, len(rates[cc]), "mois", flush=True)
                    break
            except Exception as ex:
                log.warning("FRED %s : %s", sid, ex)
        try:
            s = fred(CPI_ID[cc])
            if CPI_ID[cc].endswith("Q661N"):                      # trimestriel -> mensuel (report)
                full, lastv = {}, None
                for m in range(min(s), max(s) + 3):
                    lastv = s.get(m, lastv)
                    full[m] = lastv
                s = full
            cpi[cc] = s
        except Exception as ex:
            log.warning("CPI %s : %s", cc, ex)

    def lagged(d, m, lag):
        for k in range(m - lag, m - lag - 6, -1):                # derniere valeur publiee
            if k in d:
                return d[k]
        return np.nan
    cot = {}
    try:
        cot = data.cot(11)
    except Exception as ex:
        log.warning("COT : %s", ex)
    curs = list(CUR)
    R = np.array([[px[c][months[i + 1]] / px[c][months[i]] - 1 for c in curs] for i in range(len(months) - 1)])
    carry_acc = np.zeros_like(R)
    sig = {k: np.full_like(R, np.nan) for k in ("carry", "mom3", "mom12", "valeur", "cot")}
    for i, m in enumerate(months[:-1]):
        for j, c in enumerate(curs):
            cc = CUR[c][2]
            i_f, i_u = lagged(rates.get(cc, {}), m, 1), lagged(rates.get("US", {}), m, 1)
            if np.isfinite(i_f) and np.isfinite(i_u):
                sig["carry"][i, j] = i_f - i_u
                carry_acc[i, j] = (i_f - i_u) / 100 / 12
            if i >= 3:
                sig["mom3"][i, j] = px[c][m] / px[c][months[i - 3]] - 1
            if i >= 12:
                sig["mom12"][i, j] = px[c][m] / px[c][months[i - 12]] - 1
            if cc in cpi and "US" in cpi and i >= 0:
                def rer(mm):
                    a, b = lagged(cpi[cc], mm, 2), lagged(cpi["US"], mm, 2)
                    return np.log(px[c][mm] * a / b) if (mm in px[c] and np.isfinite(a) and np.isfinite(b)) else np.nan
                hist = [rer(mm) for mm in months[max(0, i - 60):i + 1]]
                hist = [h for h in hist if np.isfinite(h)]
                if len(hist) >= 36:
                    sig["valeur"][i, j] = -(hist[-1] - np.mean(hist))
            if c in cot and len(cot[c]):
                t_m = time.mktime((m // 12, m % 12 + 1, 1, 0, 0, 0, 0, 0, 0))
                ok = cot[c][:, 0] <= t_m
                if ok.sum() > 52:
                    x = cot[c][ok, 1][-156:]
                    sig["cot"][i, j] = (x[-1] - x.mean()) / (x.std() + 1e-9)
    t_month = np.array(months[:-1])
    cut = t_month[int(len(t_month) * 0.6)]

    def port(score, sens=1, with_carry=False):
        out = np.full(len(R), np.nan)
        for i in range(len(R)):
            s = score[i]
            ok = np.flatnonzero(np.isfinite(s))
            if len(ok) < 5:
                continue
            o = ok[np.argsort(sens * s[ok])]
            lo, hi = o[:2], o[-2:]
            r = R[i] + (carry_acc[i] if True else 0)          # le porteur touche toujours le differentiel de taux
            out[i] = (r[hi].mean() - r[lo].mean()) / 2 - 2 * 1.5e-4
        return out

    def st(x):
        x = x[np.isfinite(x)]
        if len(x) < 12:
            return {"mois": int(len(x))}
        eq = np.cumsum(x)
        return {"mois": int(len(x)), "rendement_pct_an": round(float(x.mean() * 12 * 100), 2),
                "sharpe": round(float(x.mean() / x.std() * np.sqrt(12)), 2) if x.std() > 0 else 0,
                "mois_gagnants": round(float((x > 0).mean()), 3),
                "pire_baisse_pct": round(float((np.maximum.accumulate(eq) - eq).max() * 100), 2)}
    res = {}

    def rank(a):
        out = np.full_like(a, np.nan)
        for i in range(len(a)):
            ok = np.flatnonzero(np.isfinite(a[i]))
            if len(ok):
                out[i, ok] = np.argsort(np.argsort(a[i, ok])) / max(1, len(ok) - 1)
        return out
    combos = {"carry": (sig["carry"], 1), "momentum_3_mois": (sig["mom3"], 1), "momentum_12_mois": (sig["mom12"], 1),
              "valeur": (sig["valeur"], 1),
              "combo_carry_momentum_valeur": (np.nanmean(np.stack([rank(sig["carry"]), rank(sig["mom12"]), rank(sig["valeur"])]), axis=0), 1),
              "combo_carry_momentum": (np.nanmean(np.stack([rank(sig["carry"]), rank(sig["mom12"])]), axis=0), 1)}
    # COT : sens choisi sur l'apprentissage
    best = None
    for sens in (1, -1):
        p = port(sig["cot"], sens)
        v = st(p[t_month < cut]).get("sharpe", -9)
        if best is None or v > best[0]:
            best = (v, sens)
    combos["cot_" + ("dans_le_sens_des_fonds" if best[1] > 0 else "contre_les_fonds")] = (sig["cot"], best[1])
    for nm, (s, sens) in combos.items():
        p = port(s, sens)
        res[nm] = {"apprentissage": st(p[t_month < cut]), "test": st(p[t_month >= cut]), "tout": st(p)}
        print("STRATEGIE", nm, "| APPRENTISSAGE", json.dumps(res[nm]["apprentissage"]), "| TEST", json.dumps(res[nm]["test"]),
              flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/fx_macro.json", "w") as f:
        json.dump({"genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "coupure_mois": int(cut),
                   "strategies": res}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
