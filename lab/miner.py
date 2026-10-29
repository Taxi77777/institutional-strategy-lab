"""La recherche de strategies (le « labo »). Methode des fonds quantitatifs :
1. chaque specialiste combine ses indicateurs (1, 2 puis 3 conditions, recherche en faisceau) sur la periode
   d'APPRENTISSAGE (60 % les plus anciens) ;
2. les meilleures combinaisons sont confirmees sur la periode de VALIDATION (20 % suivants), avec correction des
   tests multiples (Benjamini-Hochberg) : sinon, sur des millions d'essais, le hasard « trouve » toujours quelque chose ;
3. le verdict final se fait sur la periode de TEST (20 % les plus recents), jamais vue pendant la recherche :
   taux de reussite >= 80 % ET gain moyen positif apres frais, trades sans chevauchement ;
4. l'armee : on valide aussi le vote (au moins CONSENSUS specialistes d'accord) sur la periode de test ;
5. Monte Carlo (10 000 tirages) des resultats de test de chaque strategie retenue."""
import datetime
import json
import logging
import math
import random
import time
import warnings

import numpy as np

from . import data, ideas as IDEAS, indicators as IND, sim
from .config import CFG, EXITS, UNIVERSE

log = logging.getLogger("lab")
warnings.filterwarnings("ignore")
BENCH = {"cr": "BTC", "stocks": "SPY", "fx": "EURUSD"}


# ------------------------------------------------------------------ donnees
def align(src_ts, src_val, ts):
    """Derniere valeur connue de src a chaque instant ts (aucun regard vers le futur)."""
    idx = np.searchsorted(src_ts, ts, side="right") - 1
    out = np.where(idx >= 0, src_val[np.clip(idx, 0, None)], np.nan)
    return out


def macro_series(fetch_daily):
    """Regime de marche quotidien : S&P au-dessus de sa MM50, niveau du VIX (cloture de la VEILLE)."""
    out = {}
    try:
        spy = fetch_daily("SPY")
        c = spy[:, 4]
        ma = np.array([c[max(0, i - 49):i + 1].mean() for i in range(len(c))])
        out["spy_au_dessus_mm50"] = (spy[:, 0] + 86400, np.where(c > ma, 1.0, -1.0))
    except Exception as ex:
        log.warning("macro SPY : %s", ex)
    try:
        vix = fetch_daily("^VIX")
        out["vix"] = (vix[:, 0] + 86400, vix[:, 4])
    except Exception as ex:
        log.warning("macro VIX : %s", ex)
    return out


def build(kind, tf, fetch=None, fetch_daily=None, symbols=None, macro=None, idea_specs=None):
    """Indicateurs + resultats simules de tous les actifs d'une classe, empiles."""
    fetch = fetch or data.long_bars
    symbols = symbols or UNIVERSE[kind]
    raw = {}
    if not symbols:
        return None
    for s in symbols:
        try:
            a = fetch(s, kind, tf)
            if len(a) >= 400:
                raw[s] = a
        except Exception as ex:
            log.warning("%s %s : %s", s, tf, ex)
        time.sleep(CFG["PAUSE"])
    if not raw:
        return None
    bench = raw.get(BENCH[kind])
    macro = macro if macro is not None else (macro_series(fetch_daily) if fetch_daily else {})
    X, R, TS, AID = {}, {}, [], []
    for ai, (s, a) in enumerate(raw.items()):
        refs = {"bench": align(bench[:, 0], bench[:, 4], a[:, 0])} if bench is not None and s != BENCH[kind] else None
        mac = {k: align(v[0], v[1], a[:, 0]) for k, v in macro.items()}
        F, atr = IND.compute(a, kind, refs, mac)
        if idea_specs is not None:                  # l'Inventeur : ses idees deviennent des indicateurs testes
            if not idea_specs:
                idea_specs.extend(IDEAS.generate([(ag, nm) for ag, d in F.items() for nm in d],
                                                 seed=int(time.time() // (7 * 86400)), memory=IDEAS.load_memory()))
            for sp in idea_specs:
                x = IDEAS.apply_spec(F, sp)
                if x is not None:
                    F["inventeur"][IDEAS.name(sp)] = x
        days = IND.day_index(a[:, 0], kind) if kind == "stocks" else None
        for agent, feats in F.items():
            for name, x in feats.items():
                X.setdefault((agent, name), []).append(np.asarray(x, dtype=np.float32))
        for e in EXITS:
            o = sim.outcomes(a, atr, e, days, CFG["COST_BPS"][kind])
            for dr in (1, -1):
                R.setdefault((e["id"], dr), []).append(o[dr])
        TS.append(a[:, 0])
        AID.append(np.full(len(a), ai))
    n_assets = len(raw)
    X = {k: np.concatenate(v) for k, v in X.items() if len(v) == n_assets}
    R = {k: tuple(np.concatenate([t[i] for t in v]) for i in range(3)) for k, v in R.items()}
    ts, aid = np.concatenate(TS), np.concatenate(AID)
    q1, q2 = np.quantile(ts, [0.6, 0.8])
    split = np.where(ts <= q1, 0, np.where(ts <= q2, 1, 2))
    # purge + embargo (Lopez de Prado) : les trades dont le resultat deborde sur la periode suivante sont retires
    step = float(np.median(np.diff(np.unique(ts)))) if len(ts) > 1 else 900
    emb = max(e["h"] for e in EXITS) * step
    split = np.where(((ts > q1 - emb) & (ts <= q1)) | ((ts > q2 - emb) & (ts <= q2)), -1, split)
    hist = {s: [int(len(a)), round((a[-1, 0] - a[0, 0]) / 86400, 1)] for s, a in raw.items()}
    return {"kind": kind, "tf": tf, "X": X, "R": R, "ts": ts, "aid": aid, "split": split, "step": step, "hist": hist,
            "symbols": list(raw), "cut": (float(q1), float(q2)), "_base": {}}


# ------------------------------------------------------------------ conditions
def conditions(ds):
    """Chaque indicateur decoupe en 5 zones (quintiles de la periode d'apprentissage) ; indicateurs a valeurs
       discretes (divergence, motif de bougie...) : une condition par valeur."""
    tr = ds["split"] == 0
    out = []
    for (agent, name), x in ds["X"].items():
        xt = x[tr]
        xt = xt[np.isfinite(xt)]
        if len(xt) < 200:
            continue
        u = np.unique(xt)
        if len(u) <= 6:
            for val in u:
                if (xt == val).mean() < 0.95:
                    out.append((agent, name, float(val) - 1e-6, float(val) + 1e-6))
            continue
        qs = np.quantile(xt, [0.2, 0.4, 0.6, 0.8])
        edges = [-np.inf] + [float(q) for q in qs] + [np.inf]
        for i in range(5):
            out.append((agent, name, edges[i], edges[i + 1]))
    return out


def cmask(ds, cond):
    x = ds["X"][(cond[0], cond[1])]
    return (x > cond[2]) & (x <= cond[3])


# ------------------------------------------------------------------ statistiques
def stats(r, win, m):
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    rr = r[m]
    avg = float(rr.mean())
    sd = float(rr.std()) if n > 1 else 0.0
    return {"n": n, "win": float(win[m].mean()), "avg": avg, "se": sd / math.sqrt(n) if n > 1 else 1.0}


def thin(m, aid, ts, gap):
    """Trades sans chevauchement : apres un signal, on ignore l'actif pendant la duree du trade."""
    idx = np.flatnonzero(m)
    keep = np.zeros(len(m), dtype=bool)
    last = {}
    for i in idx:
        a = aid[i]
        if ts[i] >= last.get(a, -1e18):
            keep[i] = True
            last[a] = ts[i] + gap
    return keep


def wilson_low(k, n, z=1.64):
    if n <= 0:
        return 0.0
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def pval_mean_pos(st):
    """p-valeur (approximation normale) que le gain moyen soit > 0."""
    if st.get("n", 0) < 5 or st["se"] <= 0:
        return 1.0
    z = st["avg"] / st["se"]
    return 0.5 * math.erfc(z / math.sqrt(2))


def mc(rs, sims, n_next=20, seed=7):
    if len(rs) < 8:
        return None
    rng = random.Random(seed)
    tot, dd = [], []
    for _ in range(int(sims)):
        eq = peak = d = 0.0
        for _ in range(n_next):
            eq += rs[int(rng.random() * len(rs))]
            peak = max(peak, eq)
            d = max(d, peak - eq)
        tot.append(eq)
        dd.append(d)
    tot.sort()
    dd.sort()
    return {"p_gain": sum(1 for x in tot if x > 0) / len(tot), "mediane": tot[len(tot) // 2],
            "pire5": tot[int(0.05 * len(tot))], "baisse_max95": dd[int(0.95 * len(dd))]}


# ------------------------------------------------------------------ recherche
def search(ds, log_every=50):
    conds = conditions(ds)
    tr, va, te = (ds["split"] == k for k in (0, 1, 2))
    beam, depth, min_n = int(CFG["BEAM"]), int(CFG["MAX_DEPTH"]), max(60, int(0.002 * tr.sum()))
    tested = 0
    cands = []
    by_agent = {}
    for c_ in conds:
        by_agent.setdefault(c_[0], []).append(c_)
    for agent, cl in by_agent.items():
        masks = {c_: cmask(ds, c_) for c_ in cl}
        for (eid, dr), (r, win, valid) in ds["R"].items():
            base = valid & tr
            scored = []
            for c_, m in masks.items():
                mm = m & base
                st = stats(r, win, mm)
                tested += 1
                if st["n"] >= min_n and st["win"] >= CFG["MIN_WIN"] - 0.05:
                    scored.append((st["avg"] - st["se"], (c_,), mm))
            scored.sort(key=lambda x: -x[0])
            level = scored[:beam]
            singles = [x[1][0] for x in level]
            allc = list(level)
            for _d in range(2, depth + 1):
                nxt = []
                for sc, combo, mm in level:
                    for c2 in singles:
                        if c2 in combo or c2[1] in {x[1] for x in combo}:
                            continue
                        m2 = mm & masks[c2]
                        st = stats(r, win, m2)
                        tested += 1
                        if st["n"] >= min_n and st["win"] >= CFG["MIN_WIN"] - 0.03:
                            nxt.append((st["avg"] - st["se"], tuple(sorted(combo + (c2,))), m2))
                uniq = {}
                for x in sorted(nxt, key=lambda x: -x[0]):
                    uniq.setdefault(x[1], x)
                level = list(uniq.values())[:beam]
                allc += level
            for sc, combo, _mm in allc:
                if sc > 0:
                    cands.append({"agent": agent, "exit": eid, "dir": dr, "conds": combo})
    log.info("%s %s : %d conditions, %d combinaisons testees, %d candidates", ds["kind"], ds["tf"], len(conds), tested, len(cands))
    return cands, tested, len(conds)


def full_mask(ds, conds):
    m = np.ones(len(ds["ts"]), dtype=bool)
    for c_ in conds:
        m &= cmask(ds, c_)
    return m


def evaluate(ds, cand):
    ex = next(e for e in EXITS if e["id"] == cand["exit"])
    r, win, valid = ds["R"][(cand["exit"], cand["dir"])]
    m = full_mask(ds, cand["conds"]) & valid
    step = ds["step"]
    out = {}
    for k, name in ((0, "apprentissage"), (1, "validation"), (2, "test")):
        mk = thin(m & (ds["split"] == k), ds["aid"], ds["ts"], ex["h"] * step)
        out[name] = stats(r, win, mk)
        out[name]["k"] = int((win & mk).sum())
        if name == "test":
            out["test_r"] = r[mk].tolist()
        bk = (cand["exit"], cand["dir"], k)
        if bk not in ds["_base"]:                 # reference (toutes les bougies) calculee une seule fois
            bm = thin(valid & (ds["split"] == k), ds["aid"], ds["ts"], ex["h"] * step)
            ds["_base"][bk] = stats(r, win, bm)
        out["base_" + name] = ds["_base"][bk]
    return out


def robustness(ds, cand):
    """Sur validation + test (jamais appris) : gagnante sur la majorite des actifs pris un par un, et >= 75 % de
       reussite sur chacune des deux moitiés chronologiques (stabilite)."""
    ex = next(e for e in EXITS if e["id"] == cand["exit"])
    r, win, valid = ds["R"][(cand["exit"], cand["dir"])]
    step = ds["step"]
    m = thin(full_mask(ds, cand["conds"]) & valid & (ds["split"] >= 1), ds["aid"], ds["ts"], ex["h"] * step)
    pos = tot = 0
    for a in np.unique(ds["aid"][m]):
        ma = m & (ds["aid"] == a)
        if ma.sum() >= 5:
            tot += 1
            pos += int(r[ma].mean() > 0)
    idx = np.flatnonzero(m)
    halves = [idx[:len(idx) // 2], idx[len(idx) // 2:]]
    wins = [float(win[h_].mean()) if len(h_) else 0.0 for h_ in halves]
    ok = (tot == 0 or pos / tot >= 0.5) and min(wins) >= CFG["MIN_WIN"] - 0.05
    return {"ok": bool(ok), "actifs_gagnants": "%d/%d" % (pos, tot), "reussite_moities": [round(w, 3) for w in wins]}


def bh(pvals, q):
    """Benjamini-Hochberg : indices acceptes."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, ok = len(pvals), -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            ok = rank
    return set(order[:ok]) if ok > 0 else set()


def run_class(ds):
    cands, tested, ncond = search(ds)
    evals = []
    for cd in cands:
        ev = evaluate(ds, cd)
        v = ev["validation"]
        if v.get("n", 0) >= 15 and v["win"] >= CFG["MIN_WIN"] and v["avg"] > 0:
            evals.append((cd, ev))
    keep = bh([pval_mean_pos(ev["validation"]) for _, ev in evals], CFG["FDR"])
    final = []
    for i, (cd, ev) in enumerate(evals):
        if i not in keep:
            continue
        t = ev["test"]
        need = CFG["MIN_TRADES_TEST_15M"] if ds["tf"] == "15M" else CFG["MIN_TRADES_TEST"]
        if t.get("n", 0) >= need and t["win"] >= CFG["MIN_WIN"] and t["avg"] > 0:
            rob = robustness(ds, cd)
            if not rob["ok"]:
                continue
            ex = next(e for e in EXITS if e["id"] == cd["exit"])
            final.append({
                "id": "%s-%s-%s-%s-%d-%d" % (ds["kind"], ds["tf"], cd["agent"], cd["exit"], cd["dir"], len(final)),
                "kind": ds["kind"], "tf": ds["tf"], "agent": cd["agent"], "dir": cd["dir"], "exit": ex,
                "conds": [list(c_) for c_ in cd["conds"]],
                "stats": {k: ev[k] for k in ("apprentissage", "validation", "test", "base_test")},
                "wilson_test": wilson_low(t["k"], t["n"]),
                "robustesse": rob,
                "mc": mc(ev["test_r"], CFG["SIMS"]),
            })
    final.sort(key=lambda s: (-s["stats"]["test"]["win"], -s["wilson_test"], -s["stats"]["test"]["avg"]))   # reussite d'abord
    per, kept = {}, []
    for st in final:                              # au plus 5 strategies par specialiste et par sens (pas de doublons)
        key = (st["agent"], st["dir"])
        if per.get(key, 0) < 5:
            per[key] = per.get(key, 0) + 1
            kept.append(st)
    final = kept
    army = army_test(ds, final)
    days = sorted(v[1] for v in ds["hist"].values())
    return {"kind": ds["kind"], "tf": ds["tf"], "symbols": ds["symbols"], "historique_jours": days[len(days) // 2] if days else 0,
            "bougies": int(sum(v[0] for v in ds["hist"].values())), "conditions": ncond, "combinaisons": tested,
            "candidates": len(cands), "validees": len(evals), "apres_fdr": len(keep), "strategies": final, "armee": army}


def army_test(ds, final):
    """Le vote de l'armee sur la periode de test : signal quand au moins CONSENSUS specialistes differents ont une
       strategie validee active dans le meme sens au meme moment."""
    res = {}
    te = ds["split"] == 2
    step = ds["step"]
    for dr in (1, -1):
        votes = {}
        for s in final:
            if s["dir"] == dr:
                m = full_mask(ds, [tuple(c_) for c_ in s["conds"]])
                votes[s["agent"]] = votes.get(s["agent"], np.zeros(len(m), dtype=bool)) | m
        if len(votes) < CFG["CONSENSUS"]:
            continue
        cnt = sum(v.astype(int) for v in votes.values())
        sig = (cnt >= CFG["CONSENSUS"]) & te
        best = None
        for e in EXITS:
            r, win, valid = ds["R"][(e["id"], dr)]
            mk = thin(sig & valid, ds["aid"], ds["ts"], e["h"] * step)
            st = stats(r, win, mk)
            if st.get("n", 0) and (best is None or st["avg"] > best["avg"]):
                best = dict(st, exit=e["id"])
        if best:
            res["achat" if dr > 0 else "vente"] = best
    return res


def run_all(fetch=None, fetch_daily=None, kinds=None, tfs=None):
    t0 = time.time()
    out = {"genere": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "regle": {
        "taux_reussite_min": CFG["MIN_WIN"], "objectif": CFG["TARGET_WIN"], "trades_test_min": CFG["MIN_TRADES_TEST"],
        "fdr": CFG["FDR"], "consensus": CFG["CONSENSUS"]}, "classes": []}
    macro = macro_series(fetch_daily or data.daily) if fetch_daily is not False else {}
    memory = IDEAS.load_memory()
    all_specs = {}
    for kind in kinds or [k for k in ("cr", "fx", "stocks") if UNIVERSE[k]]:
        for tf in tfs or CFG["TIMEFRAMES"]:
            specs = []
            ds = build(kind, tf, fetch, macro=macro, idea_specs=specs)
            if not ds:
                continue
            all_specs.update({IDEAS.name(sp): sp for sp in specs})
            res = run_class(ds)
            memory = IDEAS.remember(res["strategies"], all_specs, memory)
            out["classes"].append(res)
            log.info("%s %s : %d strategies retenues", kind, tf, len(res["strategies"]))
            del ds
    out["duree_min"] = round((time.time() - t0) / 60, 1)
    out["total"] = {k: sum(c[k] for c in out["classes"]) for k in ("conditions", "combinaisons", "candidates")}
    out["total"]["strategies"] = sum(len(c["strategies"]) for c in out["classes"])
    used = {nm for c in out["classes"] for s in c["strategies"] for ag, nm, _a, _b in s["conds"] if ag == "inventeur"}
    out["idees"] = {nm: all_specs[nm] for nm in used if nm in all_specs}
    out["inventeur"] = {"idees_testees": len(all_specs), "idees_gagnantes": len(used)}
    with open(IDEAS.MEMORY, "w") as f:
        json.dump(memory, f, indent=1)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = run_all()
    with open("strategies.json", "w") as f:
        json.dump(res, f, indent=1)
    log.info("termine : %s", res["total"])
