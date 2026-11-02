"""Verification independante d'une strategie publiee sur des donnees de courtier (Dukascopy), comparee a Yahoo.
   python -m lab.validate  -> validation.json"""
import datetime
import json
import logging
import os
import numpy as np
from lab import data, indicators as IND, sim
from lab.config import UNIVERSE, CFG
from lab.miner import thin

log = logging.getLogger("lab")


def strat_best():
    """Strategie a verifier : VAL_STRAT (identifiant) dans VAL_FILE, sinon la meilleure forex 1 h."""
    with open(os.environ.get("VAL_FILE", "strategies.json")) as f:
        lab = json.load(f)
    sid = os.environ.get("VAL_STRAT", "")
    if sid:
        return next(s for c in lab["classes"] for s in c["strategies"] if s["id"] == sid)
    cl = next(c for c in lab["classes"] if c["kind"] == "fx" and c["tf"] == "1H")
    return max(cl["strategies"], key=lambda s: (s["stats"]["test"]["win"], s["wilson_test"]))


def replay(a, s, kind="fx"):
    F, atr = IND.compute(a, kind)
    m = np.ones(len(a), dtype=bool)
    for ag, nm, lo, hi in s["conds"]:
        x = F.get(ag, {}).get(nm)
        if x is None:
            return None
        m &= (x > lo) & (x <= hi)
    r, win, valid = sim.outcomes(a, atr, s["exit"], None, CFG["COST_BPS"][kind])[s["dir"]]
    keep = thin(m & valid, np.zeros(len(a), dtype=int), a[:, 0], s["exit"]["h"] * 3600)
    return a[:, 0][keep], r[keep], win[keep]


def summary(ts, r, w):
    if len(r) == 0:
        return {"n": 0}
    return {"n": int(len(r)), "win": round(float(w.mean()), 3), "avg": round(float(r.mean()), 3),
            "total_R": round(float(r.sum()), 1)}


def main():
    s = strat_best()
    if os.environ.get("VAL_DIR") in ("1", "-1"):          # meme regle, sens force (achat 1 / vente -1)
        s = dict(s, dir=int(os.environ["VAL_DIR"]), id=s["id"] + ("_achat" if os.environ["VAL_DIR"] == "1" else "_vente"))
    kind = os.environ.get("VAL_KIND", "fx")
    days = int(os.environ.get("VAL_DAYS", "730"))
    out = {"genere": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "strategie": s["id"],
           "conds": s["conds"], "exit": s["exit"], "labo_test": s["stats"]["test"], "marche": kind, "jours": days,
           "sources": {}}
    srcs = ("dukascopy", "yahoo") if kind == "fx" and not os.environ.get("VAL_STRAT") else \
        (("dukascopy",) if kind == "fx" else ("coinbase",))
    for src in srcs:
        allt, allr, allw, per = [], [], [], {}
        for sym in UNIVERSE[kind]:
            try:
                if src == "dukascopy":
                    a = data.dukascopy_hours(sym, days) if days > 800 else data.dukascopy(sym, 3600, days)
                elif src == "coinbase":
                    a = data.coinbase(sym, 3600, days)
                else:
                    a = data.bars(sym, "fx", "1H")
                res = replay(np.asarray(a, dtype=float), s, kind)
            except Exception as ex:
                log.warning("%s %s : %s", src, sym, ex)
                continue
            if res is None:
                continue
            t, r, w = res
            per[sym] = summary(t, r, w)
            allt.append(t); allr.append(r); allw.append(w)
            log.info("%s %s : %s", src, sym, per[sym])
        if not allr:
            out["sources"][src] = {"erreur": "aucune donnee"}
            continue
        t, r, w = np.concatenate(allt), np.concatenate(allr), np.concatenate(allw)
        yrs = (t // (365.25 * 86400)).astype(int)
        out["sources"][src] = {"total": summary(t, r, w),
                               "par_annee": {str(1970 + y): summary(t[yrs == y], r[yrs == y], w[yrs == y]) for y in np.unique(yrs)},
                               "par_paire": per}
    name = "validation.json" if not os.environ.get("VAL_STRAT") else "validation_%s_%s.json" % (s["id"], kind)
    with open(name, "w") as f:
        json.dump(out, f, indent=1)
    log.info("termine : %s", {k: v.get("total") for k, v in out["sources"].items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
