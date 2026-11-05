"""Kronos (Tsinghua, open source MIT, https://github.com/shiyu-coder/Kronos) : modele de fondation chinois entraine sur les
bougies de 45 places de marche. Il devient un specialiste de plus : a chaque bougie fermee, il prevoit les 5 bougies
suivantes a partir des 256 precedentes (jamais de donnees futures). Sa prevision, exprimee en ATR, est une condition comme
les autres : elle doit prouver ses resultats sur des trades jamais vus avant de compter dans un signal.

   python -m lab.kronos_ia cr 1D 0/4   -> donnees/kronos/cr_1D_<ACTIF>.csv (heure, prevision) complete, part 0 sur 4
Les previsions deja calculees sont gardees : chaque passage ne calcule que les nouvelles bougies."""
import csv
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
DIR = os.environ.get("KRONOS_DATA", "donnees/kronos")
RAW = "https://raw.githubusercontent.com/Taxi77777/institutional-strategy-lab/main/donnees/kronos/%s_%s_%s.csv"
LOOK, AHEAD, BATCH = 256, 5, 32
TOKENIZER, MODEL = "NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini"


def path(kind, tf, sym):
    return os.path.join(DIR, "%s_%s_%s.csv" % (kind, tf, sym))


def load(kind, tf, sym):
    """(n, 2) : heure d'ouverture de la bougie, prevision Kronos (en ATR) connue a sa cloture. Local sinon publie."""
    rows = {}
    p = path(kind, tf, sym)
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.reader(f):
                if r and r[0].isdigit():
                    rows[int(r[0])] = float(r[1])
    else:
        try:
            from lab.data import SESSION
            r = SESSION.get(RAW % (kind, tf, sym), params={"t": int(time.time() // 3600)}, timeout=20)
            if r.ok:
                for line in r.text.splitlines()[1:]:
                    a, b = line.split(",")
                    rows[int(a)] = float(b)
        except Exception as ex:
            log.warning("Kronos %s publie : %s", sym, ex)
    return np.array(sorted(rows.items()), dtype=float).reshape(-1, 2)


def _save(kind, tf, sym, rows):
    os.makedirs(DIR, exist_ok=True)
    with open(path(kind, tf, sym), "w") as f:
        f.write("heure,prevision\n")
        for t in sorted(rows):
            f.write("%d,%.4f\n" % (t, rows[t]))


def _atr(a, n=14):
    h, l, c = a[:, 2], a[:, 3], a[:, 4]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    out = np.full(len(a), np.nan)
    for i in range(n, len(a)):
        out[i] = tr[i - n + 1:i + 1].mean()
    return out


def predictor():
    sys.path.insert(0, os.environ.get("KRONOS_DIR", "kronos_src"))
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    torch.set_num_threads(int(os.environ.get("KRONOS_THREADS", "4")))
    torch.manual_seed(0)
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    mod = Kronos.from_pretrained(MODEL)
    tok.eval()
    mod.eval()
    return KronosPredictor(mod, tok, device="cpu", max_context=LOOK)


def run(kind, tf, part="0/1", budget_s=17000):
    import pandas as pd
    from lab import data
    from lab.config import UNIVERSE
    k, n = (int(x) for x in part.split("/"))
    syms = [s for i, s in enumerate(UNIVERSE[kind]) if i % n == k]
    pr = predictor()
    t_end = time.time() + budget_s
    step = data.TF[tf][2]
    for sym in syms:
        if time.time() > t_end:
            break
        try:
            a = data.long_bars(sym, kind, tf)
        except Exception as ex:
            log.warning("Kronos %s : %s", sym, ex)
            continue
        if kind == "fx" and step >= 86400:
            a = a[((a[:, 0] // 86400 + 3) % 7) != 6]          # bougie du dimanche soir (2 h) : ignoree
        a = a[a[:, 0] + step <= time.time()]               # bougies fermees seulement
        if len(a) < LOOK + 10:
            continue
        old = {int(t): v for t, v in load(kind, tf, sym)}       # deja calculees (local ou publiees) : gardees
        atr = _atr(a)
        todo = [i for i in range(LOOK - 1, len(a)) if int(a[i, 0]) not in old and np.isfinite(atr[i]) and atr[i] > 0]
        log.info("Kronos %s %s %s : %d bougies a prevoir", kind, tf, sym, len(todo))
        ts = pd.to_datetime(a[:, 0], unit="s")
        for b0 in range(0, len(todo), BATCH):
            if time.time() > t_end:
                break
            idx = todo[b0:b0 + BATCH]
            dfs, xts, yts = [], [], []
            for i in idx:
                w = a[i - LOOK + 1:i + 1]
                dfs.append(pd.DataFrame({"open": w[:, 1], "high": w[:, 2], "low": w[:, 3], "close": w[:, 4],
                                         "volume": w[:, 5]}))
                xts.append(pd.Series(ts[i - LOOK + 1:i + 1]))
                yts.append(pd.Series(pd.to_datetime(a[i, 0] + step * np.arange(1, AHEAD + 1), unit="s")))
            try:
                preds = pr.predict_batch(dfs, xts, yts, pred_len=AHEAD, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            except Exception as ex:
                log.warning("Kronos %s : %s", sym, ex)
                break
            for i, p_ in zip(idx, preds):
                old[int(a[i, 0])] = float((p_["close"].values[-1] - a[i, 4]) / atr[i])
        _save(kind, tf, sym, old)


def avis():
    """Avis de Kronos maintenant : pour chaque actif, les 5 prochaines bougies jour prevues (moyenne de 5 tirages).
       Ce n'est PAS un signal : Kronos n'a pas encore prouve ses resultats dans le labo."""
    import pandas as pd
    from lab import data
    from lab.config import UNIVERSE
    pr = predictor()
    rows = []
    for kind in ("cr", "fx"):
        for sym in UNIVERSE[kind]:
            try:
                if kind == "fx":
                    try:
                        h = data.dukascopy_hours(sym, 420)
                        a = data._resample(h, 86400)
                    except Exception:                  # fichiers mensuels refuses : fichiers journaliers
                        a = data.dukascopy(sym, 86400, 400)
                    a = a[((a[:, 0] // 86400 + 3) % 7) != 6]
                else:
                    a = data.coinbase(sym, 86400, 320)
                a = a[a[:, 0] + 86400 <= time.time()]
                if len(a) < LOOK:
                    continue
                w = a[-LOOK:]
                df = pd.DataFrame({"open": w[:, 1], "high": w[:, 2], "low": w[:, 3], "close": w[:, 4], "volume": w[:, 5]})
                xt = pd.Series(pd.to_datetime(w[:, 0], unit="s"))
                yt = pd.Series(pd.to_datetime(w[-1, 0] + 86400 * np.arange(1, AHEAD + 1), unit="s"))
                p_ = pr.predict(df, xt, yt, pred_len=AHEAD, T=1.0, top_p=0.9, sample_count=5, verbose=False)
                atr = _atr(a)[-1]
                last = a[-1, 4]
                fin = float(p_["close"].values[-1])
                rows.append((kind, sym, (fin - last) / atr, 100 * (fin / last - 1), last, fin))
            except Exception as ex:
                log.warning("avis %s : %s", sym, ex)
    rows.sort(key=lambda r: -abs(r[2]))
    import json
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, "avis.json"), "w") as f:       # lu par l'armee : sens de Kronos pour chaque actif
        json.dump({"genere": int(time.time()), "actifs": {sym: {"marche": kind, "atr": round(z, 3), "pct": round(pct, 3)}
                                                          for kind, sym, z, pct, last, fin in rows}}, f, indent=1)
    print("AVIS KRONOS (5 prochaines bougies jour, depuis la cloture du %s UTC)" % time.strftime("%d/%m", time.gmtime()))
    for kind, sym, z, pct, last, fin in rows:
        print("AVIS %s %-7s %s %+.2f ATR (%+.2f %%) cloture %.6g -> prevue %.6g" % (
            kind, sym, "HAUSSE" if z > 0 else "BAISSE", z, pct, last, fin))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "avis":
    logging.basicConfig(level=logging.WARNING)
    avis()
elif __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(sys.argv[1], sys.argv[2].upper(), sys.argv[3] if len(sys.argv) > 3 else "0/1",
        float(os.environ.get("KRONOS_BUDGET_S", 17000)))
