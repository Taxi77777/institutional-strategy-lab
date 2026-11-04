"""Order flow reel des devises principales cotees sur Kraken (API publique, sans inscription) : chaque transaction
porte son sens (b = achat agressif, s = vente agressive). Kraken ne donne pas de bougies historiques au-dela de
720 heures : on reconstruit tout l'historique a partir des transactions, une par une, puis on le complete.
   python -m lab.kraken_fx  -> met a jour donnees/kraken/<PAIRE>.csv (heure, volume, volume acheteur, transactions)
Le fichier d'etat garde le curseur de chaque paire : chaque passage reprend la ou le precedent s'est arrete."""
import csv
import json
import logging
import os
import time

import numpy as np

from lab.data import SESSION

log = logging.getLogger("lab")
DIR = os.environ.get("KRAKEN_DIR", "donnees/kraken")
RAW = "https://raw.githubusercontent.com/Taxi77777/institutional-strategy-lab/main/donnees/kraken/%s.csv"
# paires Kraken -> (devise, signe) : un achat de GBPUSD = flux acheteur sur la livre ; un achat de USDJPY = flux
# vendeur sur le yen
PAIRS = {"EURUSD": ("EUR", 1), "GBPUSD": ("GBP", 1), "AUDUSD": ("AUD", 1), "USDJPY": ("JPY", -1),
         "USDCAD": ("CAD", -1), "USDCHF": ("CHF", -1), "NZDUSD": ("NZD", 1)}
START = 1514764800                      # 1er janvier 2018


def _load(pair):
    path = os.path.join(DIR, "%s.csv" % pair)
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.reader(f):
                if r and r[0].isdigit():
                    rows[int(r[0])] = [float(r[1]), float(r[2]), int(r[3])]
    return rows


def _save(pair, rows):
    os.makedirs(DIR, exist_ok=True)
    with open(os.path.join(DIR, "%s.csv" % pair), "w") as f:
        f.write("heure,volume,volume_acheteur,transactions\n")
        for t in sorted(rows):
            v, b, n = rows[t]
            f.write("%d,%.8g,%.8g,%d\n" % (t, v, b, n))


def _trades(pair, since):
    r = SESSION.get("https://api.kraken.com/0/public/Trades", params={"pair": pair, "since": since, "count": 1000},
                    timeout=30)
    j = r.json()
    if j.get("error"):
        raise RuntimeError(", ".join(j["error"]))
    res = j["result"]
    last = res.pop("last")
    data = next(iter(res.values())) if res else []
    return data, last


def aggregate(trades, rows):
    for t in trades:
        vol, ts, side = float(t[1]), float(t[2]), t[3]
        h = int(ts // 3600 * 3600)
        cur = rows.setdefault(h, [0.0, 0.0, 0])
        cur[0] += vol
        cur[1] += vol if side == "b" else 0.0
        cur[2] += 1
    return rows


def update(budget_s=19000, pause=1.2):
    """Complete l'historique de chaque paire (reprise au curseur), dans la limite de budget_s secondes."""
    os.makedirs(DIR, exist_ok=True)
    t_end = time.time() + budget_s
    only = [p for p in os.environ.get("KRAKEN_PAIRS", "").split(",") if p]
    old_state = os.path.join(DIR, "etat.json")
    shared = json.load(open(old_state)) if os.path.exists(old_state) else {}
    for pair in (only or list(PAIRS)):
        state_path = os.path.join(DIR, "etat_%s.json" % pair)       # un etat par paire : collecte en parallele
        state = json.load(open(state_path)) if os.path.exists(state_path) else {pair: shared.get(pair)} if shared.get(pair) else {}
        rows = _load(pair)
        since = state.get(pair, str(START * 10 ** 9))
        calls = 0
        while time.time() < t_end:
            try:
                data, last = _trades(pair, since)
            except Exception as ex:
                log.warning("Kraken %s : %s", pair, ex)
                if "Unknown asset pair" in str(ex):
                    break
                time.sleep(10)
                calls += 1
                if calls > 20:
                    break
                continue
            calls += 1
            if not data:
                break
            # la derniere heure, encore incomplete, sera recomptee au prochain passage
            hour_now = int(time.time() // 3600 * 3600)
            done = [t for t in data if float(t[2]) < hour_now]
            aggregate(done, rows)
            if len(done) < len(data):
                since = str(int(hour_now * 10 ** 9))
                break
            since = last
            if calls % 50 == 0:
                _save(pair, rows)
                state[pair] = since
                json.dump(state, open(state_path, "w"), indent=1)
                log.info("Kraken %s : %d heures, curseur %s", pair, len(rows), since)
            time.sleep(pause)
        _save(pair, rows)
        state[pair] = since
        json.dump(state, open(state_path, "w"), indent=1)
        log.info("Kraken %s : %d heures enregistrees", pair, len(rows))


def history(pair):
    """(n, 4) : heure, volume, volume acheteur, transactions. Fichier local (recherche) sinon version publiee (direct)."""
    rows = _load(pair)
    if not rows:
        try:
            r = SESSION.get(RAW % pair, params={"t": int(time.time() // 3600)}, timeout=30)
            if r.ok:
                for line in r.text.splitlines()[1:]:
                    p = line.split(",")
                    rows[int(p[0])] = [float(p[1]), float(p[2]), int(p[3])]
        except Exception as ex:
            log.warning("Kraken %s (publie) : %s", pair, ex)
    return np.array([(t, v[0], v[1], v[2]) for t, v in sorted(rows.items())], dtype=float).reshape(-1, 4)


LIVE = {}


def live_flow(pair, hours=72):
    """Direct : historique publie + transactions des dernieres heures (rafraichi toutes les 10 min)."""
    c = LIVE.get(pair)
    if c and time.time() - c[0] < 600:
        return c[1]
    hist = {int(r[0]): [r[1], r[2], r[3]] for r in history(pair)}
    since = str(int((time.time() - hours * 3600) * 10 ** 9))
    recent = {}
    for _ in range(20):
        try:
            data, last = _trades(pair, since)
        except Exception:
            break
        if not data:
            break
        aggregate(data, recent)
        if len(data) < 1000:
            break
        since = last
        time.sleep(1.0)
    hour_now = int(time.time() // 3600 * 3600)
    for h, v in recent.items():
        if h < hour_now:                            # heures terminees seulement
            hist[h] = v
    out = np.array([(t, v[0], v[1], v[2]) for t, v in sorted(hist.items())], dtype=float).reshape(-1, 4)
    LIVE[pair] = (time.time(), out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    update(budget_s=float(os.environ.get("KRAKEN_BUDGET_S", 19000)))
