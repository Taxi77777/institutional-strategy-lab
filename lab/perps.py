"""Comparaison (demande du trader, 9 nov.) : les contrats perpetuels Kraken Futures avec au moins 10 M$ de volume sur
24 h, face aux cryptos scannees par l'armee (volume mesure sur Coinbase au comptant).
   python -m lab.perps cr   -> resultats/perps.json"""
import json
import os
import sys
import time

ALIAS = {"XBT": "BTC", "XDG": "DOGE"}


def run(kind="cr", tf=""):
    if kind != "cr":
        return
    from lab import data
    from lab.config import UNIVERSE
    r = data.SESSION.get("https://futures.kraken.com/derivatives/api/v3/tickers", timeout=30)
    r.raise_for_status()
    perps = {}
    for t in r.json().get("tickers", []):
        s = t.get("symbol", "")
        if not s.startswith("PF_") or not s.endswith("USD"):
            continue
        base = ALIAS.get(s[3:-3], s[3:-3])
        vq = float(t.get("volumeQuote") or 0)
        perps[base] = max(perps.get(base, 0), vq)
    big = {b: v for b, v in perps.items() if v >= 10e6}
    uni = set(UNIVERSE["cr"])
    cb = {}
    for s in UNIVERSE["cr"]:
        try:
            a = data.coinbase(s, 3600, 2)
            a = a[a[:, 0] >= time.time() - 86400]
            cb[s] = float((a[:, 5] * a[:, 4]).sum())
        except Exception as ex:
            cb[s] = None
    res = {
        "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "perps_10M": len(big),
        "scannees": sorted([b for b in big if b in uni], key=lambda b: -big[b]),
        "manquantes": sorted([[b, round(big[b] / 1e6, 1)] for b in big if b not in uni], key=lambda x: -x[1]),
        "bloquees_par_volume_coinbase": sorted([[s, round(big[s] / 1e6, 1), round((cb[s] or 0) / 1e6, 1)]
                                                for s in uni if s in big and (cb.get(s) or 0) < 10e6],
                                               key=lambda x: -x[1]),
        "scannees_sans_perp_10M": sorted(s for s in uni if s not in big),
    }
    res["volume_coinbase_M"] = {s: round((cb.get(s) or 0) / 1e6, 1) for s in sorted(uni, key=lambda s: -(cb.get(s) or 0))}
    print("COINBASE 24H M$ :", json.dumps(res["volume_coinbase_M"]))
    allp = sorted(perps.items(), key=lambda x: -x[1])
    res["tous_perps_M"] = {b: round(v / 1e6, 2) for b, v in allp}
    print("TOUS PERPS", len(allp), json.dumps([[b, round(v / 1e6, 1), int(b in uni)] for b, v in allp[:120]]))
    print("PERPS >= 10 M$ :", res["perps_10M"], "| deja scannees :", len(res["scannees"]))
    print("MANQUANTES (perp, volume M$) :", json.dumps(res["manquantes"]))
    print("BLOQUEES (perp M$, coinbase M$) :", json.dumps(res["bloquees_par_volume_coinbase"]))
    print("SCANNEES SANS PERP 10 M$ :", " ".join(res["scannees_sans_perp_10M"]))
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/perps.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    run(*sys.argv[1:])
