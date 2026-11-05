"""Sonde des nouvelles sources gratuites sans inscription : repondent-elles depuis les serveurs GitHub ?
   python -m lab.sonde  -> affiche le statut de chaque source (aucune cle, aucun secret)."""
import io
import os
import time
import zipfile

from lab.data import SESSION

NOW = int(time.time() * 1000)
SOURCES = {
    "binance_EURUSDT_debut": "https://data-api.binance.vision/api/v3/klines?symbol=EURUSDT&interval=1d&limit=1&startTime=0",
    "binance_GBPUSDT_debut": "https://data-api.binance.vision/api/v3/klines?symbol=GBPUSDT&interval=1d&limit=1&startTime=0",
    "binance_AUDUSDT_debut": "https://data-api.binance.vision/api/v3/klines?symbol=AUDUSDT&interval=1d&limit=1&startTime=0",
    "binance_PAXGUSDT_debut": "https://data-api.binance.vision/api/v3/klines?symbol=PAXGUSDT&interval=1d&limit=1&startTime=0",
    "binance_GBPUSDT_jour": "https://data-api.binance.vision/api/v3/klines?symbol=GBPUSDT&interval=1d&limit=1",
    "binance_AUDUSDT_jour": "https://data-api.binance.vision/api/v3/klines?symbol=AUDUSDT&interval=1d&limit=1",
    "binance_EURUSDT_jour": "https://data-api.binance.vision/api/v3/klines?symbol=EURUSDT&interval=1d&limit=1",
    "binance_JPY_CHF_CAD_NZD": "https://data-api.binance.vision/api/v3/exchangeInfo?symbols=%5B%22USDTJPY%22%5D",
    "kraken_paires_fx": "https://api.kraken.com/0/public/AssetPairs?pair=USDJPY,USDCAD,USDCHF,AUDUSD,GBPUSD,EURUSD,EURGBP,EURJPY,EURCHF,EURCAD,AUDJPY,NZDUSD",
    "kraken_trades_USDJPY": "https://api.kraken.com/0/public/Trades?pair=USDJPY&count=5",
    "kraken_volume_24h": "https://api.kraken.com/0/public/Ticker?pair=USDJPY,USDCAD,USDCHF,AUDUSD,GBPUSD,EURUSD,EURGBP",
    "coinbase_fiat": "https://api.exchange.coinbase.com/products/EURC-USD/candles?granularity=86400",
    "binance_EURUSDT_aggtrades_mois": "https://data.binance.vision/data/spot/monthly/aggTrades/EURUSDT/EURUSDT-aggTrades-2025-06.zip",
    "binance_GBPUSDT_aggtrades_mois": "https://data.binance.vision/data/spot/monthly/aggTrades/GBPUSDT/GBPUSDT-aggTrades-2025-06.zip",
    "binance_XAUUSDT_fut_klines": "https://data.binance.vision/data/futures/um/monthly/klines/XAUUSDT/1h/XAUUSDT-1h-2026-01.zip",
    "binance_XAUUSDT_fut_funding": "https://data.binance.vision/data/futures/um/monthly/fundingRate/XAUUSDT/XAUUSDT-fundingRate-2026-01.zip",
    "binance_XAUUSDT_fut_metrics": "https://data.binance.vision/data/futures/um/daily/metrics/XAUUSDT/XAUUSDT-metrics-2026-01-15.zip",
    "binance_EURUSDT_fut_klines": "https://data.binance.vision/data/futures/um/monthly/klines/EURUSDT/1h/EURUSDT-1h-2026-01.zip",
}


def main():
    for nm, url in SOURCES.items():
        try:
            r = SESSION.get(url, timeout=30, headers={"User-Agent": "strategy-lab/1.0"})
            info = "HTTP %d, %d octets" % (r.status_code, len(r.content))
            if r.ok and url.endswith(".zip"):
                z = zipfile.ZipFile(io.BytesIO(r.content))
                first = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
                info += ", %d lignes, debut : %s" % (len(first), first[0][:160])
            elif r.ok:
                info += ", debut : %s" % r.text[:200].replace("\n", " ")
        except Exception as ex:
            info = "ERREUR %s" % str(ex)[:200]
        print("%-26s %s" % (nm, info), flush=True)


def reel():
    """Telechargement reel du COT et du DVOL tel que le labo l'utilise."""
    import numpy as np
    from lab import data, positionnement as PO
    t = time.time()
    cot = data.cot()
    print("COT : %s (%.0f s)" % ({k: len(v) for k, v in cot.items()}, time.time() - t), flush=True)
    for k, v in cot.items():
        print("   %s derniere dispo %s fonds %.3f gerants %.3f" % (k, time.strftime("%Y-%m-%d", time.gmtime(v[-1, 0])),
                                                                v[-1, 1], v[-1, 2]))
    t = time.time()
    dv = {c: data.dvol(c) for c in ("BTC", "ETH")}
    print("DVOL : %s (%.0f s)" % ({k: (len(v), time.strftime("%Y-%m-%d", time.gmtime(v[0, 0])) if len(v) else "-")
                                   for k, v in dv.items()}, time.time() - t), flush=True)
    ts = np.arange(time.time() // 3600 * 3600 - 2000 * 3600, time.time() // 3600 * 3600, 3600.0)
    c = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-3, len(ts))))
    for sym, kind in (("EURUSD", "fx"), ("USDJPY", "fx"), ("ETH", "cr")):
        F = PO.compute_positionnement(sym, kind, ts, c, {"cot": cot, "dvol": dv})
        print("   %s : %s" % (sym, {k: round(float(x[-1]), 3) for k, x in F.items()}))


def flux():
    """Order flow reel : Binance (crypto) et volume des ticks Dukascopy (forex)."""
    import numpy as np
    from lab import data, indicators as IND
    from lab.miner import orderflow_refs
    t = time.time()
    bf = data.binance_flow("BTC", days=60)
    print("Binance BTC : %d bougies, acheteurs %.0f %% du volume (%.0f s)" % (len(bf), 100 * np.nanmean(bf[:, 2] / bf[:, 1]),
                                                                            time.time() - t), flush=True)
    a = data.coinbase("BTC", 3600, 60)
    F, atr = IND.compute(a, "cr", {"sym": "BTC", "orderflow": orderflow_refs("BTC", "cr", a[:, 0], flow=bf)})
    print("   crypto :", {k: round(float(x[-1]), 3) for k, x in F["orderflow"].items()}, flush=True)
    a = data.dukascopy_live("EURUSD", 3600, 40)
    F, atr = IND.compute(a, "fx", {"sym": "EURUSD", "orderflow": orderflow_refs("EURUSD", "fx", a[:, 0])})
    v = data.VOLUME.get("EURUSD")
    print("   forex : volume ticks %d bougies dont %d connues ; %s" % (len(v[1]), int(np.isfinite(v[1]).sum()),
          {k: round(float(x[-2]), 3) for k, x in F["orderflow"].items()}), flush=True)


def signaux():
    """Etat en direct : pour chaque crypto, les pistes actives remplissent-elles leurs conditions maintenant ?"""
    import numpy as np
    from lab import live
    lab = live.load_strategies()
    now = time.time()
    acts = [a for a in lab.get("activees", []) if a["kind"] == "cr"]
    print("pistes crypto actives :", len(acts), flush=True)
    for sym in live.UNIVERSE["cr"]:
        try:
            fx = live.features(sym, "cr", "1H", now, lab.get("idees"))
        except Exception as ex:
            print("  %s : erreur %s" % (sym, ex), flush=True)
            continue
        if not fx:
            continue
        for a in acts:
            if sym not in a["paires_ok"]:
                continue
            res = []
            for ag, nm, lo, hi in a["conds"]:
                v = fx["last"].get((ag, nm))
                res.append("%s=%s %s" % (nm[:28], "nan" if v is None or not np.isfinite(v) else round(v, 3),
                                         "OK" if v is not None and np.isfinite(v) and lo <= v <= hi else "non"))
            ok = live.active(a, fx["last"])
            heat = live.heat_check(sym, "cr", a["dir"], fx) if ok else (True, None)
            print("  %s %s | %s | %s%s" % ("SIGNAL" if ok and heat[0] else ("bloque heatmap" if ok else "-"), sym,
                                          a["activee"][-28:], " ; ".join(res), (" | " + heat[1]) if ok and heat[1] else ""),
                  flush=True)
        print("  %s prix %s, bougie %s" % (sym, fx["close"], time.strftime("%H:%M", time.gmtime(fx["ts"]))), flush=True)


def rejeu(heures=120):
    """Rejoue les 5 derniers jours avec les memes calculs et les memes regles que l'armee (pistes activees, actifs
       autorises, heures de volume 9 h - 18 h 30) : combien de signaux seraient partis, et quand ?"""
    import concurrent.futures as cf
    from lab import live
    live.KEEP_FULL[0] = True
    lab = live.load_strategies()
    acts = lab.get("activees", [])
    print("pistes actives :", len(acts), flush=True)
    now = time.time()
    jobs = sorted({(a["kind"], a["tf"], sym) for a in acts for sym in a["paires_ok"]})

    def one(j):
        kind, tf, sym = j
        try:
            fx = live.features(sym, kind, tf, now, lab.get("idees"))
        except Exception as ex:
            return [("erreur", "%s %s : %s" % (sym, tf, str(ex)[:80]))]
        if not fx or "F" not in fx:
            return [("erreur", "%s %s : pas de donnees" % (sym, tf))]
        F, a = fx["F"], fx["a"]
        out, n = [], len(a)
        mine = [s for s in acts if s["kind"] == kind and s["tf"] == tf and sym in s["paires_ok"]]
        for i in range(n):
            t_close = a[i, 0] + live.STEP[tf]
            if t_close < now - heures * 3600 or not live.in_hours(t_close):
                continue
            last = {(ag, nm): float(x[i]) for ag, d in F.items() for nm, x in d.items()}
            for s in mine:
                if live.active(s, last):
                    out.append(("signal", "%s %s %s %s | %s" % (time.strftime("%d/%m %H:%M", time.gmtime(t_close + 3600)),
                                                              sym, tf, "ACHAT" if s["dir"] > 0 else "VENTE", s["activee"][:60])))
                    break
        return out
    res = []
    with cf.ThreadPoolExecutor(4) as ex:
        for r in ex.map(one, jobs):
            res += r
    sig = sorted(x for k, x in res if k == "signal")
    for x in sig:
        print("  AURAIT SIGNALE", x, flush=True)
    for k, x in res:
        if k == "erreur":
            print("  erreur", x, flush=True)
    print("total signaux rejoues sur %d h (heures de volume) : %d" % (heures, len(sig)), flush=True)

if __name__ == "__main__":
    if os.environ.get("SONDE_REJEU", "1") == "1":
        rejeu()
    if os.environ.get("SONDE_SIGNAUX", "0") == "1":
        signaux()
    main()
    if os.environ.get("SONDE_REEL"):
        reel()
    if os.environ.get("SONDE_FLUX"):
        flux()

