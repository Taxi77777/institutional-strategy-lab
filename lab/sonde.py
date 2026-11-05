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


def rejeu(heures=48):
    """Rejoue les dernieres heures avec les memes calculs que l'armee : ou les pistes actives auraient-elles signale ?"""
    from lab import live
    live.KEEP_FULL[0] = True
    lab = live.load_strategies()
    acts = lab.get("activees", [])
    print("pistes actives :", len(acts), flush=True)
    now = time.time()
    tot = 0
    for kind in ("fx", "cr"):
        for sym in live.UNIVERSE[kind]:
            ok_acts = [a for a in acts if a["kind"] == kind and sym in a["paires_ok"]]
            if not ok_acts:
                continue
            try:
                fx = live.features(sym, kind, "1H", now, lab.get("idees"))
            except Exception as ex:
                print("  %s : erreur %s" % (sym, str(ex)[:80]), flush=True)
                continue
            if not fx or "F" not in fx:
                continue
            F, a = fx["F"], fx["a"]
            n = len(a)
            for i in range(max(0, n - heures), n):
                last = {(ag, nm): float(x[i]) for ag, d in F.items() for nm, x in d.items()}
                for act in ok_acts:
                    if live.active(act, last):
                        tot += 1
                        print("  AURAIT SIGNALE %s %s cloture %s | %s" % (kind, sym, time.strftime("%d/%m %H:%M",
                              time.gmtime(a[i, 0] + 3600)), act["activee"][-42:]), flush=True)
                        break
    print("total signaux rejoues sur %d h : %d" % (heures, tot), flush=True)


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

