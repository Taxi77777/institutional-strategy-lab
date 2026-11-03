"""Sonde des nouvelles sources gratuites sans inscription : repondent-elles depuis les serveurs GitHub ?
   python -m lab.sonde  -> affiche le statut de chaque source (aucune cle, aucun secret)."""
import io
import os
import time
import zipfile

from lab.data import SESSION

NOW = int(time.time() * 1000)
SOURCES = {
    "binance_vision_klines": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip",
    "binance_api_klines": "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=5",
    "binance_vision_metrics": "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2024-01-15.zip",
    "deribit_dvol": "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=3600"
                    "&start_timestamp=%d&end_timestamp=%d" % (NOW - 5 * 86400000, NOW),
    "deribit_dvol_2021": "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=3600"
                         "&start_timestamp=1617235200000&end_timestamp=1617580800000",
    "cftc_cot_financier_2024": "https://www.cftc.gov/files/dea/history/fut_fin_txt_2024.zip",
    "coinbase_usdt_usd": "https://api.exchange.coinbase.com/products/USDT-USD/candles?granularity=3600",
    "binance_data_api_klines": "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=3",
    "binance_data_api_2023": "https://data-api.binance.vision/api/v3/klines?symbol=ETHUSDT&interval=1h&limit=2"
                             "&startTime=1672531200000",
    "binance_us_klines": "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=3",
    "coinbase_trades": "https://api.exchange.coinbase.com/products/BTC-USD/trades?limit=5",
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


if __name__ == "__main__":
    main()
    if os.environ.get("SONDE_REEL"):
        reel()
