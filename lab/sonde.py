"""Sonde des nouvelles sources gratuites sans inscription : repondent-elles depuis les serveurs GitHub ?
   python -m lab.sonde  -> affiche le statut de chaque source (aucune cle, aucun secret)."""
import io
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


if __name__ == "__main__":
    main()
