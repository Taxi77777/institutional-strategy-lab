"""L'Eclaireur (28e specialiste) : veille quotidienne sur GitHub (projets de trading nouveaux et populaires) et arXiv
(dernieres publications de finance quantitative). Il ne telecharge ni n'execute jamais de code : il lit titres et
descriptions, en tire les familles d'indicateurs a la mode pour orienter l'Inventeur, et publie un resume.
   python -m lab.scout  -> veille.json"""
import datetime
import json
import logging
import os
import re
import xml.etree.ElementTree as ET

from lab.data import SESSION

log = logging.getLogger("lab")

# mots-cles -> specialiste du labo (familles d'indicateurs a explorer davantage)
KEYWORDS = {
    "momentum": "momentum", "trend": "tendance", "moving average": "tendance", "breakout": "bandes",
    "volatility": "volatilite", "garch": "volatilite", "atr": "volatilite", "bollinger": "bandes",
    "keltner": "bandes", "donchian": "bandes", "volume": "volume", "order flow": "volume", "obv": "volume",
    "vwap": "vwap", "volume profile": "profil", "market profile": "profil", "fibonacci": "fibonacci",
    "ichimoku": "ichimoku", "rsi": "oscillateurs", "stochastic": "oscillateurs", "oscillator": "oscillateurs",
    "mean reversion": "retour", "mean-reversion": "retour", "pairs trading": "retour", "cointegration": "retour",
    "statistical arbitrage": "statisticien", "entropy": "mathematicien", "hurst": "mathematicien",
    "fractal": "mathematicien", "wavelet": "mathematicien", "kalman": "tradingview", "candlestick": "chandeliers",
    "smart money": "smc", "order block": "smc", "fair value gap": "smc", "liquidity": "smc", "ict": "smc",
    "market structure": "structure", "support": "structure", "resistance": "structure",
    "session": "amd", "london": "amd", "intraday seasonality": "temps", "time-of-day": "temps",
    "regime": "macro", "macro": "macro", "funding rate": "derives", "perpetual": "derives",
    "open interest": "derives", "factor": "alpha158", "alpha": "wq101", "cross-asset": "intermarche",
    "lead-lag": "intermarche", "multi-timeframe": "multi_unites", "higher timeframe": "multi_unites",
}
GITHUB_QUERIES = ("trading strategy forex", "trading strategy crypto", "technical indicators",
                  "algorithmic trading intraday", "smart money concepts", "order flow trading")


def _agents(text):
    t = (text or "").lower()
    return sorted({ag for kw, ag in KEYWORDS.items() if re.search(r"\b%s\b" % re.escape(kw), t)})


def github(days=45, per=10):
    """Projets recents les plus etoiles (API de recherche GitHub ; jeton du workflow si present)."""
    since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    head = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        head["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    out = {}
    for q in GITHUB_QUERIES:
        try:
            r = SESSION.get("https://api.github.com/search/repositories", headers=head, timeout=20,
                            params={"q": "%s pushed:>%s stars:>20" % (q, since), "sort": "stars", "order": "desc",
                                    "per_page": per})
            for it in (r.json().get("items", []) if r.ok else []):
                out[it["full_name"]] = {"source": "GitHub", "titre": it["full_name"], "url": it["html_url"],
                                        "etoiles": it.get("stargazers_count", 0),
                                        "resume": (it.get("description") or "")[:240],
                                        "agents": _agents("%s %s %s" % (it["full_name"], it.get("description"),
                                                                         " ".join(it.get("topics") or [])))}
        except Exception as ex:
            log.warning("GitHub %s : %s", q, ex)
    return sorted(out.values(), key=lambda x: -x["etoiles"])


def arxiv(n=40):
    """Dernieres publications arXiv en finance quantitative (trading, finance statistique)."""
    try:
        r = SESSION.get("https://export.arxiv.org/api/query", timeout=30, params={
            "search_query": "cat:q-fin.TR OR cat:q-fin.ST OR cat:q-fin.CP", "sortBy": "submittedDate",
            "sortOrder": "descending", "max_results": n})
        root = ET.fromstring(r.content)
    except Exception as ex:
        log.warning("arXiv : %s", ex)
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        summ = " ".join((e.findtext("a:summary", "", ns) or "").split())
        out.append({"source": "arXiv", "titre": title, "url": e.findtext("a:id", "", ns), "etoiles": 0,
                    "resume": summ[:240], "agents": _agents(title + " " + summ),
                    "date": (e.findtext("a:published", "", ns) or "")[:10]})
    return out


def weights(items):
    """Poids par specialiste : 1 + 3 x part des trouvailles qui en parlent (borne 1..3)."""
    cnt = {}
    for it in items:
        for ag in it["agents"]:
            cnt[ag] = cnt.get(ag, 0) + 1
    tot = max(1, len(items))
    return {ag: round(min(3.0, 1.0 + 3.0 * c / tot), 2) for ag, c in sorted(cnt.items(), key=lambda x: -x[1])}


def main():
    gh, ax = github(), arxiv()
    items = gh + ax
    out = {"date": datetime.date.today().isoformat(), "poids": weights(items),
           "trouvailles": [it for it in items if it["agents"]][:40], "total": {"github": len(gh), "arxiv": len(ax)}}
    with open("veille.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log.info("veille : %d GitHub, %d arXiv, poids %s", len(gh), len(ax), out["poids"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
