"""L Eclaireur : veille quotidienne sur GitHub, arXiv, Semantic Scholar, Reddit, Quant StackExchange et Hacker News
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
    "turtle": "traders", "raschke": "traders", "holy grail": "traders", "crabel": "traders", "nr7": "traders",
    "larry williams": "traders", "connors": "traders", "elder": "traders", "demark": "traders",
    "td sequential": "traders", "alligator": "traders", "bill williams": "traders", "sperandeo": "traders",
    "tudor jones": "traders", "seykota": "traders", "al brooks": "traders", "minervini": "traders",
    "druckenmiller": "traders", "market wizards": "traders",
    "stop hunt": "liquidite", "liquidity pool": "liquidite", "liquidity sweep": "liquidite", "amihud": "liquidite",
    "bid-ask spread": "liquidite", "illiquidity": "liquidite", "market liquidity": "liquidite",
    "fixing": "flux", "london fix": "flux", "gotobi": "flux", "rebalancing": "flux", "month-end": "flux",
    "nonfarm": "flux", "nfp": "flux", "fomc": "flux", "smt": "flux", "currency strength": "flux",
    "cme gap": "flux", "weekend gap": "flux", "dealer": "flux", "hedging flows": "flux",
}
GITHUB_QUERIES = ("trading strategy forex", "trading strategy crypto", "technical indicators",
                  "algorithmic trading intraday", "smart money concepts", "order flow trading",
                  "liquidity sweep trading", "stop hunt liquidity",
                  "fx fixing flows", "month end rebalancing fx", "cme gap bitcoin")


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


def semantic_scholar(n=25):
    """Publications academiques recentes (API publique Semantic Scholar, sans cle)."""
    out = []
    for q in ("intraday trading strategy", "forex technical analysis", "cryptocurrency trading signals", "ichimoku"):
        try:
            r = SESSION.get("https://api.semanticscholar.org/graph/v1/paper/search", timeout=20, params={
                "query": q, "limit": n, "fields": "title,abstract,url,year,citationCount", "sort": "publicationDate:desc"})
            for it in (r.json().get("data", []) if r.ok else []):
                txt = "%s %s" % (it.get("title", ""), it.get("abstract") or "")
                out.append({"source": "Semantic Scholar", "titre": it.get("title", ""), "url": it.get("url", ""),
                            "etoiles": it.get("citationCount") or 0, "resume": (it.get("abstract") or "")[:240],
                            "agents": _agents(txt)})
        except Exception as ex:
            log.warning("Semantic Scholar %s : %s", q, ex)
    return out


def reddit(n=25):
    """Meilleurs sujets du mois sur r/algotrading et r/Forex (flux public)."""
    out = []
    for sub in ("algotrading", "Forex", "CryptoMarkets"):
        try:
            r = SESSION.get("https://www.reddit.com/r/%s/top.json" % sub, params={"t": "month", "limit": n}, timeout=20,
                            headers={"User-Agent": "strategy-lab-scout/1.0"})
            for ch in (r.json().get("data", {}).get("children", []) if r.ok else []):
                d = ch.get("data", {})
                txt = "%s %s" % (d.get("title", ""), d.get("selftext", "")[:500])
                out.append({"source": "Reddit r/%s" % sub, "titre": d.get("title", ""),
                            "url": "https://www.reddit.com" + d.get("permalink", ""), "etoiles": d.get("score", 0),
                            "resume": d.get("selftext", "")[:240], "agents": _agents(txt)})
        except Exception as ex:
            log.warning("Reddit %s : %s", sub, ex)
    return out


def quant_stackexchange(n=30):
    """Questions les plus votees du mois sur Quant StackExchange (API publique, sans cle)."""
    try:
        r = SESSION.get("https://api.stackexchange.com/2.3/questions", timeout=20, params={
            "site": "quant", "sort": "votes", "order": "desc", "pagesize": n,
            "fromdate": int(datetime.datetime.now().timestamp()) - 30 * 86400})
        return [{"source": "Quant StackExchange", "titre": it.get("title", ""), "url": it.get("link", ""),
                 "etoiles": it.get("score", 0), "resume": " ".join(it.get("tags", [])),
                 "agents": _agents("%s %s" % (it.get("title", ""), " ".join(it.get("tags", []))))}
                for it in (r.json().get("items", []) if r.ok else [])]
    except Exception as ex:
        log.warning("Quant StackExchange : %s", ex)
        return []


def hacker_news(n=20):
    """Sujets Hacker News sur le trading (API publique Algolia)."""
    try:
        r = SESSION.get("https://hn.algolia.com/api/v1/search", timeout=20, params={
            "query": "trading strategy", "tags": "story", "hitsPerPage": n})
        return [{"source": "Hacker News", "titre": it.get("title", ""),
                 "url": it.get("url") or "https://news.ycombinator.com/item?id=%s" % it.get("objectID"),
                 "etoiles": it.get("points") or 0, "resume": "", "agents": _agents(it.get("title", ""))}
                for it in (r.json().get("hits", []) if r.ok else [])]
    except Exception as ex:
        log.warning("Hacker News : %s", ex)
        return []


def grands_traders(per=3):
    """Le specialiste « grands traders » cherche sur internet (GitHub, Hacker News, Reddit) ce qui se publie sur les
       methodes des meilleurs traders du monde (lab/traders.py). Il ne lit que titres et descriptions."""
    from lab.traders import TRADERS
    head = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        head["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    out = []
    for name in TRADERS:
        q = "%s strategy" % name
        try:
            r = SESSION.get("https://api.github.com/search/repositories", headers=head, timeout=20,
                            params={"q": q, "sort": "stars", "order": "desc", "per_page": per})
            for it in (r.json().get("items", []) if r.ok else []):
                out.append({"source": "GitHub (grands traders)", "titre": it["full_name"], "url": it["html_url"],
                            "etoiles": it.get("stargazers_count", 0), "trader": name,
                            "resume": (it.get("description") or "")[:240],
                            "agents": sorted(set(["traders"] + _agents(it.get("description"))))})
        except Exception as ex:
            log.warning("GitHub %s : %s", name, ex)
        try:
            r = SESSION.get("https://hn.algolia.com/api/v1/search", timeout=20,
                            params={"query": '"%s" trading' % name, "tags": "story", "hitsPerPage": per})
            for it in (r.json().get("hits", []) if r.ok else []):
                if name.split()[-1].lower() not in (it.get("title") or "").lower():
                    continue
                out.append({"source": "Hacker News (grands traders)", "titre": it.get("title", ""), "trader": name,
                            "url": it.get("url") or "https://news.ycombinator.com/item?id=%s" % it.get("objectID"),
                            "etoiles": it.get("points") or 0, "resume": "", "agents": ["traders"]})
        except Exception as ex:
            log.warning("Hacker News %s : %s", name, ex)
        try:
            r = SESSION.get("https://www.reddit.com/search.json", timeout=20, headers={"User-Agent": "strategy-lab-scout/1.0"},
                            params={"q": q, "sort": "top", "t": "year", "limit": per})
            for ch in (r.json().get("data", {}).get("children", []) if r.ok else []):
                d = ch.get("data", {})
                out.append({"source": "Reddit (grands traders)", "titre": d.get("title", ""), "trader": name,
                            "url": "https://www.reddit.com" + d.get("permalink", ""), "etoiles": d.get("score", 0),
                            "resume": d.get("selftext", "")[:240], "agents": ["traders"]})
        except Exception as ex:
            log.warning("Reddit %s : %s", name, ex)
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
    ss, rd, qs, hn = semantic_scholar(), reddit(), quant_stackexchange(), hacker_news()
    gt = grands_traders()
    items = gh + ax + ss + rd + qs + hn + gt
    out = {"date": datetime.date.today().isoformat(), "poids": weights(items),
           "trouvailles": sorted([it for it in items if it["agents"]], key=lambda x: -(x["etoiles"] or 0))[:60], "total": {"github": len(gh), "arxiv": len(ax), "semantic_scholar": len(ss), "reddit": len(rd),
                     "quant_stackexchange": len(qs), "hacker_news": len(hn), "grands_traders": len(gt)},
           "grands_traders": sorted(gt, key=lambda x: -(x["etoiles"] or 0))[:40]}
    with open("veille.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log.info("veille : %d GitHub, %d arXiv, poids %s", len(gh), len(ax), out["poids"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
