import json
"""Tests hors ligne du labo : le hasard ne doit rien donner, un vrai avantage doit etre trouve, l'armee doit voter."""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("LAB_MODE", "win")                    # anciens tests ecrits pour le mode 92 %

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lab import config, indicators as IND, live, miner, telegram as TG  # noqa: E402


def synth(seed, n=2600, edge=False):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 3600 + 1790000000
    r = rng.normal(0, 0.003, n)
    if edge:
        for i in range(3, n):
            if r[i - 1] < 0 and r[i - 2] < 0 and r[i - 3] < 0:
                r[i] = abs(r[i]) + 0.004
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    return np.c_[t, o, np.maximum(o, c) * 1.0015, np.minimum(o, c) * 0.9985, c, rng.uniform(1, 10, n)]


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    monkeypatch.setenv("POSITIONNEMENT", "0")                      # pas de COT / DVOL reseau pendant les tests
    monkeypatch.setenv("ORDERFLOW", "0")                           # pas de Binance reseau pendant les tests
    monkeypatch.setitem(config.CFG, "SEULEMENT_ACTIVEES", 0)       # anciens tests : messages de l'armee actifs
    monkeypatch.setitem(config.CFG, "MODE", "win")
    monkeypatch.setitem(config.CFG, "PAUSE", 0)
    monkeypatch.setitem(config.CFG, "SIMS", 500)
    monkeypatch.setitem(config.CFG, "BEAM", 15)
    monkeypatch.setitem(config.CFG, "MIN_TRADES_TEST", 30)         # petites donnees de test
    monkeypatch.setitem(config.CFG, "MIN_WIN", 0.80)               # avantage synthetique modeste (le reel exige 92 %)
    monkeypatch.setitem(config.CFG, "STATE_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def run(edge):
    ds = miner.build("cr", "1H", lambda s, k, tf: synth(sum(map(ord, s)), edge=edge), symbols=["BTC", "ETH", "SOL"], macro={})
    return ds, miner.run_class(ds)


def test_indicators_are_numerous_and_causal():
    a = synth(1)
    F, atr = IND.compute(a, "cr")
    assert IND.count(F) > 450
    F2, _ = IND.compute(a[:-50], "cr")                    # le passe ne change pas quand le futur arrive
    for ag in ("tendance", "momentum", "ichimoku", "fibonacci"):
        for nm, x in F2[ag].items():
            y = F[ag][nm][:len(x)]
            ok = np.isfinite(x) & np.isfinite(y)
            assert np.allclose(x[ok], y[ok], rtol=1e-6, atol=1e-9), (ag, nm)


def test_random_market_gives_no_strategy():
    _, res = run(False)
    assert res["combinaisons"] > 10000 and res["strategies"] == []


def test_real_edge_is_found_and_army_votes():
    ds, res = run(True)
    assert res["strategies"], "un vrai avantage doit etre trouve"
    s = res["strategies"][0]
    assert s["stats"]["test"]["win"] >= 0.80 and s["stats"]["test"]["avg"] > 0 and s["dir"] == 1


def test_signal_message_and_tracking(monkeypatch):
    sent = []
    monkeypatch.setattr(TG, "send", lambda m: sent.append(m) or True)
    strat = {"id": "x", "agent": "momentum", "dir": 1, "wilson_test": 0.8, "exit": config.EXITS[1],
             "stats": {"test": {"win": 0.86, "n": 50, "avg": 0.12}}, "mc": {"p_gain": 0.9}}
    fx = {"close": 100.0, "atr": 1.0, "ts": 1790000000.0}
    st = live.load_state()
    lab = {"classes": [{"kind": "cr", "tf": "1H", "armee": {"achat": {"win": 0.9, "n": 40}}, "strategies": [strat]}]}
    agents = {"momentum": strat, "tendance": strat, "carnet": {"agent": "carnet", "wilson_test": 0}}
    live.send_signal(st, "cr", "1H", "BTC", 1, fx, strat, agents, {}, lab)
    m = sent[0]
    assert m.startswith("🪙 <b>CRYPTO</b>") and "ACHETER BTC" in m and "3 spécialistes" in m and "Sortie de sécurité" in m
    o = st["open"][0]
    bars = np.array([[o["t_entry"] + k * 3600, 100, 100.6, 99.9, 100.4, 1] for k in range(3)])
    monkeypatch.setattr(live, "hist", lambda s, k, tf: bars)
    live.track(st, o["t_entry"] + 3 * 3600)
    assert "GAGNANT" in sent[-1] and st["track"][0]["win"] and not st["open"]


def test_elite_setup_signals_alone(monkeypatch):
    """Un setup elite (>= 90 % sur >= 100 trades) envoie seul ; un setup ordinaire seul n'envoie rien."""
    sent = []
    monkeypatch.setattr(TG, "send", lambda m: sent.append(m) or True)
    monkeypatch.setattr(live, "save_state", lambda st: None)
    mk = lambda win, n, w: {"id": "e%d" % n, "agent": "tendance", "dir": -1, "wilson_test": w, "exit": config.EXITS[1],
                            "conds": [], "stats": {"test": {"win": win, "n": n, "avg": 0.1}},
                            "robustesse": {"par_actif": {"EURUSD": {"reussite": win, "trades": 20}}}}
    fx = {"close": 100.0, "atr": 1.0, "ts": 1790000000.0, "last": {}}
    monkeypatch.setattr(live, "features", lambda *a, **k: fx)
    monkeypatch.setattr(live, "UNIVERSE", {"fx": ["EURUSD"]})
    for strat, expect in ((mk(0.86, 60, 0.75), 0), (mk(0.95, 127, 0.91), 1)):
        st = live.load_state()
        lab = {"classes": [{"kind": "fx", "tf": "1H", "armee": {}, "strategies": [strat]}]}
        n0 = len(sent)
        assert live.scan(st, lab, "1H", fx["ts"] + 3600 + 60) == expect
        assert len(sent) - n0 == expect
    assert "Setup élite" in sent[-1] and "VENDRE" in sent[-1]
    assert live.scan(st, lab, "1H", fx["ts"] + 7200 + 60) == 0       # deja un trade ouvert sur EURUSD : pas de doublon
    unproven = mk(0.95, 127, 0.91)
    unproven["robustesse"] = {"par_actif": {"GBPUSD": {"reussite": 0.95, "trades": 20}}}
    st2 = live.load_state()
    lab2 = {"classes": [{"kind": "fx", "tf": "1H", "armee": {}, "strategies": [unproven]}]}
    assert live.scan(st2, lab2, "1H", fx["ts"] + 3600 + 60) == 0      # pas prouvee sur EURUSD : aucun signal


def test_pairing_roundtrip(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    for cid in ("987654321", "-1001234567890"):
        assert TG.pair_decode(TG.pair_encode(cid)) == cid


def test_inventor_generates_applies_and_remembers():
    from lab import ideas
    a = synth(3)
    F, _ = IND.compute(a, "cr")
    names = [(ag, nm) for ag, d in F.items() for nm in d]
    specs = ideas.generate(names, n=60, seed=1)
    assert len(specs) >= 50
    built = [ideas.apply_spec(F, sp) for sp in specs]
    assert sum(x is not None and len(x) == len(a) for x in built) >= 50
    sp = specs[0]
    strat = {"conds": [["inventeur", ideas.name(sp), 0, 1]], "stats": {"test": {"win": 0.9}}}
    mem = ideas.remember([strat], {ideas.name(sp): sp}, {})
    nxt = ideas.generate(names, n=60, seed=2, memory=mem)
    assert ideas.name(sp) in {ideas.name(x) for x in nxt}       # l'idee gagnante est reprise et mutee


def test_merge_parallel_parts(tmp_path, monkeypatch):
    """Fusion des recherches paralleles : forex 15 min en premier, classe manquante gardee de l'ancienne version."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(config.CFG, "TIMEFRAMES", ["1H"])
    (tmp_path / "parts").mkdir()
    cls = lambda k, tf, n: {"kind": k, "tf": tf, "strategies": [{"conds": [["a", "x%d" % i, 0, 1]]} for i in range(n)],
                             "conditions": 1, "combinaisons": 2, "candidates": 3}
    for k, tf, n in (("cr", "1H", 0), ("fx", "1H", 2)):
        (tmp_path / "parts" / ("%s_%s.json" % (k, tf))).write_text(json.dumps(
            {"regle": {"x": 1}, "classes": [cls(k, tf, n)], "memoire": {"gagnantes": [{"nom": "a", "score": 0.8 + n / 100}]}}))
    (tmp_path / "strategies.json").write_text(json.dumps({"classes": [cls("fx", "1H", 4), cls("fx", "15M", 9)]}))   # 15 min retire
    out = miner.merge()
    assert [(c["kind"], c["tf"]) for c in out["classes"]] == [("fx", "1H"), ("cr", "1H")]
    assert len(out["classes"][0]["strategies"]) == 2 and out["total"]["strategies"] == 2   # anciennes sans date : retirees
    assert json.load(open(tmp_path / "ideas.json"))["gagnantes"][0]["score"] == pytest.approx(0.82)


def test_dukascopy_live_ticks(monkeypatch):
    """Direct forex Dukascopy : ticks horaires decodes en bougies, jours termines + heures du jour assembles."""
    import lzma, struct, datetime as dt
    from lab import data
    recs = b"".join(struct.pack(">3i2f", ms, 108510, 108500 + (ms // 60000), 1.0, 2.0) for ms in range(0, 3600000, 30000))
    blob = lzma.compress(recs, format=lzma.FORMAT_ALONE)
    monkeypatch.setattr(data, "_duka_get", lambda sym, day, fname, cache: blob if fname.endswith("ticks.bi5") else None)
    h = data._duka_hour("EURUSD", dt.datetime(2026, 10, 29, 10, tzinfo=dt.timezone.utc))
    assert len(h) == 60 and h[0][1] == pytest.approx(1.085) and h[-1][4] == pytest.approx(1.08559) and h[0][5] == 4.0
    data.LIVE_CACHE.clear()
    day_rows = [(1.7e9 + k * 60, 1.1, 1.1, 1.1, 1.1, 1.0) for k in range(1440)]
    monkeypatch.setattr(data, "_duka_day", lambda sym, d: [(r[0] + (d.toordinal() - 739000) * 86400,) + r[1:] for r in day_rows])
    a = data.dukascopy_live("EURUSD", 3600, 5)
    assert a.shape[1] == 6 and len(a) >= 24 * 3 and np.all(np.diff(a[:, 0]) > 0)


def test_truefx_realtime_minutes(monkeypatch):
    """Flux temps reel TrueFX : lignes CSV -> bougies 1 min, utilisees seulement apres une bougie complete."""
    from lab import data
    import time as _t
    now = _t.time()
    data.TRUEFX.update(start=now - 2000, min={})
    class R:
        status_code = 200
        text = "EUR/USD,%d,1.08,512,1.08,515,1.09,1.07,1.08\nUSD/JPY,%d,157.,612,157.,615,158,156,157\n" % (now * 1000, now * 1000)
    monkeypatch.setattr(data.SESSION, "get", lambda *a, **k: R())
    assert data.poll_truefx() == 2
    m = data.truefx_minutes("EURUSD", 900)          # pas de 15 min ici : debut de la premiere bougie complete
    assert m and m[-1][4] == 1.08512 and m[-1][5] == 0.0
    assert data.truefx_minutes("GBPUSD", 900) == []


def test_truefx_account_session(monkeypatch):
    """Compte TrueFX : connexion avec les secrets, puis lecture par session ; le mot de passe n'est jamais stocke."""
    from lab import data
    import time as _t
    now = _t.time()
    monkeypatch.setenv("TRUEFX_USER", "moi")
    monkeypatch.setenv("TRUEFX_PASS", "secret")
    data.TRUEFX.update(start=None, min={}, session=None, mode="public", paires=0, tente=False)
    calls = []
    class R:
        def __init__(self, text): self.status_code, self.text = 200, text
    def fake(url, params=None, timeout=None):
        calls.append(dict(params))
        if "u" in params:
            return R("moi:secret:ozrates:1793300000000")
        return R("EUR/AUD,%d,1.59,236,1.59,240,1,1,1\n" % (now * 1000))
    monkeypatch.setattr(data.SESSION, "get", fake)
    assert data.poll_truefx() == 1 and data.TRUEFX["mode"] == "compte" and data.TRUEFX["paires"] == 1
    assert calls[-1]["id"].endswith("1793300000000") and "secret" not in str(data.TRUEFX["mode"])


def test_scout_weights_inventor_and_digest(monkeypatch, tmp_path):
    """Eclaireur : mots-cles -> specialistes, poids, Inventeur oriente, resume Telegram envoye une seule fois."""
    from lab import scout, ideas
    assert scout._agents("Smart money order block and VWAP momentum") == ["momentum", "smc", "vwap"]
    w = scout.weights([{"agents": ["smc"]}, {"agents": ["smc", "momentum"]}, {"agents": []}])
    assert w["smc"] > w["momentum"] > 1
    feats = [("smc", "x%d" % i) for i in range(5)] + [("bandes", "y%d" % i) for i in range(50)]
    sp = ideas.generate(feats, n=200, seed=3, weights={"smc": 3.0})
    share = sum(1 for s in sp if s["a"][0] == "smc") / len(sp)
    assert share > 5 / 55                                         # la famille a la mode est tiree plus souvent
    sent = []
    monkeypatch.setattr(TG, "send", lambda m: sent.append(m) or True)
    v = {"date": "2026-10-30", "poids": {"smc": 3.0}, "trouvailles": [
        {"source": "GitHub", "titre": "a/b", "url": "https://github.com/a/b", "etoiles": 500, "resume": "SMC tool", "agents": ["smc"]},
        {"source": "arXiv", "titre": "Order flow paper", "url": "http://arxiv.org/abs/1", "etoiles": 0, "resume": "", "agents": ["volume"]}]}
    class R:
        def json(self): return v
    monkeypatch.setattr(live.data.SESSION, "get", lambda *a, **k: R())
    st = live.load_state()
    assert live.scout_digest(st, 1793300000) and not live.scout_digest(st, 1793300000)
    assert "ÉCLAIREUR" in sent[0] and "a/b" in sent[0] and "Order flow paper" in sent[0]


def test_round_numbers_and_weekend():
    """Conseil du 30/10 : chiffres ronds (niveaux psychologiques) et volatilite du week-end crypto."""
    from lab import extra as EX
    c = np.array([1.0849, 1.0851, 1.0899, 1.0901])
    rn = EX.round_numbers(c + 0.0002, c - 0.0002, c, np.full(4, 0.001))
    assert np.allclose(rn["dist_fin"][:2], [-0.1, 0.1]) and rn["traverse_rond"][3] == 1.0
    ts = 1.7e9 - (1.7e9 % 86400) + np.arange(24 * 7 * 6) * 3600.0       # 6 semaines de bougies 1 h
    px = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, len(ts))))
    wk = EX.weekend_vol(ts, px)
    day = (ts // 86400).astype(int)
    assert np.all(np.isnan(wk["vol_weekend"][(day + 3) % 7 >= 5]))          # jamais pendant le week-end en cours
    assert np.isfinite(wk["vol_weekend"]).sum() > 0


def test_ichimoku_core_candidates():
    """Noyau Ichimoku : chaque candidate part d'une condition Ichimoku confirmee par un AUTRE specialiste."""
    ds = miner.build("cr", "1H", lambda s, k, tf: synth(sum(map(ord, s)), edge=True), symbols=["BTC", "ETH", "SOL"], macro={})
    by = {}
    for c_ in miner.conditions(ds):
        by.setdefault(c_[0], []).append(c_)
    cands = []
    tested, n = miner.ichimoku_core(ds, by, cands, 10, 3, 60, ds["split"] == 0, pool_n=40)
    assert tested > 100 and n == len(cands)
    for cd in cands:
        ags = {c_[0] for c_ in cd["conds"]}
        assert cd["agent"] == "noyau_ichimoku" and "ichimoku" in ags and len(ags) >= 2
    F, _ = IND.compute(synth(5), "fx")
    assert {"etat_nuage_9_26", "signal_complet_9_26", "tk_croisement_9_26"} <= set(F["ichimoku"])


def test_dukascopy_monthly_hours_and_yearly_check(monkeypatch):
    """10 ans : fichiers mensuels de bougies 1 h decodes ; validation par annee presente dans la robustesse."""
    import lzma, struct
    from lab import data
    recs = b"".join(struct.pack(">5if", h * 3600, 108500, 108510, 108490, 108520, 1.0) for h in range(24 * 28))
    blob = lzma.compress(recs, format=lzma.FORMAT_ALONE)
    class R:
        status_code, content = 200, blob
    monkeypatch.delenv("DUKA_CACHE", raising=False)
    monkeypatch.setattr(data.SESSION, "get", lambda *a, **k: R())
    rows = data._duka_month("EURUSD", 2020, 2)
    assert len(rows) == 24 * 28 and rows[0][1] == pytest.approx(1.085) and rows[0][2] == pytest.approx(1.0852)
    monkeypatch.setattr(data, "dukascopy", lambda *a, **k: None)
    a = data.dukascopy_hours("EURUSD", 120)
    assert a.shape[1] == 6 and np.all(np.diff(a[:, 0]) > 0) and np.all(a[:, 5] == 0)
    ds, res = run(True)
    if res["strategies"]:
        assert "annees_reussies" in res["strategies"][0]["robustesse"]


def test_numerologie_causale():
    """Specialiste numerologies (toutes traditions) : ~75 indicateurs, sans regard vers le futur."""
    from lab import extra as EX
    ts = 1.79e9 + np.arange(500) * 3600.0
    c = 1.08 + np.cumsum(np.random.default_rng(2).normal(0, 1e-4, 500))
    a, b = EX.numerologie(ts, c, np.full(500, 1e-3)), EX.numerologie(ts[:300], c[:300], np.full(300, 1e-3))
    assert len(a) >= 70 and all(np.allclose(a[k][:300], b[k], equal_nan=True) for k in a)
    assert set(np.unique(a["racine_numerique"])) <= set(range(1, 10))


def test_numerologie_a_part(monkeypatch):
    """La numerologie ne vote pas avec l'armee et signale a part (>= 92 % prouve sur la paire)."""
    sent = []
    monkeypatch.setattr(TG, "send", lambda m: sent.append(m) or True)
    monkeypatch.setattr(live, "save_state", lambda st: None)
    s = {"id": "n1", "agent": "numerologie", "dir": 1, "wilson_test": 0.9, "exit": config.EXITS[1], "conds": [],
         "stats": {"test": {"win": 0.95, "n": 120, "avg": 0.1}},
         "robustesse": {"par_actif": {"EURUSD": {"reussite": 0.95, "trades": 20}}}}
    lab = {"classes": [{"kind": "fx", "tf": "1H", "armee": {}, "strategies": [s]}]}
    fx = {"close": 100.0, "atr": 1.0, "ts": 1790000000.0, "last": {}}
    assert live.council_votes(lab, "fx", "1H", "EURUSD", fx, set()) == {1: {}, -1: {}}
    monkeypatch.setattr(live, "features", lambda *a, **k: fx)
    monkeypatch.setattr(live, "UNIVERSE", {"fx": ["EURUSD"]})
    st = live.load_state()
    assert live.scan(st, lab, "1H", fx["ts"] + 3600 + 60) == 1 and "numérologie (à part)" in sent[-1]


def test_recherche_dediee_numerologie(monkeypatch):
    """Recherche parallele : AGENTS_ONLY=numerologie ne produit que des candidates numerologie."""
    monkeypatch.setenv("AGENTS_ONLY", "numerologie")
    ds = miner.build("cr", "1H", lambda s, k, tf: synth(sum(map(ord, s)), edge=True), symbols=["BTC", "ETH"], macro={})
    cands, tested, _ = miner.search(ds)
    assert tested > 0 and all(cd["agent"] == "numerologie" for cd in cands)
    assert all(c_[0] == "numerologie" for cd in cands for c_ in cd["conds"])


def test_merge_familles_de_sorties(tmp_path, monkeypatch):
    """Recherches paralleles par familles de sorties : les strategies d'une meme classe s'additionnent."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(config.CFG, "TIMEFRAMES", ["1H"])
    (tmp_path / "parts").mkdir()
    st = lambda w, e: {"conds": [], "exit": {"id": e}, "wilson_test": w, "stats": {"test": {"win": w}}}
    for sh, w, e in (("e1e2e3", 0.93, "e1"), ("e6e7", 0.96, "e6")):
        (tmp_path / "parts" / ("fx_1H_%s.json" % sh)).write_text(json.dumps({"regle": {}, "classes": [
            {"kind": "fx", "tf": "1H", "strategies": [st(w, e)], "conditions": 10, "combinaisons": 5, "candidates": 2,
             "validees": 1, "apres_fdr": 1, "armee": {"vente": {"win": w, "n": 30}}}]}))
    out = miner.merge()
    c = out["classes"][0]
    assert len(c["strategies"]) == 2 and c["strategies"][0]["stats"]["test"]["win"] == 0.96
    assert c["combinaisons"] == 10 and c["armee"]["vente"]["win"] == 0.96


def test_concepts_des_maitres():
    """Hosoda / Sasaki / Peloille, Steidlmayer / Dalton, Shannon, Wyckoff : presents et sans regard vers le futur."""
    a = synth(11)
    F, _ = IND.compute(a, "cr")
    for ag, k in (("ichimoku", "temps_hosoda_cle"), ("ichimoku", "objectif_N"), ("ichimoku", "chikou_libre"),
                  ("profil", "regle_80"), ("profil", "poc_veille_non_teste"), ("vwap", "vwap_ancre_semaine"),
                  ("structure", "spring_upthrust")):
        assert k in F[ag]
    m = 1800
    F2, _ = IND.compute(a[:m], "cr")
    for ag in ("ichimoku", "profil", "vwap", "structure"):
        for k, x in F2[ag].items():
            assert np.allclose(np.nan_to_num(F[ag][k][:m]), np.nan_to_num(x)), (ag, k)


def test_talib_complet_causal():
    """Suite TA-Lib complete : nombreux indicateurs, aucun regard vers le futur."""
    a = synth(13)
    F, _ = IND.compute(a, "fx")
    assert len(F["talib"]) >= 40
    F2, _ = IND.compute(a[:1800], "fx")
    for k, x in F2["talib"].items():
        assert np.allclose(np.nan_to_num(F["talib"][k][:1800]), np.nan_to_num(x)), k


def test_grands_traders_causal():
    """Specialiste grands traders : regles publiques, aucun regard vers le futur."""
    a = synth(17)
    F, _ = IND.compute(a, "cr")
    assert len(F["traders"]) >= 25
    F2, _ = IND.compute(a[:1800], "cr")
    for k, x in F2["traders"].items():
        assert np.allclose(np.nan_to_num(F["traders"][k][:1800]), np.nan_to_num(x)), k


def test_liquidite_causal():
    """Specialiste liquidite : niveaux, chasses de stops, mesures ; aucun regard vers le futur."""
    a = synth(19)
    F, _ = IND.compute(a, "cr")
    assert len(F["liquidite"]) >= 25
    F2, _ = IND.compute(a[:1800], "cr")
    for k, x in F2["liquidite"].items():
        assert np.allclose(np.nan_to_num(F["liquidite"][k][:1800]), np.nan_to_num(x)), k


def test_flux_operateurs_causal():
    """Flux des operateurs (fixings, Gotobi, fin de mois, NFP, SMT, force, gaps) : aucun regard vers le futur."""
    a, b = synth(21), synth(22)
    pan = {"EURUSD": b[:, 4], "GBPUSD": a[:, 4], "EURGBP": b[:, 4] / a[:, 4]}
    refs = {"sym": "GBPUSD", "bench": b[:, 4], "panier": pan}
    F, _ = IND.compute(a, "fx", refs)
    assert len(F["flux"]) >= 20
    m = 1800
    F2, _ = IND.compute(a[:m], "fx", {"sym": "GBPUSD", "bench": b[:m, 4], "panier": {k: x[:m] for k, x in pan.items()}})
    for k, x in F2["flux"].items():
        assert np.allclose(np.nan_to_num(F["flux"][k][:m]), np.nan_to_num(x)), k


def test_positionnement_causal():
    """COT (publie avec retard) et DVOL alignes sur la derniere valeur connue : aucun regard vers le futur."""
    from lab import positionnement as PO
    a = synth(23)
    ts, c = a[:, 0], a[:, 4]
    wk = np.arange(ts[0] - 200 * 7 * 86400, ts[-1], 7 * 86400)
    rng = np.random.default_rng(1)
    cot = {k: np.c_[wk, rng.normal(0, .2, (len(wk), 3))] for k in ("BTC", "EUR", "USD")}
    dv = {"BTC": np.c_[ts, 50 + rng.normal(0, 3, len(ts))]}
    F = PO.compute_positionnement("ETH", "cr", ts, c, {"cot": cot, "dvol": dv})
    G = PO.compute_positionnement("EURUSD", "fx", ts, c, {"cot": cot})
    assert len(F) >= 6 and len(G) >= 8
    m = 1800
    F2 = PO.compute_positionnement("ETH", "cr", ts[:m], c[:m], {"cot": cot, "dvol": {"BTC": dv["BTC"][:m]}})
    for k, x in F2.items():
        assert np.allclose(np.nan_to_num(F[k][:m]), np.nan_to_num(x)), k


def test_plafond_meilleures_pistes():
    """Le labo publie le classement honnete des meilleures pistes qui ont echoue, avec la raison."""
    _, res = run(True)
    assert "plafond" in res
    for r in res["plafond"]:
        assert 0 <= r["test_win"] <= 1 and r["test_n"] >= 30 and r["raison"]


def test_strategie_activee_par_le_trader(monkeypatch, tmp_path):
    """activation.json : la strategie est prise dans le plafond avec ses zones, signal seulement sur les paires
       gagnantes, message Telegram marque « activee par toi »."""
    json.dump({"regles": [{"nom": "Fixing de Londres", "kind": "fx", "tf": "1H", "agent": "flux", "dir": -1, "exit": "e2",
                           "doit_contenir": [["flux", "heures_fixing_londres"]], "taux_min": 0.88, "trades_min": 500,
                           "paire_taux_min": 0.85, "paire_trades_min": 10}]}, open(tmp_path / "activation.json", "w"))
    monkeypatch.setattr(live, "ACT_URL", "http://127.0.0.1:9/activation.json")
    pl = {"agent": "flux", "dir": -1, "exit": "e2", "test_win": 0.898, "test_n": 1217, "test_avg": 0.06,
          "conds": [["flux", "heures_fixing_londres", 0.5, 2.5], ["flux", "mouvement_avant_tokyo", -1.0, 0.3]],
          "robustesse": {"par_actif": {"EURUSD": {"reussite": 0.9, "trades": 60, "gain_moyen": 0.05},
                                       "USDJPY": {"reussite": 0.8, "trades": 60, "gain_moyen": -0.01}}}}
    lab = {"classes": [{"kind": "fx", "tf": "1H", "strategies": [], "plafond": [pl]}]}
    act = live.activated(lab)
    assert len(act) == 1 and act[0]["paires_ok"] == ["EURUSD"]
    lab["activees"] = act
    sent = []
    monkeypatch.setattr(TG, "send", lambda txt, *a, **k: sent.append(txt))
    monkeypatch.setattr(live, "save_state", lambda st: None)
    st = {"open": [], "alerts": {}, "retired": []}
    fx = {"last": {("flux", "heures_fixing_londres"): 1.0, ("flux", "mouvement_avant_tokyo"): 0.0},
          "close": 1.1, "atr": 0.001, "ts": 1790000000.0}
    assert live.activated_signals(st, lab, "fx", "1H", "USDJPY", fx) == 0      # paire perdante : jamais de signal
    assert live.activated_signals(st, lab, "fx", "1H", "EURUSD", fx) == 1
    assert "ACTIVÉE PAR TOI" in sent[0] and "VENDRE" in sent[0]
    assert live.activated_signals(st, lab, "fx", "1H", "EURUSD", fx) == 0      # un seul trade par paire


def test_seulement_activees_coupe_le_reste(monkeypatch):
    """Exigence du trader : avec SEULEMENT_ACTIVEES, aucun resultat d'un ancien trade de l'armee sur Telegram."""
    monkeypatch.setitem(config.CFG, "SEULEMENT_ACTIVEES", 1)
    sent = []
    monkeypatch.setattr(TG, "send", lambda txt, *a, **k: sent.append(txt))
    monkeypatch.setattr(live, "save_state", lambda st: None)
    now = 1790000000.0
    a = np.c_[[now - 7200, now - 3600], [1.0, 1.0], [1.2, 1.2], [0.9, 0.9], [1.1, 1.1], [0, 0]]
    monkeypatch.setattr(live, "hist", lambda *x: a)
    st = {"open": [{"kind": "fx", "tf": "1H", "sym": "EURUSD", "dir": 1, "entry": 1.0, "sl": 0.8, "tp": 1.05,
                    "t_entry": now - 7200, "t_exit": now - 10, "ids": ["x"], "agents": ["tendance"]}],
          "alerts": {}, "retired": [], "live": {}, "track": []}
    live.track(st, now)
    assert sent == [] and len(st["track"]) == 1


def test_mode_gros_gains(monkeypatch):
    """Choix du trader : objectif >= stop, barre = facteur de profit, classement par gain total."""
    monkeypatch.setitem(config.CFG, "MODE", "gain")
    monkeypatch.setitem(config.CFG, "MIN_PF", 1.2)
    monkeypatch.setattr(miner, "EXITS", config.EXITS_GAIN)
    assert all(e["tp"] >= e["sl"] for e in config.EXITS_GAIN)
    ds, res = run(True)
    for s in res["strategies"]:
        t = s["stats"]["test"]
        assert t["avg"] > 0 and t["pf"] >= 1.2 and s["exit"]["tp"] >= s["exit"]["sl"]
    tots = [s["stats"]["test"]["avg"] * s["stats"]["test"]["n"] for s in res["strategies"]]
    assert tots == sorted(tots, reverse=True)
    assert all("test_pf" in r for r in res["plafond"])


def test_vraie_tendance_stop_suiveur(monkeypatch):
    """Vraie tendance : stop suiveur, pas d'objectif ; au hasard pas d'avantage, en tendance de gros gains ;
       en direct le stop se deplace (message Telegram) puis la sortie se fait au stop suiveur."""
    from lab import sim
    rng = np.random.default_rng(4)
    n = 6000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))
    o = np.r_[c[0], c[:-1]]
    a = np.c_[np.arange(n) * 3600.0, o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c, np.zeros(n)]
    atr = np.full(n, 0.35)
    e = {"id": "t", "tp": 99, "sl": 1.0, "trail": 1.5, "h": 72}
    r, w, v = sim.outcomes(a, atr, e, None, 0.0)[1]
    assert abs(r[v].mean()) < 0.15 and r[v].max() > 3          # au hasard : rien de gratuit, mais des gains qui courent
    up = np.linspace(100, 130, 300)
    a2 = np.c_[np.arange(300) * 3600.0, up, up + 0.05, up - 0.05, up, np.zeros(300)]
    r2, w2, v2 = sim.outcomes(a2, np.full(300, 0.2), e, None, 0.0)[1]
    assert r2[v2][:100].min() > 3                              # tendance reguliere : on laisse courir (> 3 x le risque)
    # direct
    sent = []
    monkeypatch.setattr(TG, "send", lambda txt, *a_, **k: sent.append(txt))
    monkeypatch.setattr(live, "save_state", lambda st: None)
    t0 = 1790000000.0
    bars = np.c_[t0 + np.arange(6) * 3600, [1.00, 1.01, 1.02, 1.03, 1.03, 1.03], [1.011, 1.021, 1.031, 1.035, 1.032, 1.031],
                 [0.999, 1.009, 1.019, 1.029, 1.0, 1.0], [1.01, 1.02, 1.03, 1.03, 1.01, 1.01], np.zeros(6)]
    monkeypatch.setattr(live, "hist", lambda *x: bars)
    st = {"open": [{"kind": "fx", "tf": "1H", "sym": "EURUSD", "dir": 1, "entry": 1.0, "sl": 0.99, "sl0": 0.99,
                    "sl_sent": 0.99, "tp": 99, "trail": 1.0, "atr": 0.01, "t_entry": t0, "t_exit": t0 + 100 * 3600,
                    "ids": ["x"], "agents": ["liquidite"]}], "alerts": {}, "retired": [], "live": {}, "track": []}
    live.track(st, t0 + 4 * 3600)                              # 4 bougies cloturees, prix monte : le stop suit
    assert any("Déplace ton stop" in m for m in sent) and st["open"][0]["sl"] > 1.0
    live.track(st, t0 + 6 * 3600)                              # retour sous le stop suiveur : sortie gagnante
    assert st["open"] == [] and st["track"][-1]["win"]


def test_labo_en_mode_tendance(monkeypatch):
    """Le labo complet tourne avec les sorties « vraie tendance » (stop suiveur) et les regles gros gains."""
    monkeypatch.setitem(config.CFG, "MODE", "tendance")
    monkeypatch.setitem(config.CFG, "MIN_PF", 1.1)
    monkeypatch.setattr(miner, "EXITS", config.EXITS_TREND[:2])
    ds, res = run(True)
    assert "plafond" in res
    for s in res["strategies"]:
        assert s["exit"].get("trail") and s["stats"]["test"]["avg"] > 0


def test_sortie_en_deux_temps(monkeypatch):
    """>= 90 % et gros gains : moitie encaissee au 1er objectif, stop au prix d'entree, le reste court la tendance.
       Au hasard aucun gain gratuit ; en direct : message « moitie encaissee » puis sortie gagnante au prix d'entree."""
    from lab import sim
    rng = np.random.default_rng(5)
    n = 8000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, n)))
    o = np.r_[c[0], c[:-1]]
    a = np.c_[np.arange(n) * 3600.0, o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c, np.zeros(n)]
    e = config.EXITS_SCALE[0]
    r, w, v = sim.outcomes(a, np.full(n, 0.35), e, None, 1.5)[1]
    assert r[v].mean() < 0.05 and w[v].mean() < 0.85            # au hasard : pas de 90 % gratuit
    assert r[w & v].min() > 0                                    # tout trade gagnant a encaisse sa moitie
    up = np.linspace(100, 130, 300)
    a2 = np.c_[np.arange(300) * 3600.0, up, up + 0.05, up - 0.05, up, np.zeros(300)]
    r2, w2, v2 = sim.outcomes(a2, np.full(300, 0.2), e, None, 0.0)[1]
    assert w2[v2][:100].all() and r2[v2][:100].min() > 2         # vraie tendance : la 2e moitie fait le gros gain
    sent = []
    monkeypatch.setattr(TG, "send", lambda txt, *a_, **k: sent.append(txt))
    monkeypatch.setattr(live, "save_state", lambda st: None)
    t0 = 1790000000.0
    bars = np.c_[t0 + np.arange(4) * 3600, [1.0, 1.004, 1.003, 1.001], [1.004, 1.006, 1.004, 1.002],
                 [0.999, 1.003, 1.002, 0.998], [1.004, 1.003, 1.001, 0.999], np.zeros(4)]
    monkeypatch.setattr(live, "hist", lambda *x: bars)
    st = {"open": [{"kind": "fx", "tf": "1H", "sym": "EURUSD", "dir": 1, "entry": 1.0, "sl": 0.99, "sl0": 0.99,
                    "sl_sent": 0.99, "tp": 1.005, "tp1": 1.005, "frac": 0.5, "trail": 1.0, "atr": 0.01,
                    "t_entry": t0, "t_exit": t0 + 100 * 3600, "ids": ["x"], "agents": ["flux"]}],
          "alerts": {}, "retired": [], "live": {}, "track": []}
    live.track(st, t0 + 4 * 3600)
    assert any("Moitié encaissée" in m for m in sent)
    assert st["open"] == [] and st["track"][-1]["win"]            # le reste sort au prix d'entree : trade gagnant


def test_labo_en_mode_tendance90(monkeypatch):
    monkeypatch.setitem(config.CFG, "MODE", "tendance90")
    monkeypatch.setattr(miner, "EXITS", config.EXITS_SCALE[:2])
    ds, res = run(True)
    assert "plafond" in res
    for s in res["strategies"]:
        assert s["exit"].get("tp1") and s["stats"]["test"]["win"] >= config.CFG["MIN_WIN"]


def test_activation_auto_meilleures_pistes(monkeypatch, tmp_path):
    """Les meilleures pistes du plafond >= seuil sont activees automatiquement (identifiant stable)."""
    json.dump({"regles": [{"nom": "Meilleure piste crypto", "auto": True, "kind": "cr", "tf": "1H", "taux_min": 0.84,
                           "trades_min": 50, "max": 2, "exclure": ["numerologie"], "paire_taux_min": 0.8,
                           "paire_trades_min": 5}]}, open(tmp_path / "activation.json", "w"))
    monkeypatch.setattr(live, "ACT_URL", "http://127.0.0.1:9/activation.json")
    pa = {"BTC": {"reussite": 0.9, "trades": 8, "gain_moyen": 0.2}, "ETH": {"reussite": 0.6, "trades": 8, "gain_moyen": -0.1}}
    mk = lambda w, n, ag="ichimoku": {"agent": ag, "dir": 1, "exit": config.EXITS[0]["id"], "test_win": w, "test_n": n,
                                      "test_avg": 0.1, "conds": [[ag, "x", 0.0, 1.0]], "robustesse": {"par_actif": pa}}
    lab = {"classes": [{"kind": "cr", "tf": "1H", "plafond": [mk(0.95, 40), mk(0.887, 71), mk(0.99, 80, "numerologie"),
                                                               mk(0.80, 300)]}]}
    a1, a2 = live.activated(lab), live.activated(lab)
    assert len(a1) == 1 and a1[0]["paires_ok"] == ["BTC"] and a1[0]["id"] == a2[0]["id"] and "89 %" in a1[0]["activee"]


def test_deux_voies_resultats_recents_gardes(tmp_path, monkeypatch):
    """Deux voies en parallele : les pistes recentes (< 24 h) de l'autre voie sont gardees, les vieilles retirees."""
    import time as _t
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(config.CFG, "TIMEFRAMES", ["1H"])
    (tmp_path / "parts").mkdir()
    row = lambda nm, w, vu=None: dict({"agent": "flux", "dir": 1, "exit": "e2", "conds": [["flux", nm, 0, 1]],
                                       "test_win": w, "test_n": 100}, **({"vu": vu} if vu else {}))
    (tmp_path / "parts" / "fx_1H.json").write_text(json.dumps({"regle": {}, "classes": [
        {"kind": "fx", "tf": "1H", "strategies": [], "plafond": [row("neuf", 0.84)], "conditions": 1, "combinaisons": 1,
         "candidates": 1}]}))
    (tmp_path / "strategies.json").write_text(json.dumps({"classes": [{"kind": "fx", "tf": "1H", "strategies": [],
        "plafond": [row("autre_voie", 0.88, _t.time() - 3600), row("vieux", 0.95, _t.time() - 2 * 86400)]}]}))
    pl = miner.merge()["classes"][0]["plafond"]
    assert [r["conds"][0][1] for r in pl] == ["autre_voie", "neuf"]


def test_noyau_liquidite_order_block(monkeypatch):
    """Noyau Liquidite / Order Block : chaque piste part d'un setup de liquidite (smc / liquidite / balayage)."""
    ds = miner.build("cr", "1H", lambda s, k, tf: synth(sum(map(ord, s)), edge=True), symbols=["BTC", "ETH", "SOL"], macro={})
    conds = miner.conditions(ds)
    by_agent = {}
    for c_ in conds:
        by_agent.setdefault(c_[0], []).append(c_)
    cands = []
    tested, n = miner.liquidity_core(ds, by_agent, cands, 10, 2, 30, ds["split"] == 0, pool_n=30)
    assert tested > 0
    for cd in cands:
        assert cd["agent"] == "noyau_liquidite"
        assert any(c_[0] in miner.LIQ_CORE or c_[1] in miner.LIQ_AMD for c_ in cd["conds"])


def test_smc_ict_avance_causal():
    """Order blocks valides, breakers, FVG / iFVG, modele chasse -> MSS -> OTE : aucun regard vers le futur."""
    a = synth(29)
    F, _ = IND.compute(a, "fx")
    for k in ("ob_haussier_dist", "dans_breaker_baissier", "ifvg_haussier_dist", "modele_ict", "ict_haussier_ote",
              "ob_haussier_premier_contact", "chasse_puis_mss"):
        assert k in F["smc"]
    F2, _ = IND.compute(a[:1800], "fx")
    for k, x in F2["smc"].items():
        assert np.allclose(np.nan_to_num(F["smc"][k][:1800]), np.nan_to_num(x)), k


def test_order_flow_crypto_et_forex():
    """Order flow : vrai delta (crypto), delta estime (forex), checklist d'entree ; aucun regard vers le futur ;
       volume inconnu -> aucun signal."""
    from lab import orderflow as OF
    a = synth(31)
    ts, o, h, l, c, v = (a[:, i] for i in range(6))
    F, atr = IND.compute(a, "cr")
    rng = np.random.default_rng(2)
    vol = rng.uniform(100, 200, len(c))
    buy = vol * rng.uniform(0.3, 0.7, len(c))
    for flow in ({"vol": vol, "buy": buy}, {"vol": vol, "buy": None}):
        G = OF.compute_orderflow("cr", ts, o, h, l, c, atr, F, flow)
        assert len(G) >= 12 and "modele_order_flow" in G
        m = 1800
        Fm, atm = IND.compute(a[:m], "cr")
        G2 = OF.compute_orderflow("cr", ts[:m], o[:m], h[:m], l[:m], c[:m], atm, Fm,
                                  {"vol": vol[:m], "buy": None if flow["buy"] is None else buy[:m]})
        for k, x in G2.items():
            assert np.allclose(np.nan_to_num(G[k][:m]), np.nan_to_num(x)), k
    vol2 = vol.copy()
    vol2[-1] = np.nan
    G3 = OF.compute_orderflow("fx", ts, o, h, l, c, atr, F, {"vol": vol2, "buy": None})
    assert all(not np.isfinite(x[-1]) for x in G3.values())
