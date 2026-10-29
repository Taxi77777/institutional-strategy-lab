import json
"""Tests hors ligne du labo : le hasard ne doit rien donner, un vrai avantage doit etre trouve, l'armee doit voter."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lab import config, indicators as IND, live, miner, telegram as TG  # noqa: E402


def synth(seed, n=2600, edge=False):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 900 + 1790000000
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
    monkeypatch.setitem(config.CFG, "PAUSE", 0)
    monkeypatch.setitem(config.CFG, "SIMS", 500)
    monkeypatch.setitem(config.CFG, "BEAM", 15)
    monkeypatch.setitem(config.CFG, "MIN_TRADES_TEST_15M", 30)     # petites donnees de test
    monkeypatch.setitem(config.CFG, "STATE_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def run(edge):
    ds = miner.build("cr", "15M", lambda s, k, tf: synth(sum(map(ord, s)), edge=edge), symbols=["BTC", "ETH", "SOL"], macro={})
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
    lab = {"classes": [{"kind": "cr", "tf": "15M", "armee": {"achat": {"win": 0.9, "n": 40}}, "strategies": [strat]}]}
    agents = {"momentum": strat, "tendance": strat, "carnet": {"agent": "carnet", "wilson_test": 0}}
    live.send_signal(st, "cr", "15M", "BTC", 1, fx, strat, agents, {}, lab)
    m = sent[0]
    assert m.startswith("🪙 <b>CRYPTO</b>") and "ACHETER BTC" in m and "3 spécialistes" in m and "Sortie de sécurité" in m
    o = st["open"][0]
    bars = np.array([[o["t_entry"] + k * 900, 100, 100.6, 99.9, 100.4, 1] for k in range(3)])
    monkeypatch.setattr(live, "hist", lambda s, k, tf: bars)
    live.track(st, o["t_entry"] + 3 * 900)
    assert "GAGNANT" in sent[-1] and st["track"][0]["win"] and not st["open"]


def test_elite_setup_signals_alone(monkeypatch):
    """Un setup elite (>= 90 % sur >= 100 trades) envoie seul ; un setup ordinaire seul n'envoie rien."""
    sent = []
    monkeypatch.setattr(TG, "send", lambda m: sent.append(m) or True)
    monkeypatch.setattr(live, "save_state", lambda st: None)
    mk = lambda win, n, w: {"id": "e%d" % n, "agent": "tendance", "dir": -1, "wilson_test": w, "exit": config.EXITS[1],
                            "conds": [], "stats": {"test": {"win": win, "n": n, "avg": 0.1}}}
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
    (tmp_path / "parts").mkdir()
    cls = lambda k, tf, n: {"kind": k, "tf": tf, "strategies": [{"conds": []}] * n, "conditions": 1, "combinaisons": 2,
                             "candidates": 3}
    for k, tf, n in (("cr", "1H", 0), ("fx", "5M", 1), ("fx", "15M", 2)):
        (tmp_path / "parts" / ("%s_%s.json" % (k, tf))).write_text(json.dumps(
            {"regle": {"x": 1}, "classes": [cls(k, tf, n)], "memoire": {"gagnantes": [{"nom": "a", "score": 0.8 + n / 100}]}}))
    (tmp_path / "strategies.json").write_text(json.dumps({"classes": [cls("fx", "1H", 4), cls("fx", "15M", 9)]}))
    out = miner.merge()
    assert [(c["kind"], c["tf"]) for c in out["classes"]] == [("fx", "15M"), ("fx", "5M"), ("fx", "1H"), ("cr", "1H")]
    assert out["classes"][0]["strategies"] == [{"conds": []}] * 2 and out["total"]["strategies"] == 7
    assert json.load(open(tmp_path / "ideas.json"))["gagnantes"][0]["score"] == pytest.approx(0.82)
