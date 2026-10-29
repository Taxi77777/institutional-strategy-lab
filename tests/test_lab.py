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


def test_pairing_roundtrip(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    for cid in ("987654321", "-1001234567890"):
        assert TG.pair_decode(TG.pair_encode(cid)) == cid
