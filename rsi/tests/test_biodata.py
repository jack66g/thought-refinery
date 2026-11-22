# -*- coding: utf-8 -*-
import math

from rsi.biodata.state import BioData, load_biodata

NOW = 1000000.0


def test_first_event_does_not_move_traits():
    bd = BioData()
    bd.apply({"trait_hint": {"neuroticism": 1.0}}, now=NOW)
    assert bd.traits["neuroticism"] == 50.0  # shrink=0/(0+50)=0
    bd.apply({"trait_hint": {"neuroticism": 1.0}}, now=NOW + 60)
    assert 0 < bd.traits["neuroticism"] - 50.0 < 0.1  # 收缩且限幅


def test_mood_ema_and_clamp():
    bd = BioData()
    bd.apply({"mood_delta": {"p": 1.0}}, now=NOW)
    assert abs(bd.mood["p"] - 0.2) < 1e-9  # α=0.2
    bd.apply({"mood_delta": {"p": -1.0, "a": 1.0, "d": 1.0}}, now=NOW)  # 同时刻无衰减
    assert abs(bd.mood["p"]) < 1e-9
    assert abs(bd.mood["a"] - 0.2) < 1e-9


def test_mood_decays_toward_baseline():
    bd = BioData()
    bd.apply({"mood_delta": {"p": 1.0}}, now=NOW)      # p=0.2
    bd.tick(now=NOW + 24 * 3600)                        # 半衰期24h
    assert abs(bd.mood["p"] - 0.1) < 1e-6


def test_emotion_decay():
    bd = BioData()
    bd.apply({"emotion": {"label": "喜", "intensity": 60}}, now=NOW)
    bd.tick(now=NOW + 10 * 60)                          # 10 分钟
    assert abs(bd.emotion["intensity"] - 60 * 0.95 ** 10) < 1e-6
    bd.tick(now=NOW + 10 * 3600)                        # 长时间后归零
    assert bd.emotion["label"] is None


def test_need_pulse_and_homeostasis():
    bd = BioData()
    bd.apply({"need_pulse": {"energy": -1}}, now=NOW)   # 60-15=45
    assert abs(bd.needs["energy"] - 45.0) < 1e-9
    bd.tick(now=NOW + 10 * 3600)                        # τ=10h → 缺口×e^-1
    assert abs(bd.needs["energy"] - (60 - 15 * math.exp(-1))) < 1e-6


def test_hard_clamp():
    bd = BioData()
    bd.apply({"need_pulse": {"energy": -2}}, now=NOW)
    bd.apply({"need_pulse": {"energy": -2}}, now=NOW + 60)  # -60 但夹断到基线-40
    assert bd.needs["energy"] == 20.0


def test_baseline_reestimate():
    bd = BioData()
    bd.since_baseline = 50
    bd.apply({"mood_delta": {"p": 1.0}}, now=NOW)       # p=0.2, 触发重估
    assert abs(bd.mood_baseline["p"] - 0.02) < 1e-9     # 0.9*0+0.1*0.2


def test_mood_bucket_and_card():
    bd = BioData()
    assert bd.mood_bucket() == "neutral"
    bd.apply({"mood_delta": {"p": 1.0}}, now=NOW)
    bd.apply({"mood_delta": {"p": 1.0}}, now=NOW)  # p≈0.4，越过 0.2 分桶线
    assert bd.mood_bucket() == "positive"
    bd2 = BioData()
    bd2.apply({"mood_delta": {"p": -1.0}}, now=NOW)
    bd2.apply({"mood_delta": {"p": -1.0}}, now=NOW)
    assert bd2.mood_bucket() == "negative"
    assert "人格状态卡" in BioData().persona_card()


def test_persistence_roundtrip(tmp_path):
    bd = BioData()
    bd.apply({"mood_delta": {"p": 0.5}, "need_pulse": {"curiosity": 1},
              "emotion": {"label": "喜", "intensity": 40}}, now=NOW)
    path = tmp_path / "state.json"
    bd.save(path)
    bd2 = load_biodata(path)
    assert bd2.mood == bd.mood
    assert bd2.needs == bd.needs
    assert bd2.updates == bd.updates
