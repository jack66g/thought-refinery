"""仿生人格状态机（V3-T1）——四层数值模型 + 防漂移三件套。

依据（docs/05 调研）：
- 特质层 Big5+H，T 分制，慢速贝叶斯收缩 T+=N/(N+50)·δ（月年级变化）
- 心境层 PAD [-1,1]，EMA α=0.2 + 均值回归基线（半衰期 24h）
- 情绪层 Ekman-6 × 强度，0.95^t 分钟级衰减
- 需求层 稳态 RL：向设定点指数回归（τ≈10h）+ 事件脉冲
- 防漂移：均值回归 + 硬夹断(基线±2SD) + 基线周期重估(EWMA 0.9)
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRAIT_KEYS = ["openness", "conscientiousness", "extraversion",
              "agreeableness", "neuroticism", "honesty_humility"]
NEED_KEYS = ["energy", "stress", "curiosity", "social", "autonomy", "competence"]
EMOTIONS = ("喜", "怒", "哀", "惧", "厌", "惊")

_K_SHRINK = 50          # 特质收缩系数
_TRAIT_MAX_STEP = 1.0   # 单事件特质最大移动（T 分）
_MOOD_ALPHA = 0.2       # 心境 EMA
_MOOD_HALFLIFE_H = 24.0
_EMOTION_DECAY_PER_MIN = 0.95
_NEED_TAU_H = 10.0
_NEED_PULSE = 15.0
_BASELINE_EVERY = 50    # 每 N 次更新重估基线
_BASELINE_EWMA = 0.9
_SD = {"trait": 10.0, "mood": 0.5, "need": 20.0}


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class BioData:
    traits: dict[str, float] = field(default_factory=lambda: {k: 50.0 for k in TRAIT_KEYS})
    trait_baseline: dict[str, float] = field(default_factory=lambda: {k: 50.0 for k in TRAIT_KEYS})
    mood: dict[str, float] = field(default_factory=lambda: {"p": 0.0, "a": 0.0, "d": 0.0})
    mood_baseline: dict[str, float] = field(default_factory=lambda: {"p": 0.0, "a": 0.0, "d": 0.0})
    emotion: dict[str, Any] = field(default_factory=lambda: {"label": None, "intensity": 0.0})
    emotion_at: float = 0.0
    needs: dict[str, float] = field(default_factory=lambda: {k: 60.0 for k in NEED_KEYS})
    need_baseline: dict[str, float] = field(default_factory=lambda: {k: 60.0 for k in NEED_KEYS})
    updates: int = 0
    since_baseline: int = 0
    _last_t: float = 0.0

    # ---------- 时间演化（读取/更新前先调用） ----------

    def tick(self, now: float | None = None) -> None:
        """向基线回归：心境指数半衰期、情绪 0.95^t、需求稳态回归。"""
        now = now if now is not None else time.time()
        if self._last_t <= 0:
            self._last_t = now
            return
        dt_h = max(0.0, (now - self._last_t) / 3600.0)
        self._last_t = now
        # 心境：向基线指数回归（半衰期 24h）
        keep = 0.5 ** (dt_h / _MOOD_HALFLIFE_H)
        for ax in self.mood:
            self.mood[ax] = self.mood_baseline[ax] + (self.mood[ax] - self.mood_baseline[ax]) * keep
        # 情绪：分钟级衰减
        mins = dt_h * 60.0
        self.emotion["intensity"] *= _EMOTION_DECAY_PER_MIN ** mins
        if self.emotion["intensity"] < 5 or self.emotion["label"] is None:
            self.emotion = {"label": None, "intensity": 0.0}
        # 需求：向设定点（基线）指数回归 τ=10h
        keep_n = math.exp(-dt_h / _NEED_TAU_H)
        for k in self.needs:
            self.needs[k] = self.need_baseline[k] + (self.needs[k] - self.need_baseline[k]) * keep_n
        self._clamp()

    # ---------- 事件更新（观察者输出 → 各层） ----------

    def apply(self, obs: dict[str, Any], now: float | None = None) -> dict[str, Any]:
        """obs = 观察者 JSON：mood_delta{p,a,d∈[-1,1]} / emotion{label,intensity} /
        need_pulse{各∈[-2,2]} / trait_hint{各∈[-1,1]}。返回应用摘要。"""
        self.tick(now)
        shrink = self.updates / (self.updates + _K_SHRINK)
        # 特质：慢速收缩 + 单事件限幅 + 硬夹断
        for k, dv in (obs.get("trait_hint") or {}).items():
            if k in self.traits:
                step = _clip(float(dv), -1, 1) * shrink * _TRAIT_MAX_STEP
                self.traits[k] = _clip(self.traits[k] + step,
                                       self.trait_baseline[k] - 2 * _SD["trait"],
                                       self.trait_baseline[k] + 2 * _SD["trait"])
        # 心境：EMA 前移
        for ax, dv in (obs.get("mood_delta") or {}).items():
            if ax in self.mood:
                self.mood[ax] = _clip(self.mood[ax] + _MOOD_ALPHA * _clip(float(dv), -1, 1),
                                      self.mood_baseline[ax] - 2 * _SD["mood"],
                                      self.mood_baseline[ax] + 2 * _SD["mood"])
        # 情绪：事件脉冲（取新值，强度做温和混合防单次跳满）
        emo = obs.get("emotion") or {}
        label = emo.get("label")
        if label in EMOTIONS:
            inten = _clip(float(emo.get("intensity", 30)), 0, 100)
            self.emotion = {"label": label,
                            "intensity": max(inten, self.emotion["intensity"] * 0.5)}
            self.emotion_at = self._last_t
        # 需求：脉冲 ±15×Δ
        for k, dv in (obs.get("need_pulse") or {}).items():
            if k in self.needs:
                self.needs[k] = _clip(self.needs[k] + _clip(float(dv), -2, 2) * _NEED_PULSE,
                                      self.need_baseline[k] - 2 * _SD["need"],
                                      self.need_baseline[k] + 2 * _SD["need"])
        self._clamp()
        self.updates += 1
        self.since_baseline += 1
        if self.since_baseline >= _BASELINE_EVERY:
            self.reestimate_baselines()
        return self.summary()

    def _clamp(self) -> None:
        for k in self.traits:
            self.traits[k] = _clip(self.traits[k],
                                   self.trait_baseline[k] - 2 * _SD["trait"],
                                   self.trait_baseline[k] + 2 * _SD["trait"])
        for ax in self.mood:
            self.mood[ax] = _clip(self.mood[ax], -1.0, 1.0)
        for k in self.needs:
            self.needs[k] = _clip(self.needs[k], 0.0, 100.0)

    def reestimate_baselines(self) -> None:
        """基线重估：当前值向基线做温和 EWMA 收拢（快层），特质基线基本不动。"""
        for ax in self.mood_baseline:
            self.mood_baseline[ax] = round(
                _BASELINE_EWMA * self.mood_baseline[ax] + (1 - _BASELINE_EWMA) * self.mood[ax], 4)
        for k in self.need_baseline:
            self.need_baseline[k] = round(
                _BASELINE_EWMA * self.need_baseline[k] + (1 - _BASELINE_EWMA) * self.needs[k], 2)
        self.since_baseline = 0

    # ---------- 读取面 ----------

    def mood_bucket(self) -> str:
        """情绪时间平衡采样用的分桶（PAD 愉悦度 P 维）。"""
        p = self.mood["p"]
        if p > 0.2:
            return "positive"
        if p < -0.2:
            return "negative"
        return "neutral"

    def hungriest_needs(self, top: int = 2) -> list[str]:
        """需求缺口最大的前 N 项（离设定点向下偏最多者）。"""
        gaps = {k: self.need_baseline[k] - self.needs[k] for k in self.needs}
        return sorted(gaps, key=gaps.get, reverse=True)[:top]

    def summary(self) -> dict[str, Any]:
        return {"traits": {k: round(v, 1) for k, v in self.traits.items()},
                "mood": {k: round(v, 3) for k, v in self.mood.items()},
                "emotion": self.emotion, "mood_bucket": self.mood_bucket(),
                "needs": {k: round(v, 1) for k, v in self.needs.items()},
                "updates": self.updates}

    def persona_card(self) -> str:
        """注入提示词用的人格状态卡（影响出题风格/自探索任务倾向）。"""
        emo = f"{self.emotion['label']}({self.emotion['intensity']:.0f})" if self.emotion["label"] else "平稳"
        return (f"人格状态卡：心境 愉悦{self.mood['p']:+.2f}/唤醒{self.mood['a']:+.2f}"
                f"/支配{self.mood['d']:+.2f}；当前情绪 {emo}；"
                f"需求缺口最大 {','.join(self.hungriest_needs())}；"
                f"特质 N={self.traits['neuroticism']:.0f} O={self.traits['openness']:.0f}")

    # ---------- 持久化 ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.__dict__.copy()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def append_history(self, path: Path, obs: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "obs": obs,
               "state": self.summary()}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_biodata(path: Path) -> BioData:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        bd = BioData()
        for key, val in data.items():
            if hasattr(bd, key):
                setattr(bd, key, val)
        return bd
    return BioData()
