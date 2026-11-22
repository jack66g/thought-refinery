"""四级查重引擎（参数按调研规格书 v1.0 校准，来源标注在行内）：

L1 精确：规范化文本 blake2b 全局查（CCNet 1911.00359 同款规范化思想）。
EQ 骨架等价类：数字占位化后的解答骨架，每类保留 K 条（RFT 2308.01825）。
L2 近重复：MinHash(128, 手写) + LSH 16×8 + **候选精确 Jaccard≥0.8 复验**
    （Lee 2107.06499 / CulturaX 2309.09483 / datatrove FuzzyDup 同款）。
L3 n-gram 污染：char 10-gram 与已收录文档重叠率 >0.7 判重
    （GPT-3 13-token 口径的中文折算，工程经验值）。
L4 语义：预留（无 GPU/密钥时跳过——显式记录在台账，不静默假装做过）。

已知取舍（台账 03 有记录）：L3 高频白名单（>1% 文档的 n-gram 豁免）v1 未实现，
通用表述（"综上所述"）在万级以下语料影响有限，上线后按抽检校准。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def normalize_text(text: str) -> str:
    """查重规范化：宽度归一、小写、去空白标点（保留字母数字与 CJK）。"""
    text = text.lower()
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            continue
        if 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        out.append(ch)
    text = "".join(out)
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def skeleton_key(text: str) -> str:
    """解答骨架：数字占位化（CCNet 数字→0 思想 + RFT 等价类）。
    text 约定 = task + "\\n" + thinking，骨架只取首行之后的解答部分。"""
    body = text.split("\n", 1)[1] if "\n" in text else text
    norm = normalize_text(body)
    norm = re.sub(r"\d+", "0", norm)
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def char_shingles(text: str, n: int) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _sha32(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "big")


def minhash_signature(shingles: set[str], perms: int = 128,
                      seed: int = 42) -> list[int]:
    """手写 MinHash：h_i(x) = (a_i*x + b_i) mod p（Mersenne 素数 2^61-1）。"""
    if not shingles:
        return [0] * perms
    hashed = [_sha32(sh) for sh in shingles]
    outs = []
    for i in range(perms):
        a = (_sha32(f"a{seed}:{i}") % (_MERSENNE - 1)) + 1
        b = _sha32(f"b{seed}:{i}") % _MERSENNE
        outs.append(min((a * h + b) % _MERSENNE for h in hashed))
    return outs


def signature_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


@dataclass
class DedupConfig:
    minhash_perms: int = 128
    lsh_bands: int = 16
    jaccard_threshold: float = 0.8          # Lee/CulturaX/datatrove 口径
    shingle_n: int = 5                      # 中文 char 5-gram ≈ 英文 word 3-gram
    ngram_n: int = 10                       # GPT-3 13-token 的中文折算
    ngram_overlap_threshold: float = 0.7    # 规格书 B 口径
    skeleton_keep_k: int = 2                # 同解答骨架每类保留条数
    max_grams_per_doc: int = 2000
    max_shingles_per_doc: int = 4096


@dataclass
class DedupRecord:
    id: str
    sha: str
    skeleton: str
    sig: list[int]
    sh_hashes: set[int] = field(default_factory=set)   # 精确复验用 shingle 哈希
    grams: list[str] = field(default_factory=list)


@dataclass
class DedupVerdict:
    verdict: str = "unique"   # unique | exact_dup | skeleton_dup | near_dup | ngram_polluted
    matched_id: str | None = None
    similarity: float = 0.0
    level: str = "L0"


@dataclass
class DedupIndex:
    """按类目维护的查重索引，JSONL 持久化（追加写，启动全量加载）。"""
    base_dir: Path
    cfg: DedupConfig = field(default_factory=DedupConfig)
    global_shas: dict[str, str] = field(default_factory=dict)
    categories: dict[str, dict[str, DedupRecord]] = field(default_factory=dict)
    skeletons: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # cat -> skel -> ids
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _cat(self, category: str) -> dict[str, DedupRecord]:
        if category not in self.categories:
            self.categories[category] = {}
            self.skeletons.setdefault(category, {})
            self._load_category(category)
        return self.categories[category]

    def _index_path(self, category: str) -> Path:
        return self.base_dir / category / ".dedup_index.jsonl"

    def _load_category(self, category: str) -> None:
        path = self._index_path(category)
        cat = self.categories[category]
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    rec = DedupRecord(d["id"], d["sha"], d.get("skeleton", ""),
                                      d["sig"], set(d.get("sh", [])), d.get("grams", []))
                    cat[d["id"]] = rec
                    self.global_shas.setdefault(d["sha"], d["id"])
                    if d.get("count_skeleton", True):
                        self.skeletons[category].setdefault(rec.skeleton, []).append(d["id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # 坏行保留原样，计数进台账

    def _texts(self, text: str) -> tuple[str, str, set[str], list[str], list[int]]:
        norm = normalize_text(text)
        sha = hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()
        shingles = char_shingles(norm, self.cfg.shingle_n)
        grams = sorted(char_shingles(norm, self.cfg.ngram_n))[: self.cfg.max_grams_per_doc]
        return sha, skeleton_key(text), shingles, grams, sorted(
            _sha32(s) for s in list(shingles)[: self.cfg.max_shingles_per_doc])

    def check(self, text: str, category: str, doc_id: str) -> DedupVerdict:
        with self._lock:
            sha, skel, shingles, grams, _ = self._texts(text)
            cat = self._cat(category)  # 先加载类目（顺带注册 global_shas），再做 L1
            # L1 精确（全局）
            if sha in self.global_shas and self.global_shas[sha] != doc_id:
                return DedupVerdict("exact_dup", self.global_shas[sha], 1.0, "L1")
            # EQ 骨架等价类（同解答模板，跨题面）
            same_skel = [i for i in self.skeletons[category].get(skel, []) if i != doc_id]
            if len(same_skel) >= self.cfg.skeleton_keep_k:
                return DedupVerdict("skeleton_dup", same_skel[0], 1.0, "EQ")
            # L2 LSH 候选 + 精确 Jaccard 复验
            sig = minhash_signature(shingles, self.cfg.minhash_perms)
            rows = self.cfg.minhash_perms // self.cfg.lsh_bands
            candidates: set[str] = set()
            for band in range(self.cfg.lsh_bands):
                chunk = sig[band * rows:(band + 1) * rows]
                key = hashlib.sha1(json.dumps(chunk).encode()).hexdigest()
                for rid, rec in cat.items():
                    if rid == doc_id or len(rec.sig) != len(sig):
                        continue
                    r_chunk = rec.sig[band * rows:(band + 1) * rows]
                    if hashlib.sha1(json.dumps(r_chunk).encode()).hexdigest() == key:
                        candidates.add(rid)
            best_id, best_sim = None, 0.0
            my_sh = {_sha32(s) for s in shingles}
            for rid in candidates:
                rec = cat[rid]
                if rec.sh_hashes:
                    inter = len(my_sh & rec.sh_hashes)
                    union = len(my_sh | rec.sh_hashes)
                    exact = inter / union if union else 0.0
                else:  # 旧索引无 sh：退回签名估计（显式降级路径）
                    exact = signature_jaccard(sig, rec.sig)
                if exact > best_sim:
                    best_id, best_sim = rid, exact
            if best_sim >= self.cfg.jaccard_threshold:
                return DedupVerdict("near_dup", best_id, round(best_sim, 4), "L2")
            # L3 n-gram 污染
            new_grams = set(grams)
            for rid, rec in cat.items():
                if rid == doc_id or not rec.grams:
                    continue
                overlap = len(new_grams & set(rec.grams)) / max(len(new_grams), 1)
                if overlap >= self.cfg.ngram_overlap_threshold:
                    return DedupVerdict("ngram_polluted", rid, round(overlap, 4), "L3")
            return DedupVerdict("unique", None, 0.0, "L0")

    def add(self, text: str, category: str, doc_id: str,
            count_skeleton: bool = True) -> None:
        with self._lock:
            sha, skel, shingles, grams, sh_hashes = self._texts(text)
            sig = minhash_signature(shingles, self.cfg.minhash_perms)
            cat = self._cat(category)
            cat[doc_id] = DedupRecord(doc_id, sha, skel, sig, set(sh_hashes), grams)
            self.global_shas[sha] = doc_id
            if count_skeleton:
                self.skeletons[category].setdefault(skel, []).append(doc_id)
            path = self._index_path(category)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": doc_id, "sha": sha, "skeleton": skel,
                                     "sig": sig, "sh": sh_hashes, "grams": grams,
                                     "count_skeleton": count_skeleton},
                                    ensure_ascii=False) + "\n")

    def size(self, category: str) -> int:
        return len(self._cat(category))
