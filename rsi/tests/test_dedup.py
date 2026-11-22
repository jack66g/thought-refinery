# -*- coding: utf-8 -*-
from rsi.core.dedup import (DedupConfig, DedupIndex, minhash_signature,
                            normalize_text, signature_jaccard)


def test_normalize():
    assert normalize_text("Hello　World！ 你好。") == normalize_text("helloworld你好")
    assert normalize_text("ＡＢＣ ｄｅｆ") == "abcdef"


def test_minhash_similarity():
    a = "先设未知数x根据题意列出方程再移项求解最后代回验证得出最终答案为十七"
    b = "先设未知数x根据题意列出方程再移项求解最后代回验证得出最终答案为十七"
    c = "完全不同的一段话讲的是别的主题比如天气历史和人文完全没有数学推导"
    from rsi.core.dedup import char_shingles
    sig_a = minhash_signature(char_shingles(a, 5))
    sig_b = minhash_signature(char_shingles(b, 5))
    sig_c = minhash_signature(char_shingles(c, 5))
    assert signature_jaccard(sig_a, sig_b) == 1.0
    assert signature_jaccard(sig_a, sig_c) < 0.3


def _make_index(tmp_path):
    return DedupIndex(base_dir=tmp_path / "dataset",
                      cfg=DedupConfig(minhash_perms=64, lsh_bands=8,
                                      jaccard_threshold=0.85))


def test_exact_dup(tmp_path):
    idx = _make_index(tmp_path)
    text = "先设未知数x，列方程2x+3=17，移项解得x=7，答案 是 17。" * 2
    v1 = idx.check(text, "math", "doc-a")
    assert v1.verdict == "unique"
    idx.add(text, "math", "doc-a")
    v2 = idx.check(text + " ", "math", "doc-b")  # 空白差异不影响规范化
    assert v2.verdict == "exact_dup" and v2.level == "L1"


def test_near_dup(tmp_path):
    idx = _make_index(tmp_path)
    base = ("先设未知数x，根据题意列出方程2x+3等于17。移项得到2x等于14，"
            "两边同除以2，解得x等于7。代回原方程验证：2*7+3等于17成立。"
            "所以最终答案是x等于7。")
    idx.add(base, "math", "doc-a")
    variant = base.replace("根据题意", "依题意").replace("最终", "")
    v = idx.check(variant, "math", "doc-b")
    assert v.verdict in ("near_dup", "ngram_polluted", "exact_dup")
    assert v.matched_id == "doc-a"


def test_unique_passes(tmp_path):
    idx = _make_index(tmp_path)
    idx.add("数学推导一：设x解方程验算得答案为七。", "math", "doc-a")
    other = "写一篇关于秋天的散文开头：落叶铺满小径，风里有桂花的香气，行人放慢了脚步。"
    v = idx.check(other, "chinese", "doc-b")
    assert v.verdict == "unique"


def test_persistence_reload(tmp_path):
    idx = _make_index(tmp_path)
    text = "先设未知数x，列方程，移项，求解，验算，得答案。" * 2
    idx.add(text, "math", "doc-a")
    idx2 = _make_index(tmp_path)  # 同一目录重开
    v = idx2.check(text, "math", "doc-x")
    assert v.verdict == "exact_dup"
