# -*- coding: utf-8 -*-
from rsi.core.segmenter import fcs_truncate, find_conclusion_segment, segment


def test_segment_by_size():
    text = ("这是第一句话。" * 5) + ("这是第二段话！" * 5) + "最终答案是42。"
    segs = segment(text, max_chars=50)
    assert len(segs) >= 2
    assert all(len(s) >= 10 for s in segs)


def test_conclusion_anchor_and_fcs():
    segs = ["推导第一步成立。推导第二步成立。", "中间继续推导无误。",
            "所以答案是 42。", "再检查一遍，答案仍是42，验证完毕。",
            "多余的第一段。多多余的第二段。"]
    assert find_conclusion_segment(segs) == 3
    cut = fcs_truncate(segs, keep_reflection=True)
    assert len(cut) == 4  # 首答案段 + 1 个反思段
    cut2 = fcs_truncate(segs, keep_reflection=False)
    assert len(cut2) == 3
