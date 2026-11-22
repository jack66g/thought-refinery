# ThoughtRefinery 🛢️✨

**常驻式思考蒸馏插件（MCP）——把每一次对话与推理，蒸馏成可直接训练的数据。**

> An always-on MCP plugin that refines everyday conversations & reasoning traces into
> training-ready CoT / SFT / pretraining corpora — with a biometric persona state machine
> and pluggable long-term memory.

挂进 ZCode / OpenCode / DeepSeek Harness 等任意 MCP 宿主或自动化编程工具，你照常聊天、照常干活；
它在后台静默运行：**提纯思维链、积累既定事实、维护你的量化人格画像**，产出三份拿走即训的数据——

| 数据线 | 产物 | 用途 |
|---|---|---|
| **CoT 线** | `messages + <think>`（双门验证：答案验证 × 过程审计） | 推理模型 SFT |
| **预训练线** | 第一人称叙事（主观事实）/ 对话转录 / 风格保持改写 | 个人底色增量预训练 |
| **风格 SFT 线** | assistant=你的原话（角色对调） | 让模型学会像你说话 |

---

## 它有什么特别的

1. **答案对 ≠ 过程对**：内置过程审计员，定位"结论对但推理有跳步/碰巧对"的段落——可修则修复后收编，不可修降级为 DPO 负样本，修复对照保留为 LEMA 纠错对。这条红线来自 CoT 忠实性研究（见 `docs/01`）。
2. **四级查重**：精确 hash → 解答骨架等价类（防"题面不同解答模板相同"）→ MinHash+LSH（候选精确复验）→ n-gram 污染，全部纯标准库实现。
3. **仿生人格状态机**：Big5+H 特质（月级贝叶斯收缩）／PAD 心境（24h 半衰期回归基线）／Ekman-6 情绪（分钟级衰减）／六项稳态需求——全部随对话更新、防漂移三件套（均值回归+硬夹断+基线重估）。画像驱动两件事：**难度自适应出题**（锚+1 档）与**自探索任务生成**（需求缺口→任务类型），两者默认关闭。
4. **情绪时间平衡采样**：预训练语料按捕获时刻心境分桶，manifest 自带平衡采样函数——防止"状态差的那几周"污染模型底色。
5. **零侵入接线**：对宿主是 MCP 插件（一行挂载），对记忆后端 [Spatial Memory Engine](https://github.com/jack66g/spatial-memory-engine) 是纯 REST 客户端——双方源码一行不改；记忆服务不在线时索引卡进 backlog，恢复自动补写。

## 快速开始

```bash
# 1) 环境（Python >= 3.10）
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows
# pip install -r requirements.txt                    # Linux/macOS

# 2) 记忆后端（可选但推荐）：获取 SME 并安装到同一 venv
git clone https://github.com/jack66g/spatial-memory-engine.git
.venv\Scripts\pip install -e ./spatial-memory-engine --no-deps
.venv\Scripts\pip install numpy networkx python-multipart   # SME 运行依赖

# 3) LLM 密钥（OpenAI 兼容，任选：DeepSeek / GLM / ...）
setx RSI_LLM_API_KEY sk-xxx
# 默认指向 DeepSeek，改 rsi/configs/daemon.json 的 llm 节即可换服务商

# 4) 一键起停（SME:8000 + daemon:8100）
rsi\start_rsi.bat        # Windows；Linux 手动跑两个 python -m 进程
```

挂载进 MCP 宿主（与挂任何 MCP server 同款）：

```json
{"mcpServers": {"rsi": {
    "command": "<你的venv里的python>",
    "args": ["<本项目路径>/rsi/mcp_server.py"]
}}}
```

六个工具：`rsi_distill`（投喂思考，主入口）／`rsi_status`／`rsi_query`／`rsi_lessons`／`rsi_set_budget`（预算与三开关）／`rsi_next_mission`（自探索领任务，默认关）。
非 MCP 工具直接 `POST http://127.0.0.1:8100/ingest`（支持 zcode / deepseek-harness / opencode / generic 四种轨迹格式）。

## 开关一览（默认状态即安全状态）

| 开关 | 默认 | 说明 |
|---|---|---|
| 核心循环（三线分流+画像） | **开** | 随对话静默积累 |
| 自驱出题（T1 保底循环） | 关 | 难度自适应；`rsi_set_budget` 打开 |
| 自探索任务（宿主自主干活） | 关 | 日封顶 5 个 |
| LLM 预算 | 5.0/天 | 超限自动停 |

## 测试

```bash
python -m pytest rsi/tests -q    # 53 项（含 FakeLLM 全链路 + MCP stdio 冒烟，无需密钥）
```

## 目录

```
rsi/
├── daemon/      常驻引擎：队列/预算/自驱/自探索/meta 教训循环
├── core/        提纯内核：体检/四级查重/审计/修复/净化/硬软验证/三线分流
├── biodata/     仿生人格状态机（四层+防漂移）
├── explore/     自探索任务生成
├── connectors/  宿主轨迹适配器（zcode/deepseek-harness/opencode/generic）
├── llm/         OpenAI 兼容客户端（含确定性 FakeLLM 测试模式）
├── sme_wrap/    SME REST 客户端（显式降级+backlog 补写）
├── mcp_server.py  MCP 插件本体
├── prompts/v0/  12 个种子提示词（判类/价值/审计/修复/净化/出题/解题/元学习/观察者/叙事/改写/探索）
└── configs/     daemon.json + sme.rsi.json
docs/            设计文档与调研存档（查重算法规格/仿生系统/底色预训练配方，全部带文献编号）
```

## 设计依据

关键决策不是拍的，全部有文献支撑（详见 `docs/` 目录调研存档）：CoT 忠实性与过程过滤、RFT 推理等价类去重、
≤4 epoch 重复暴露上限、个人语料 1:4~1:9 回放配方、"保守清洗保底色"（C4 审计实证规范过滤会杀风格）、
Big Five/PAD/ALMA 分层情感模型、稳态 RL 需求动力学。

## 路线图

- [ ] M4：提示词血统 A/B 自进化（improver gain 曲线）
- [ ] 代码题自驱通道（出题附单元测试）
- [ ] L4 语义去重（embedding 服务就绪后）
- [ ] 24h 长跑验收与观测面板

## License

Apache-2.0 © 2026 jack66g（与记忆后端 SME 同协议）
