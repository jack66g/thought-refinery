# 01 调研笔记：CoT 提取 × RSI 循环（2026-11-22）

> 本文档是四路调研的**提纯摘录**，只保留与本项目直接相关的内容，供随时翻查。
> 所有 arXiv 编号均已核查（2026 年新论文用前建议再点开确认一眼）。
> 配套文档：`02_RSI_V1_设计_CoT提纯循环.md`（怎么用这些结论）。

---

## A. CoT 提取：直接要用的方法

### A1. 过度思考的事实基础

- **"Do NOT Think That Much for 2+3=?"**（[2412.21187](https://arxiv.org/abs/2412.21187)）：
  - 按-anchor 切 solution：每个"给出明确答案的完整段落"算一个 solution。
  - **92%+ 答对案例中第一个 solution 已正确**，首答后的思考对准确率贡献极小。
  - 极端案例：901 token 中首个正确 solution 只占 39（有效占比 4.3%）；QwQ 答 2+3 生成 13 个 solution，后 12 个全浪费。
  - 修剪变体三档：**FCS**（只留首个正确 solution）/ **FCS+Reflection**（再留一个正确 solution 保反思能力）/ **GDS**（贪心保策略多样）。
- 综述：Stop Overthinking（[2503.16419](https://arxiv.org/abs/2503.16419)）。

### A2. 关键实证结论（直接影响管线设计）

| 结论 | 来源 | 对管线的含义 |
|---|---|---|
| 把蒸馏数据里的 "Wait"/"Hmm" 反思 token 删掉再训，基准**不降反升** | EMNLP 2025 Findings（[链接](https://aclanthology.org/2025.findings-emnlp.394.pdf)） | 这些多半是教师风格残留，可删 |
| 是否保留回溯**取决于任务结构**：搜索型难题受益，线性可解任务受损 | [2504.07052](https://arxiv.org/abs/2504.07052) | 按题型分策略，不一刀切 |
| **majority voting 是整体最佳过滤策略**（该领域消融） | OpenThoughts / [2506.04178](https://arxiv.org/abs/2506.04178) | 采样 n 条 → 群体答案过半才收 |
| 去重维度应是**推理路径等价类**而非题面 | RFT [2308.01825](https://arxiv.org/abs/2308.01825) | 同题同路径只留一条，防表层过拟合 |
| **答案对 ≠ 过程对**；CoT 不一定是真实计算的记录 | Anthropic [2307.13702](https://arxiv.org/abs/2307.13702) / [2505.05410](https://arxiv.org/abs/2505.05410) | 过程级审计不可省 |
| 只按答案过滤的自训练 → reward hacking → **突然完全崩溃** | [2505.21444](https://arxiv.org/abs/2505.21444) | 红线：过程错的轨迹绝不直接当正样本 |
| 错误数据的合法形态只有两种：**DPO 负样本** / **LEMA 式结构化纠错对** | LEMA [2310.20689](https://arxiv.org/abs/2310.20689) 等 | 见 A4 |
| s1K 仅 53.6% 轨迹完全正确仍训出强模型 | s1 [2501.19393](https://arxiv.org/abs/2501.19393) | 小规模高质量选题下轨迹含小错不致命；**规模大了必须过滤** |
| SelFee：纯 SFT 可注入自我纠正行为（数据格式：回答→自反馈→修订→终止信号链） | [2305.15923](https://arxiv.org/abs/2305.15923) | 纠正行为可以靠数据格式教 |
| 无外部反馈的自我纠错平均**降低**性能；真自我纠正需要多轮 RL（SCoRe） | [2310.01798](https://arxiv.org/abs/2310.01798) / [2409.12917](https://arxiv.org/abs/2409.12917) | v1 不指望模型自己纠，靠管线纠 |

### A3. 过滤与修剪工具箱

- **题目侧**：只留可验证题（sympy 可解析/单元测试）；难度过滤——小模型不用 CoT 就能答对的删（Qwen3/GLM-4.5 官方做法；GLM-4.5 剔最易 50% 换来 +2-4%）。
- **答案级**：确定性验证器 > 多数投票 > 模型自评（模型只能做粗筛，**绝不当最终裁判**）。
- **步骤级**：
  - 隐式 PRM（[2412.01981](https://arxiv.org/abs/2412.01981)）：只用 outcome 好/坏样本对即可训步骤打分器，免 rollout；
  - OmegaPRM（[2406.06592](https://arxiv.org/abs/2406.06592)）：MCTS+二分定位**第一个错误步骤**；
  - Math-Shepherd（[2312.08935](https://arxiv.org/abs/2312.08935)）：对步骤前缀采样 k 次补全、以最终答对率当步骤价值（贵）。
  - 共识：PRM 当**过滤器和首错定位器**用，别当 RL 奖励源（易被 hack，R1 因此弃用）。
- **反向验证（round-trip）**：FOBAR（[2308.07758](https://arxiv.org/abs/2308.07758)）——遮住题中变量、把候选答案当已知**反推**，一致才算对。比正向自查难被骗，适合无 sympy 场景。
- **工程体检**（便宜，先跑）：n-gram 重复率、截断检测、语言混杂、循环结构。

### A4. 错误→纠正数据的三种归宿（本项目核心）

1. **修复后收编**：定位首错 → 段级改写 → **重放复验**（修复后轨迹重推答案必须仍对）→ 干净 SFT。
2. **LEMA 结构化纠错对**（[2310.20689](https://arxiv.org/abs/2310.20689)）："错误推理+纠正+正确解"三元组，比纯正确轨迹训练效果更好；建议占最终数据 5-10%。
3. **降级为负样本/弃用**：循环论证、"碰巧对"（结论与推理无关）、修复失败 → DPO rejected 素材（v1 先存档不训练）。

### A5. 官方 reasoning 数据清洗规则对照（抄作业清单）

| 模型 | 规则 |
|---|---|
| DeepSeek-R1（[2501.12948](https://arxiv.org/abs/2501.12948)） | sympy 验答案；滤重复/语言混杂/超长段落/代码块；RS 只留答对 |
| Qwen3（[2505.09388](https://arxiv.org/abs/2505.09388)） | 响应级六滤：答案错误 / 大量重复 / 明显猜测无推理 / **思考与总结不一致** / 语言混杂 / 疑似评测集泄漏；题级滤"无需 CoT 即可答对" |
| GLM-4.5（[2508.06471](https://arxiv.org/abs/2508.06471)） | 去重/过短/截断/格式无效 → 客观验证 → RM 过滤 → 工具轨迹须达终态；剔最易 50% prompt |

### A6. 数据格式（已验证的模板）

- **SFT**：messages 格式，assistant 内嵌 `<think>…</think>`；多轮时非思考轮保留空 `<think></think>`（Qwen3 做法，保格式一致）。
- **DPO 对构造三模板**：
  1. 最短正确（chosen）vs 同题最长（rejected）——Do NOT Think；
  2. 最短正确 vs 正确但 ≥1.5 倍长 + 错误长回答——Kimi k1.5（[2501.12599](https://arxiv.org/abs/2501.12599)，实测 token -45% 准确率不掉）；
  3. 同预算达标 vs 超长——L1/LCPO。
- **GRPO 组内优势复用**：同题 rollout G 条，advantage=(r−mean)/std，adv>0 且最短的轨迹沉淀为 SFT 集（GLM-4.5 / Qwen3 官方思想）。

---

## B. RSI 循环：借鉴的机制清单

### B1. 度量语言（来自 [lobehub/awesome-rsi](https://github.com/lobehub/awesome-rsi)）

- **task gain**：这道题变好了（最弱）
- **retained gain**：别的题没变坏
- **improver gain**：**改进器本身在变强**（唯一算真 RSI 的）
- 术语：self-refinement（不持久）< persistent SI（持久改 prompt/记忆/代码）< RSI（改进机制本身被改进）。本项目 v1 目标 = persistent SI，v2 目标 = improver gain。

### B2. OpenEvolve（[algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve)，7.8k★）

- 用户契约：程序里标 `# EVOLVE-BLOCK-START/END` + 写 `evaluate(program_path) -> dict[str,float]`（多指标，越大越好），其余自动。
- 循环：岛模型（2-4 个种群隔离）+ MAP-Elites 保多样性 + 双重选择（父代按性能、灵感样本可跨岛）+ artifact 反馈（把上代报错/评估输出喂回 prompt）+ 周期迁移。
- 全组件可 seed，JSON checkpoint 血统可复现。**16GB 机器纯 API 可跑，本项目阶段二的骨架。**

### B3. ShinkaEvolve（[2509.19349](https://arxiv.org/abs/2509.19349)，Sakana，ICLR 2026）

样本效率路线（150 次评估超 AlphaEvolve 原解）。三个可搬机制：
1. **宽松父代选择**：幂律采样 + 按"性能×新颖度/(1+已有子代数)"加权——好但被榨干的自动让位；
2. **新颖度拒绝采样**：embedding 余弦 >0.95 直接拒绝重采样（省评估钱）；
3. **meta-scratchpad**：每 10 代 meta-agent 总结评估史 → ≤5 条可执行建议附进突变 prompt（它自己进化出了 7 调用 scaffold 超过人工设计）。
另有 prompt 自进化 archive（内置 PromptBreeder）+ API 预算封顶。

### B4. AlphaEvolve（[2506.13131](https://arxiv.org/abs/2506.13131)，未开源，机制可抄）

- **评测级联**：测试由易到难，前一关不过不进下一关（先便宜过滤）；
- **模型分工**：廉模型高频出候选 + 贵模型低频做突破；
- meta-prompt 在独立数据库里共同进化。

### B5. Darwin Gödel Machine（[2505.22954](https://arxiv.org/abs/2505.22954)）——防退化教科书

- **archive 血统树只增不减 = 天然回滚**：父代按性能加权抽样，允许回到任意祖先重启支线；
- 每次只评锚点问题的**随机子集**（防过拟合固定测试集）；
- held-out 与 evaluator 对被进化对象不可见（防偷改评测）；
- 基座权重不动，只进化 scaffold（限制爆炸半径）。SWE-bench 20%→50%。

### B6. GEPA / DSPy（prompt 进化，免训练）

- GEPA（[2507.19457](https://arxiv.org/abs/2507.19457)，已并入 [DSPy](https://github.com/stanfordnlp/dspy)）：LLM proposer **读运行轨迹反思**→变异 prompt；候选池只保留"分数×长度"**帕累托前沿**；比 GRPO 平均高 6% 且省 35 倍 rollout。**本项目中"提示词自主进化"的蓝本。**

### B7. 数据闭环家族（SFT 层的祖先们）

- STaR（[2203.14465](https://arxiv.org/abs/2203.14465)）：采样→留答对→SFT→迭代；
- ReST-EM（[2308.08998](https://arxiv.org/abs/2308.08998)）：Grow（采样建库）与 Improve（训练）解耦，训练期数据冻结防漂移；
- Self-Rewarding（[2401.10020](https://arxiv.org/abs/2401.10020)）：多候选自评造偏好对→DPO→迭代。
- 本项目 v1 = 上述闭环的"数据生产半边"（只 Grow 不 Improve，不训练）。

### B8. 明确不碰的（记录在案）

SEAL（自编辑权重）、Absolute Zero（自博弈多卡 RL）、DGM 完整版（SWE-bench 基建）——机制已吸收进设计，实现跳过。

---

## C. 记忆系统速查（SME 的定位）

- 三层记忆共识：**参数记忆**（慢、稳、难更新）/ **上下文**（快、易失）/ **外部检索记忆**（显式、可增删改、可审计）→ 冻结参数 + 外部记忆承接全部易变信息 + 异步巩固管道。
- CLS 互补学习系统：海马（快/稀疏/易变）↔ 新皮层（慢/稠密/稳定），睡眠重放+交错巩固 = sleep-time compute（[2504.13171](https://arxiv.org/abs/2504.13171)）。
- 再巩固：记忆被提取后才可改写 → 检索触发的 UPDATE（A-Mem / Mem0 四操作）。
- 遗忘是特性：记忆库要主动做减法，否则检索污染（裸 RAG 越用越糟的根源）。
- **SME 在本项目中的角色 = improver 的海马体**：题目库索引、错误模式库、教训库（会取代更新，factversion）、提示词血统库、自画像统计（profile）。原始 trace 走文件系统，SME 只存索引卡与经验——大对象不进记忆库。

---

## D. 来源总表

**CoT**：2412.21187 / 2503.16419 / 2501.19393 (s1) / 2502.03387 (LIMO) / 2501.12599 (Kimi k1.5) / 2501.12948 (R1) / 2505.09388 (Qwen3) / 2508.06471 (GLM-4.5) / 2307.13702 / 2505.05410 / 2310.01798 / 2409.12917 (SCoRe) / 2310.20689 (LEMA) / 2505.21444 / 2305.20050 (PRM800K) / 2312.08935 (Math-Shepherd) / 2406.06592 (OmegaPRM) / 2412.01981 (隐式PRM) / 2308.07758 (FOBAR) / 2308.01825 (RFT) / 2312.06585 (ReSTEM) / 2506.04178 (OpenThoughts) / 2504.07052
**RSI**：2506.13131 (AlphaEvolve) / [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) / 2509.19349 (ShinkaEvolve) / [ShinkaEvolve repo](https://github.com/SakanaAI/ShinkaEvolve) / 2505.22954 (DGM) / 2410.04444 (Gödel Agent, 正确仓库 [Arvid-pku/Godel_Agent](https://github.com/Arvid-pku/Godel_Agent)) / 2507.19457 (GEPA) / [DSPy](https://github.com/stanfordnlp/dspy) / 2203.14465 (STaR) / 2308.08998 (ReST) / 2401.10020 / [awesome-rsi](https://github.com/lobehub/awesome-rsi)
**记忆**：2310.08560 (MemGPT) / 2504.19413 (Mem0) / 2504.13171 (sleep-time) / 2501.13956 (Zep) / 2502.12110 (A-Mem) / 2405.14831+2502.14802 (HippoRAG) / 2507.07957 (MIRIX) / 2507.03724 (MemOS) / McClelland 1995 CLS / 1612.00796 (EWC)
**工具链**：[open-r1](https://github.com/huggingface/open-r1) / [open-thoughts](https://github.com/open-thoughts/open-thoughts) / [TRL](https://github.com/huggingface/trl) / [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
