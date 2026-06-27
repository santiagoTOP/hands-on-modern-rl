# 22.2 多轮 RL 形式化

[22.1 总览](./overview) 用订机票的例子说明了 Agentic RL 与单轮 RL 的四个根本差异。本节把这些差异写成精确的数学对象——从单轮 MDP 出发，逐步扩展到多轮 MDP，再到 POMDP，最后讨论工业实现中的两个关键技术细节：**step-level 轨迹结构**和 **action mask**。

形式化本身不复杂。Agentic RL 的真正难点，很多时候不在 RL 公式本身，而在"如何让 RL 公式能作用于真实的 agent loop"——状态怎么定义、动作怎么结构化、token 流怎么区分模型生成还是环境返回。这些工程决策都建立在本节的形式化之上。

## 单轮 RL 的 MDP

前面章节的 GRPO 本质是一个**退化的 MDP**。模型接收 prompt，自回归地生成 token，最后由奖励模型或 verifier 给出一个标量 reward。它的 MDP 组件可以这样定义：

- **状态** $s$：当前 token context（prompt + 已生成 token）
- **动作** $a$：下一个 token
- **转移** $P$：确定性 append——把选中的 token 加到 context
- **奖励** $r$：整条 rollout 结束后给一次（通常由 reward model 或 verifier 给出）
- **轨迹** $\tau$：完整的 token 序列

每一步的动作是从 LLM 的 next-token 分布中采样的——每个 token 是一个独立的动作。选完一个 token 后，它被加到当前状态中，模型再预测下一个 token，如此往复直到模型输出 `<eos>` 终止符。

优化目标是让单轮输出的期望奖励最大：

$$
\mathbb{E}_{a \sim \pi_\theta}[r(a)] \quad \longrightarrow \quad \max_\theta
$$

## 多轮 RL 的 MDP

把单轮的四个组件分别扩展，就得到多轮 RL 的 MDP。**核心变化是：模型不再只是闭门生成文本，而是在每一步可以调用工具、接收环境反馈**。

### 状态：联合状态

单轮 RL 的状态就是 token context。多轮 RL 的状态是一个**联合状态**，包含两部分：

$$
s_t = (c,\ x_{1:t},\ e_t)
$$

- $c$：任务指令（system prompt + user query）
- $x_{1:t}$：到目前为止的所有 token（包括模型生成的文本和工具返回的观测）
- $e_t$：环境当前状态（如航班数据库快照、代码执行结果、文件系统状态）

模型能"看到"的只是 token context $x_{1:t}$，但 MDP 的完整状态必须包含 $e_t$——因为同样的 token 序列，在不同的环境状态下，转移结果可能完全不同。

### 动作：文本 + 结构化工具调用

单轮 RL 的动作就是 token。多轮 RL 的动作扩展为：

$$
A = A_{\text{text}} \cup A_{\text{action}}
$$

- $A_{\text{text}}$：普通文本 token
- $A_{\text{action}}$：结构化工具调用（如 `<tool_call>{"name":"search","args":...}</tool_call>`）

在 token 层面，所有动作最终都是 token 序列；但在语义层面，文本 token 和工具调用 token 有本质区别——前者只更新 context，后者还会触发环境状态变化。

### 转移：动态、可能非确定性

单轮 RL 的转移是确定性 append。多轮 RL 的转移更复杂：

$$
P(s_{t+1} \mid s_t, a_t) = \begin{cases}
\text{确定性 append} & \text{if } a_t \in A_{\text{text}} \\
\text{环境驱动 + 可能随机} & \text{if } a_t \in A_{\text{action}}
\end{cases}
$$

工具调用触发环境状态变化，并返回观测 $o_{t+1}$。这个返回可能：
- **随机**（搜索引擎结果会变）
- **失败**（API 报错、代码语法错误）
- **依赖隐藏状态**（数据库当前内容）

这就是为什么 Agentic rollout 的 wall-clock 时间方差极大——同一条策略跑两次，可能一次 3 秒、一次 30 秒。

### 奖励：终点 + 中间

单轮 RL 的 reward 只在 rollout 结束时给一次。多轮 RL 可以两种都支持：

- **Outcome Reward**（终点奖励）：$r_T$，整条轨迹完成时给
- **Process Reward**（过程奖励）：$r_t$ for $t < T$，每一步独立给

这种灵活性是必要的——长程任务（10+ 步）如果只给终点 reward，信号太稀疏，模型学不动。Process reward 提供密集的中间信号，但代价是标注成本高（详见 [22.3 轨迹信用分配](./credit-assignment)）。

### 多轮 RL 的优化目标

把单轮目标 $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$ 推广到多步累积：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right] \quad \longrightarrow \quad \max_\theta
$$

差别在于：期望是对**整条轨迹** $\tau$ 取的，而不是单个动作 $a$。

## 单轮 vs 多轮：一张表对比

|              | 单轮 RL（GRPO）                        | 多轮 Agentic RL                                                  |
| ------------ | -------------------------------------- | ---------------------------------------------------------------- |
| **状态**     | 单个 prompt，episode 立即结束          | 联合状态 $(c, x_{1:t}, e_t)$，随交互动态演化                     |
| **动作**     | 纯文本 token                           | 文本 + 结构化工具调用                                            |
| **转移**     | 确定性 append                          | 动态转移，环境可能非确定性                                       |
| **奖励**     | 单步标量 $r(a)$                        | 步级或终点，可能是稀疏的任务完成信号                             |
| **优化目标** | $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$ | $\mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t R(s_t, a_t)]$ |
| **rollout 周期** | 几百毫秒                           | 秒到分钟级（受环境延迟主导）                                     |

## POMDP：为什么 Agentic RL 通常是部分可观测

仔细看多轮 MDP 的状态定义：完整状态是 $(c, x_{1:t}, e_t)$，但模型实际能看到的只是 $(c, x_{1:t})$——环境状态 $e_t$ 是隐藏的，模型只能通过工具调用的返回观测 $o_t = O(s_t)$ 间接推断。

这正好对应**部分可观测马尔可夫决策过程（POMDP）**：

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle
$$

其中 $O$ 是观测函数，$o_t = O(s_t)$——模型看到的是观测，不是完整状态。

POMDP 视角的实践意义：

- **Belief state**：模型必须基于历史观测推断当前环境状态。这就是为什么 instruction 里要写"如果搜索失败就重试"——模型需要从观测序列中识别出"环境当前不可靠"这种隐藏状态。
- **信息收集动作**：某些工具调用本身不是为了改变环境，而是为了获取信息（如查询当前时间、读取文件内容）。这类动作在 POMDP 中有特殊价值——它们降低了 belief state 的不确定性。
- **Grounding 的形式化解释**：模型把输出锚定到真实世界状态，本质上是在 POMDP 中维护一个准确的 belief。

## Step-Level 轨迹结构

理论上轨迹 $\tau = (s_0, a_0, s_1, a_1, \ldots, a_T)$ 就够了。但工业实现中，**如何存储轨迹**直接影响训练稳定性和工程效率。

### Flat Token Sequence 的问题

最简单的存储方式：把整条轨迹拍平成一个 token 序列。优点是实现简单、与单轮 RL 框架兼容。但有两个问题：

1. **Step 边界隐式**：哪几个 token 属于"第 3 轮的模型输出"、哪几个属于"第 3 轮的工具返回"——全靠特殊 token 来切分，错误处理容易出 bug。
2. **Retokenization drift**：rollout 时模型在 token 空间生成，但存储时常常解析成 message list，训练时再把 message 重新 tokenize。tokenization 不是可逆操作——同一个文本可能对应不同的 token 序列，导致训练数据和 rollout 时不一致。

### Agent-R1 的 Step-Level 表示

Agent-R1（Cheng et al., 2025）提出把轨迹存储为**结构化的 step-level 记录**，每一步显式保存：

```python
@dataclass
class Step:
    state_before: str          # 该步开始时的 context
    action_tokens: List[int]   # 模型生成的原始 token id（不重新 tokenize）
    observation: str           # 工具返回的观测（如果有）
    reward: float              # 该步的 reward
    is_terminal: bool          # 是否是最后一步
```

这种表示有三个好处：

- **精确的 step 边界**：训练时可以精确地知道哪些 token 属于哪一步。
- **无 retokenization drift**：直接存储 token id，训练时直接用，不做文本-token 往返转换。
- **灵活的 context 管理**：可以选择 append-only（把所有历史都喂给模型），也可以选择 sliding-window、summarization 等策略——因为 step-level 结构允许任意组合、过滤、摘要。

### Context 管理策略

Append-only 是最简单的 context 策略，但长程任务下会撑爆上下文窗口。常见的替代：

- **Sliding-window**：只保留最近 $k$ 步的观测。
- **LLM summarization**：用另一个 LLM 把早期观测总结成简短摘要。
- **Selective retention**：根据 reward 信号保留"重要"步骤，丢弃冗余步骤。

Agent-R1 的实验显示，**sliding-window 在 GSM8K 上比 append-only 表现更好**——"less is more"，模型不需要看到所有历史也能做出好决策。

## Action Mask：区分模型生成与环境返回

多轮轨迹的 token 流混合了两类内容：

- **Agent-generated tokens**：模型生成的文本和工具调用
- **Environment-generated tokens**：工具返回的观测、prompt、special tokens

**只有 agent-generated tokens 应该参与策略梯度更新**——环境返回的 token 不是模型"选择"的，对它们求梯度没有意义。这就是 **action mask**：

$$
L_{\text{policy}}(\theta) = \sum_{t,i} m_{t,i} \cdot \log \pi_\theta(a_{t,i} \mid s_t) \cdot A_t
$$

其中 $m_{t,i} \in \{0, 1\}$ 是 action mask：1 表示该 token 是模型生成的，0 表示是环境提供的。

```python
# Action mask 示例
# 1 = 模型生成的 token（参与梯度）
# 0 = prompt / 工具返回 / padding（不参与梯度）
action_mask = torch.tensor([
    [0, 0, 0, 1, 1, 1, 0, 0, 1, 1],   # 第 1 轮：prompt(0) + 生成(1) + 工具返回(0) + 生成(1)
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],   # 第 2 轮
])
```

Action mask 是几乎所有 Agentic RL 框架共享的实现细节。论文 [4]（Agent-R1）发现：**完全排除非 agent token 不是最优**——可以对环境 token 施加 SFT loss（学习预测环境行为），相当于边学策略边学世界模型。这条路线被 Echo、PaW 等后续工作进一步发展。

## 工业实现的两条原则

把上面的形式化收敛成两条工业实现原则：

**1. 结构化存储优于线性存储。** Step-level 轨迹结构（Agent-R1 风格）避免了 retokenization drift，并为 context 管理、process reward、action mask 提供干净的接入点。

**2. Action mask 是必须的，不是可选的。** 没有 action mask 的训练相当于让模型"学习预测环境返回"——这会污染策略梯度，导致训练不稳定。

这两条原则在所有主流框架（OpenRLHF、verl、AgentGym-RL、Agent-R1、AgentRL）中都是一等公民。它们的代价是工程复杂度——轨迹不再是 token 列表，而是嵌套的结构化数据；loss 计算需要逐 token 应用 mask。但这些代价是值得的。

## 本节总结

本节把 Agentic RL 的形式化骨架搭起来了：

- **多轮 MDP** = 单轮 MDP 的四组件分别扩展：联合状态、结构化动作、非确定性转移、终点 + 过程奖励。
- **POMDP** 是 Agentic RL 的自然视角——模型只能看到观测，不能看到完整环境状态。Belief state 和信息收集动作都有 POMDP 解释。
- **Step-level 轨迹结构** 和 **action mask** 是工业实现的两个关键细节——前者避免 retokenization drift，后者保证策略梯度的正确性。

形式化只是起点。接下来要回答的核心问题是：**一条轨迹最终失败了，reward 怎么回拆到每一步？** 这就是信用分配——[22.3 轨迹信用分配](./credit-assignment)。
