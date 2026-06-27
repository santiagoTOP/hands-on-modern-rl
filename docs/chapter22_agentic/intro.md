# 第 22 章 · Agentic RL

## 本章导读

**核心内容**

- 理解 Agentic RL 与单轮 RL 的根本差异：训练对象从 completion 变成 trajectory，rollout 必须在真实环境里执行。
- 掌握多轮交互的 MDP 形式化——联合状态、结构化动作、POMDP 视角，以及它为什么不是单轮 MDP 的简单扩展。
- 区分 ORM（结果奖励）与 PRM（过程奖励）、trajectory-level 与 step-level 信号，理解信用分配在多轮场景中为何变得尖锐。
- 建立 Agentic RL 训练系统的工程图景：异步 rollout、沙箱环境、异构轨迹长度，以及同步与异步框架的核心取舍。

**核心公式**

$$
\tau = (s_0, a_0, o_1, a_1, o_2, \ldots, a_T) \quad \text{（轨迹：模型生成 token、工具调用、环境观测的混合序列）}
$$

$$
\langle S_{\text{agent}},\ A_{\text{agent}},\ P_{\text{agent}},\ R_{\text{agent}},\ \gamma,\ O \rangle \quad \text{（POMDP：agent 只能观测部分状态）}
$$

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right] \quad \text{（多轮策略优化目标：从单步期望到轨迹期望）}
$$

$$
A_t = R(\tau) - \bar{R}(s_t) \quad \text{（step-level advantage：把轨迹奖励回拆到每一步）}
$$

**本章公式的作用**

第 22 章把强化学习的训练对象从一段回答扩展到一条完整交互轨迹。轨迹 $\tau$ 定义了多轮交互的数据形态；POMDP 六元组刻画了"模型只能看到部分状态"这一 Agentic 场景的本质；轨迹期望 $J(\theta)$ 把单轮的 $\mathbb{E}_{a \sim \pi_\theta}[r(a)]$ 推广到多步累积奖励；step-level advantage $A_t$ 则是信用分配的核心工具，决定最终 reward 如何回拆到每一步。后续小节中的 ORM/PRM、turn-level discounting、group-based advantage 都建立在这组对象之上。

第 21 章讨论的 Constitutional AI 与 RLAIF 解决的是"标注信号从哪来"——让 AI 当裁判替代人类标注。但无论是人类标注还是 AI 标注，前面所有章节的 RL 训练都共享同一个隐含假设：**模型接收一个 prompt，输出一段完整回答，奖励模型给出一个分数，策略据此更新一次**。"一问一答一打分"的骨架始终未变。

但真实的智能体不这样工作。考虑一个订机票 Agent：用户说"帮我订一张明天北京到上海最便宜的早班机票"，Agent 必须分步行动——先搜索航班、对比价格和时间、确认座位库存、调用下单 API、等待出票确认。中间任何一步出错（搜索 query 太宽、没比价直接选第一条、库存判断失误、下单参数错误），整个任务就失败。环境只在最后给出一个二元信号：出票成功或失败。

这种从"一问一答"到"多步与环境交互"的转变，正是 Agentic RL 要解决的核心问题。本章把前面建立的 RL 工具——MDP、策略梯度、GRPO、可验证奖励——系统地扩展到多轮交互场景，并补充单轮 RL 完全不涉及的工程议题：环境异步、沙箱管理、异构轨迹、长时程信用分配。

## 章节安排

| 小节                                                  | 核心问题                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ |
| [22.1 Agentic RL 总览](./overview)                    | 单轮到多轮的范式转移、智能体四组件、最简 Agent Loop、工业框架全景              |
| [22.2 多轮 RL 形式化](./formulation)                  | 多轮 MDP 怎么写？联合状态、结构化动作、POMDP、step-level 轨迹、action mask     |
| [22.3 轨迹信用分配](./credit-assignment)              | 多轮交互失败了，该怪谁？ORM vs PRM、turn-level discounting、group-based advantage |
| [22.4 工具调用 RL](./tool-use-and-trajectory)         | 训练数据从哪来？轨迹合成、工具调用策略、沙箱与异步 rollout                     |
| [22.5 Search-Augmented RL](./tool-use-agents)         | 搜索增强 Agent 怎么训？DeepSeek-Researcher、Kimi-Researcher 的训练范式         |
| [22.6 Code Interpreter RL 工业实战](./industrial-practice) | 代码 Agent 的真实训练陷阱：不稳定、长度失控、reward hacking 的工程对策         |
| [22.7 多智能体协作与 Agent Swarm](./multi-agent-swarm) | 多个 Agent 怎么协作？角色分工、通信协议、群体优势分配                          |

## 学习目标

读完本章后，你应该能够：

- 用 POMDP 六元组 $\langle S, A, P, R, \gamma, O \rangle$ 形式化一个多轮 Agent 任务，并指出它相比单轮 MDP 新引入的要素（联合状态、结构化动作、外部环境）。
- 说清 **ORM** 与 **PRM** 的取舍——为什么可验证任务适合 ORM、为什么复杂推理需要 PRM、以及 SALT/group-based 这类介于两者之间的方案在解决什么问题。
- 理解 Agentic RL 训练系统的工程骨架——rollout 与 policy update 的交替、异步训练的代价与收益、沙箱环境的设计原则——并能据此判断 OpenRLHF、verl、AReaL 等框架的适用场景。

本章会频繁用到以下概念，建议先复习：

- [GRPO 与 RLVR](../chapter18_grpo/rlvr)——"可验证奖励"是 Agentic RL 的天然奖励来源
- [PPO 与奖励模型](../chapter10_ppo/intro)——策略优化的基础框架
- [MDP 五元组](../chapter03_mdp/mdp)——形式化多轮交互的出发点

准备好后，先从 Agentic RL 的整体图景开始——[22.1 Agentic RL 总览](./overview)。
