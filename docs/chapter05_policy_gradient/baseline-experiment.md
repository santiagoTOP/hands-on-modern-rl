# 5.3 动手：Baseline 平衡小车挑战

> **本节目标**：用 `CartPole-v1` 对比原始 REINFORCE 和带 Value Baseline 的 REINFORCE，观察基线如何让策略梯度训练更快、更稳。

> **本节代码**：[reinforce_with_baseline.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_with_baseline.py) · [reinforce_cartpole.py](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/reinforce_cartpole.py) · [requirements.txt](https://github.com/walkinglabs/hands-on-modern-rl/blob/main/code/chapter05_policy_gradient/requirements.txt)

前两节已经说明了 REINFORCE 的基本思想：
如果一段轨迹得到高回报，
就提高这段轨迹中动作的概率。
这个思想很直接，
但它有一个明显缺点：
同一个策略在不同回合里可能得到很不一样的回报，
于是梯度更新会被运气牵着走。

本节不再用无状态赌博机作为主实验。
赌博机适合解释公式，
但它太抽象，
很难看出“策略到底学会了什么”。
我们换成 `CartPole-v1`：
小车可以向左或向右推，
目标是让杆子尽量久地保持竖直。
这仍然是离散动作任务，
但它有清楚的画面和失败方式：
推晚了，杆子会倒；
推反了，小车会把杆子越带越偏。

## 5.3.1 为什么 CartPole 更适合看 Baseline

CartPole 的状态有 4 个数字：
小车位置、小车速度、杆子角度和杆子角速度。
动作只有两个：
向左推或向右推。
每坚持一个时间步，环境给 `+1` 奖励；
如果杆子倒得太厉害，或者小车离开边界，episode 结束。
`CartPole-v1` 的最高回合长度是 `500`，
所以回合回报也可以直接理解为“杆子立住了多少步”。

这个任务刚好暴露 REINFORCE 的高方差问题。
在训练早期，
策略可能只是偶然多坚持了几十步。
原始 REINFORCE 会把这一整段轨迹中的动作都当作“好动作”来强化，
即使其中有些动作只是碰巧没有立刻造成失败。
下一回合若开局扰动不同，
同一个策略又可能很快倒下。
于是训练曲线会出现明显抖动。

Baseline 要解决的不是“让公式方向改变”，
而是把学习信号从

$$G_t$$

换成

$$A_t = G_t - V(s_t).$$

这里，$G_t$ 是从时间步 $t$ 开始的折扣累计回报；
$V(s_t)$ 是价值网络对当前状态平均回报的估计。
减掉 $V(s_t)$ 之后，
策略网络不再只问“这一步之后拿了多少分”，
而是问“这一步之后比原本预期好了多少”。
如果实际结果比预期好，
就强化这个动作；
如果实际结果比预期差，
就降低这个动作的概率。

这就是 Baseline 在 CartPole 中最直观的作用：
不是让小车多一个动作，
而是让它少被偶然的好坏回合误导。

## 5.3.2 运行对比实验

先安装依赖：

```bash
pip install -r code/chapter05_policy_gradient/requirements.txt
```

然后运行对比实验：

```bash
python code/chapter05_policy_gradient/reinforce_with_baseline.py
```

这个脚本会训练两个策略：

| 实验 | 更新信号 | 额外网络 | 直观含义 |
| ---- | -------- | -------- | -------- |
| Vanilla REINFORCE | `G_t` | 无 | 只看这一回合之后实际拿了多少分 |
| REINFORCE + Baseline | `G_t - V(s_t)` | Value Network | 看实际结果比当前状态的平均预期好多少 |

两个版本都使用同一个 CartPole 环境和同一种策略网络。
区别只在更新权重：
原始版本用完整回报 $G_t$；
Baseline 版本先训练一个价值网络估计 $V(s_t)$，
再用优势 $G_t - V(s_t)$ 更新策略。

脚本结束后会生成两张图：

| 输出文件 | 说明 |
| -------- | ---- |
| `output/reinforce_baseline_reward_comparison.png` | 两种方法的回合奖励曲线 |
| `output/reinforce_baseline_variance_comparison.png` | 两种方法的梯度估计方差曲线 |

本节讲义中的图像就是由这个脚本导出的。

## 5.3.3 看奖励曲线

先看最直接的结果：小车能立住多久。

![CartPole 上原始 REINFORCE 与 REINFORCE + Baseline 的奖励曲线对比。Baseline 版本更早接近 500 步上限，原始版本学习更慢且波动更明显。](./images/reinforce-baseline-cartpole-reward.png)

图中浅色线是单个 episode 的原始回报，
深色线是滑动平均。
单个 episode 的回报会剧烈跳动，
这是策略梯度任务中很正常的现象：
同一个策略在不同初始状态下可能撑很久，
也可能很快失败。
因此，更应该看滑动平均趋势。

这次运行中，
原始 REINFORCE 的最后 50 回合平均回报约为 `276.5`。
它确实在学习，
但学习过程比较慢，
中途还有明显回落。
加入 Value Baseline 后，
最后 50 回合平均回报约为 `484.1`，
已经非常接近 CartPole 的 `500` 步上限。

这个差异说明：
Baseline 不是一个装饰性的数学项。
在同一个任务中，
它能让策略更快进入“基本能立住杆子”的区域，
并减少训练后期突然退步的概率。

## 5.3.4 看方差曲线

奖励曲线回答“策略表现是否变好”。
方差曲线回答另一个问题：
为什么 Baseline 会让训练更稳？

![CartPole 上原始 REINFORCE 与 REINFORCE + Baseline 的梯度估计方差对比。Baseline 把回报变成优势后，梯度信号更集中。](./images/reinforce-baseline-cartpole-variance.png)

这张图画的是滑动窗口中的梯度估计方差。
数值越大，
说明不同 episode 给出的更新方向差异越大；
数值越小，
说明策略每次更新更一致。

在这次运行中，
原始 REINFORCE 的梯度估计方差约为 `115.34`，
Baseline 版本约为 `31.16`。
也就是说，
Baseline 把方差降到了原来的约 `27%`。
这和奖励曲线中的现象对应起来：
更新信号更稳，
策略就更容易持续朝着“让杆子站住”的方向移动。

## 5.3.5 代码里到底改了什么

原始 REINFORCE 的核心更新是：

```python
returns_t = torch.FloatTensor(returns)
log_probs = torch.log(action_probs + 1e-8)
loss = -(log_probs * returns_t).mean()
```

这里的 `returns_t` 就是 $G_t$。
如果某一回合刚好撑了很久，
这段轨迹里的所有动作都会被较大权重强化。
这并不总是错，
但它会把很多“碰巧没有出事”的动作也一起强化。

加入 Baseline 后，
脚本多了一个价值网络：

```python
values = value_net(states_t)
value_loss = nn.MSELoss()(values, returns_t)
```

价值网络学习的是：
从状态 $s_t$ 出发，
通常能拿多少分。
然后策略网络不再直接使用 $G_t$，
而是使用优势：

```python
with torch.no_grad():
    values_pred = value_net(states_t)

advantages = returns_t - values_pred
policy_loss = -(log_probs * advantages).mean()
```

这几行代码就是 Baseline 的核心。
如果 `advantages` 为正，
说明这一步之后比预期更好，
对应动作应该更常出现；
如果 `advantages` 为负，
说明这一步之后比预期更差，
对应动作应该减少。

注意，Baseline 不依赖当前动作本身，
因此不会改变策略梯度的期望方向。
它改变的是估计的噪声大小。
这也是为什么它叫“降方差”，
而不是“改目标”。

## 5.3.6 回到画面中理解

想象小车已经把杆子扶到接近竖直的位置。
如果它本来就能从这个状态继续坚持很久，
那么再多坚持几步并不一定说明刚才那个动作特别神奇；
这只是一个好状态本来就应该有的结果。
此时 $V(s_t)$ 会比较高，
减掉它以后，
优势不会被夸大。

反过来，
如果杆子已经明显倾斜，
小车却通过一个正确动作把局面救回来，
实际回报可能明显超过价值网络的预期。
这时 $G_t - V(s_t)$ 为正，
策略会更明确地强化这个补救动作。

这就是 Baseline 比“只看总分”更细的地方：
它让策略知道，
同样是拿到 100 分，
在危险状态下拿到 100 分，
和在容易状态下拿到 100 分，
含义并不一样。

## 5.3.7 常见误读

**误读一：Baseline 会让奖励变大。**
Baseline 不改环境奖励。
CartPole 每一步仍然只给 `+1`。
它改变的是训练时如何解释这些奖励。

**误读二：Baseline 越大越好。**
如果基线估计很差，
优势也会很吵。
这里使用价值网络学习 $V(s)$，
是因为状态不同，合理的平均回报也不同。
一个固定常数基线只能处理很简单的无状态问题。

**误读三：有 Baseline 就是 Actor-Critic。**
本节仍然是 REINFORCE with Baseline。
它要等一个完整 episode 结束，
用 Monte Carlo 回报 $G_t$ 更新。
下一章的 Actor-Critic 会进一步用 TD 目标替代完整回报，
做到每一步都可以更新。

## 小结

- CartPole 比赌博机更适合展示 Baseline 的作用，因为它有状态、有失败形态，也能通过回合长度直观看出策略好坏。
- 原始 REINFORCE 使用 $G_t$ 更新策略，容易被单个 episode 的运气误导。
- Value Baseline 学习 $V(s_t)$，把更新信号从 $G_t$ 改成 $G_t - V(s_t)$。
- Baseline 不改变策略梯度的期望方向，但能显著降低方差，使训练更稳定。
- 本节中的 Baseline 已经出现了 Critic 的影子；下一章会把它发展成真正的 Actor-Critic。

## 练习

1. 把 `num_episodes` 改成 `200`，观察两种方法谁更早学到可用策略。
2. 把学习率从 `1e-3` 改成 `5e-4` 或 `2e-3`，比较 Baseline 是否仍然更稳。
3. 在脚本中打印 `advantages.mean()` 和 `advantages.std()`，观察优势信号是否围绕 0 波动。
4. 把 Value Network 的隐藏层从 `128` 改成 `32`，观察基线估计变弱后训练曲线是否更抖。

## 参考文献

[^1]: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. _Machine Learning_, 8(3-4), 229-256. [DOI](https://doi.org/10.1007/BF00992696)

[^2]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. _Advances in Neural Information Processing Systems_, 12.

[^3]: Gymnasium. CartPole-v1 documentation. <https://gymnasium.farama.org/environments/classic_control/cart_pole/>
