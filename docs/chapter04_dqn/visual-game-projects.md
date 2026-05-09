# 4.6 项目：把 DQN 从 LunarLander 推到视觉游戏

前面几节我们已经把 DQN 拆开看过：Q 网络负责估计动作价值，经验回放负责打散样本，目标网络负责稳定 TD Target。现在要做的不是再记一遍定义，而是把这些零件放回一个完整训练过程里，看看它们在不同任务中各自解决什么问题。

先从 LunarLander 开始，因为它足够具体，也足够克制。它不是只有几步就能满分的玩具平衡任务，也不是一上来就要处理图像、帧堆叠和卷积网络的 Atari。它给我们的状态仍然是 8 个数字，动作仍然是离散的 4 个选择；但飞船的位置、速度、角度和落地接触都会影响奖励，训练曲线也会明显波动。这样的环境很适合回答本节的第一个问题：**DQN 的三个组件放在一起，怎样让一个随机控制策略逐步学到可解释的动作偏好？**

等这个问题看清楚，再把状态从 8 个数字换成屏幕像素。到那时，DQN 的核心公式没有变，变的是表示方式和工程压力：MLP 要换成 CNN，单帧要变成帧堆叠，训练时间也会从课堂练习变成更长的实验。

## 本节导读

**核心内容**

- 在 LunarLander 上从零搭出一个最小 DQN，读懂 Q 网络、经验回放、目标网络和探索策略在代码里的位置。
- 解释为什么从低维状态迁移到 Atari 像素输入时，真正新增的问题是表示学习，而不是 TD Target 本身。
- 用 ViZDoom 和宝可梦作为复杂度边界，理解部分可观测、稀疏奖励和长时规划为什么会让朴素 DQN 变吃力。

**核心公式**

$$
y_i = r_i + \gamma(1-d_i)\max_{a'}Q(s'_i,a';\theta^-)
\quad \text{（用目标网络构造 TD Target）}
$$

$$
\mathcal{L}(\theta)
=
\frac{1}{B}\sum_{i=1}^{B}
\left(y_i-Q(s_i,a_i;\theta)\right)^2
\quad \text{（一批样本上的均方 TD Error）}
$$

这两行公式对应代码里的两个动作。第一行是在说“这一步经验应该给当前动作打多少分”；第二行是在说“网络现在的回答和这个目标差多少”。经验回放只决定这一批样本从哪里来，目标网络只决定 $y_i$ 用哪套参数算，梯度下降则负责把 $\theta$ 往误差更小的方向推。

## 4.6.1 先在 LunarLander 上跑通 DQN

先看任务本身。LunarLander 里，智能体控制一艘小型登月舱，让它尽量平稳地落在两个旗帜之间。动作只有 4 个：不喷、开左侧喷口、开主发动机、开右侧喷口。

![LunarLander-v3 环境：控制登月舱平稳降落在两个旗帜之间](./images/lunarlander.gif)

状态是 8 维向量：

| 分量                  | 含义                 | 读数时先问什么             |
| --------------------- | -------------------- | -------------------------- |
| `x, y`                | 飞船相对着陆区的位置 | 离中心有多远，高度还剩多少 |
| `vx, vy`              | 水平和垂直速度       | 是否正在横漂，下降是否过快 |
| `angle, angular_vel`  | 倾斜角和角速度       | 姿态是否稳定               |
| `left_leg, right_leg` | 两条支架是否接触地面 | 是否已经接触并接近落稳     |

先看直觉：如果飞船下降太快，主发动机应该更有价值；如果姿态向一侧歪，某个侧喷口应该更有价值。DQN 要学的不是“某个状态好不好”，而是“在这个状态下，四个动作分别有多值钱”。所以 Q 网络的输入是 8 个状态数字，输出是 4 个动作价值。

### Q 网络：把一行状态变成四个动作分数

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)
```

这段网络很朴素：8 维输入，两个隐藏层，4 维输出。输出的四个数可以理解成：

```text
[Q(不喷), Q(左侧喷口), Q(主发动机), Q(右侧喷口)]
```

选择动作时，利用阶段只需要取最大值对应的动作；探索阶段则故意随机试一试。这样做的原因和第 3 章一样：如果一开始就永远选择当前看起来最好的动作，智能体可能永远没有机会发现更好的动作。

### 经验回放：不要只盯着刚发生的几步

强化学习的数据不是预先打乱好的训练集，而是智能体一边行动一边产生的。连续几步经验通常高度相关：飞船刚刚下降、下一帧还是下降，只是位置和速度微微变了一点。如果立刻拿最近几步反复训练，网络很容易被当前局面牵着走。

经验回放的做法很直接：先把交互得到的转移都存起来，再随机抽一批训练。

```python
import random
from collections import deque

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)
```

这段代码只做两件事：`push` 存经验，`sample` 随机取一批。看起来不复杂，但它改变了训练数据的性质。网络每次更新时，不再只看到刚刚发生的一小段轨迹，而是看到来自不同阶段的混合经验。

### 智能体：把四个组件接起来

现在把 Q 网络、目标网络、回放池和 $\varepsilon$-贪婪策略放进一个类里。先看完整代码，再逐行解释关键位置。

```python
import numpy as np
import torch.optim as optim

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=20_000,
        batch_size=64,
        target_update=1_000,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    def epsilon(self):
        progress = min(self.steps_done / self.epsilon_decay, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon():
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return int(self.q_net(state_tensor).argmax(dim=1).item())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * (1 - dones) * next_q_values

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
```

真正要抓住的是 `update` 里的三行。

第一行：

```python
q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
```

Q 网络会为每个状态输出 4 个动作分数，但这批经验里每条样本只实际执行了一个动作。`gather` 做的事情，就是从四个分数里取出“当时真的执行的那个动作”的 Q 值。

第二行：

```python
targets = rewards + self.gamma * (1 - dones) * next_q_values
```

这就是 TD Target。`dones` 的作用很重要：如果 episode 已经结束，下一状态就没有未来价值，未来项必须被清零。

第三行：

```python
loss = nn.functional.mse_loss(q_values, targets)
```

这就是均方 TD Error。网络不是直接改某个表格格子，而是通过反向传播调整参数，让实际执行动作的 Q 值更接近 TD Target。

### 训练循环：每一步都在收集和修正

```python
import gymnasium as gym

env = gym.make("LunarLander-v3")
agent = DQNAgent(state_dim=8, action_dim=4)

num_episodes = 800
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset(seed=episode)
    total_reward = 0.0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.push(state, action, reward, next_state, float(done))
        agent.update()

        if agent.steps_done % agent.target_update == 0:
            agent.update_target()

        state = next_state
        total_reward += reward

        if done:
            break

    reward_history.append(total_reward)
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(reward_history[-50:])
        print(
            f"Episode {episode + 1:4d} | "
            f"Avg50={avg_reward:8.1f} | "
            f"epsilon={agent.epsilon():.3f}"
        )

env.close()
```

这段循环可以按“交互、存储、学习、同步”四个词来读：

1. 交互：用当前策略选择动作，让环境返回下一状态和奖励。
2. 存储：把 `(state, action, reward, next_state, done)` 放进回放池。
3. 学习：从回放池随机采样，更新 Q 网络。
4. 同步：每隔一段时间，把 Q 网络参数复制给目标网络。

训练结果不应该被读成一条平滑上升曲线。LunarLander 的短训练经常会波动：某一阶段学会减速，之后又因为探索或 Q 值偏差退步。严谨的判断方式不是盯某一轮，而是和随机策略基线比较。随机策略通常在 -200 左右；短训练 DQN 如果能把平均回报稳定推到明显高于这个水平，就说明它已经学到了一些控制规律。要稳定解决环境，通常还需要更长训练、更稳的超参数，或者使用 Double DQN、Dueling DQN 等改进。

测试时要关闭探索，只按 Q 值最大动作行动：

```python
test_env = gym.make("LunarLander-v3")
returns = []

for seed in range(10):
    state, _ = test_env.reset(seed=10_000 + seed)
    total_reward = 0.0

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = int(agent.q_net(state_tensor).argmax(dim=1).item())

        state, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    returns.append(total_reward)

test_env.close()
print(f"测试平均回报: {np.mean(returns):.1f}")
```

这里用 10 轮平均，而不是只跑 1 轮。原因很简单：强化学习环境有随机性，一次成功或失败都不足以说明策略真的稳定。

## 4.6.2 从 8 个数字到一张屏幕

现在换个角度。LunarLander 的状态是 8 个数字，Q 网络只需要一个 MLP。Atari Pong 的状态却是一张屏幕：你看到的是像素，不是“球的坐标”和“球拍的位置”。这时 DQN 的 TD Target 仍然是：

$$
r+\gamma\max_{a'}Q(s',a';\theta^-)
$$

真正改变的是 $Q(s,a;\theta)$ 里的“状态表示”。网络必须先从图像里自己学出有用特征，再输出动作价值。

| 问题     | LunarLander        | Atari Pong                       |
| -------- | ------------------ | -------------------------------- |
| 状态     | 8 维向量           | 4 帧堆叠的 84×84 图像            |
| 网络     | MLP                | CNN + 全连接层                   |
| 关键困难 | 控制噪声和训练波动 | 从像素中提取位置、速度和运动方向 |
| 训练成本 | 可以作为课堂短训练 | 通常需要更长时间和更强硬件       |

为什么要堆叠 4 帧？因为单张图像只告诉你“球在哪里”，却不告诉你“球往哪里走”。连续几帧放在一起，网络才能从位置变化里推断速度和方向。这一步不是装饰，而是把“静态图片”变成“含有短期运动信息的状态”。

Gymnasium 已经提供了常用预处理：

```python
import gymnasium as gym

def make_atari_env(game_id="ALE/Pong-v5"):
    env = gym.make(game_id)
    env = gym.wrappers.AtariPreprocessing(
        env,
        grayscale_new=True,
        scale_obs=True,
        frame_skip=4,
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    return env

env = make_atari_env()
state, _ = env.reset()
print(state.shape)  # (4, 84, 84)
```

这段代码背后有三步：灰度化减少颜色维度，缩放到 84×84 降低计算量，帧堆叠保留运动信息。这样处理后，输入从原始游戏画面变成了一个适合 CNN 的张量。

### CNN Q 网络：让网络先学会“看”

```python
class CNNQNetwork(nn.Module):
    def __init__(self, input_channels=4, num_actions=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

这不是一套新算法。它仍然输出每个动作的 Q 值，仍然用经验回放和目标网络训练。只是前半段网络从“读 8 个数字”变成了“读图像局部结构”。卷积层会先学习边缘、形状和运动线索，全连接层再把这些线索合成动作价值。

和 LunarLander 代码相比，Atari 版本主要增加这些工程点：

| 改动             | 为什么需要                                     |
| ---------------- | ---------------------------------------------- |
| CNN              | 像素有空间结构，MLP 展平后会浪费这种结构       |
| 帧堆叠           | 单帧无法判断运动方向                           |
| 更大的回放池     | 图像状态更多样，需要保存更多经验               |
| 奖励裁剪         | 不同 Atari 游戏奖励尺度不同，裁剪便于统一训练  |
| 梯度裁剪         | CNN 参数更多，训练时更容易出现不稳定更新       |
| 更慢的目标网同步 | 像素任务更难，目标网络需要保持更长时间的稳定性 |

所以，从 LunarLander 到 Atari 的迁移，不是把 `env_id` 换掉这么简单。**TD 学习的骨架没有变，状态表示和训练工程变复杂了。**

## 4.6.3 再往前一步：部分可观测的 ViZDoom

Atari Pong 里，你几乎总能看到完整局面：球在哪里，两个球拍在哪里，都在屏幕上。ViZDoom 不一样。第一人称视角只给你眼前画面，身后和转角外发生了什么，你看不到。

这就是部分可观测性。它带来的问题不是“CNN 不够大”，而是“当前一帧本来就缺信息”。帧堆叠可以缓解一部分，因为最近几帧能告诉你自己刚才怎么移动、敌人是否正在靠近；但在复杂地图里，4 帧短期记忆通常不够。

![ViZDoom Deadly Corridor 场景：第一人称视角下的 3D 走廊](./images/vizdoom-deadly-corridor.png)

如果把 Atari 的 CNN-DQN 迁移到 ViZDoom，代码结构表面上差不多：预处理图像、堆叠帧、用 CNN 输出动作 Q 值、用经验回放训练。但要关注的困难已经变了：

| 变化           | 对 DQN 的影响                        |
| -------------- | ------------------------------------ |
| 第一人称视角   | 单帧看不到全局，状态不再完全描述环境 |
| 3D 导航        | 走廊、转角、距离感让动作后果更难预测 |
| 延迟反馈       | 当前移动可能很久之后才影响生存或得分 |
| 动作含义更复杂 | 前进、转向、射击之间存在组合关系     |

这时，朴素 DQN 还能作为起点，但它不再是完整答案。更自然的改进方向是加入更长记忆，例如用 RNN 处理历史帧，或者使用更强的探索方法。这里要学到的不是“ViZDoom 应该怎么调参”，而是看到 DQN 的边界：当状态本身不完整时，只把 Q 网络做深，并不能自动补齐缺失信息。

## 4.6.4 宝可梦：长时规划和稀疏奖励

最后看一个更极端的例子：宝可梦红。它和 Pong 都是像素游戏，但难度不是同一个层级。

Pong 的一局很短，得分反馈很直接。球没接住，马上扣分；球打赢一回合，马上加分。宝可梦则完全不同：走路、对话、进房间、战斗、练级、拿徽章之间隔着很长的动作链。很多随机操作在短期内看不出好坏，真正有意义的奖励可能几千步之后才出现。

| 问题       | Pong                     | 宝可梦红                         |
| ---------- | ------------------------ | -------------------------------- |
| 决策链长度 | 短，一局几十到几百步     | 长，关键目标可能隔着数千步       |
| 奖励密度   | 得分变化频繁             | 徽章、剧情推进等主奖励很稀疏     |
| 状态含义   | 球和球拍位置             | 地图、背包、等级、剧情状态等混合 |
| 探索难度   | 随机动作也能很快看到反馈 | 随机动作很难碰巧完成长序列目标   |

如果只把屏幕像素交给 DQN，再把“获得徽章”当作奖励，智能体大概率什么都学不到。不是因为 TD Target 写错了，而是因为探索几乎找不到有学习价值的样本。现实项目里通常需要辅助奖励、状态抽象、课程设计，甚至换成更适合长时规划的算法。

这也是本节想传达的最后一个判断：DQN 是深度强化学习的关键起点，但有明确边界。它把 Q-Learning 从表格推到了神经网络，让智能体可以处理连续状态和像素输入；但环境一旦出现严重稀疏奖励、长时依赖和部分可观测性，就需要新的算法和更细的任务设计。

## 本节收获

- 在 LunarLander 中，DQN 的核心训练循环可以概括为：交互、存储、采样、计算 TD Target、反向传播、同步目标网络。
- 从 LunarLander 到 Atari，核心公式不变，变化的是状态表示：MLP 处理向量，CNN 处理像素，帧堆叠提供运动信息。
- ViZDoom 提醒我们：部分可观测会让“当前状态”本身不完整，DQN 需要记忆机制或更强表示来补足历史信息。
- 宝可梦提醒我们：稀疏奖励和长时规划不是简单扩大网络就能解决的，它们需要奖励设计、探索策略和更合适的算法。

下一节我们回到算法本身，看研究者如何沿着 DQN 继续改进：减少 Q 值高估、分离状态价值和动作优势、以及组合多个技巧形成 Rainbow。[DQN 家族与视角迁移](./dqn-family)
