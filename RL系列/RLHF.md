# RLHF

# 前言



# 1.强化学习基础

## 1.1 有限马尔可夫决策过程



## 1.2价值函数与优势函数

​	**状态价值函数**，在Sutton的著作中，将状态价值函数的定义如下：
$$
v_{\pi}(s) \doteq \mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]=\mathbb E_{\pi}\bigg[ \sum_{k=0}^{\infin} \gamma^{k}R_{t+k+1} \bigg],\forall s \in \mathcal S
$$
​	状态价值函数反映了从状态 $S_t=s$ 开始，按照策略 $\pi$​执行动作后，智能体预计将获得的总奖励的期望值。

![image-20250210155625587](E:\Study\知识构建\RL系列\assets\image-20250210155625587.png)

​	而价值函数满足某种递归关系，对于任何策略$\pi$和状态$s$，$s$的价值与其后继状态的价值关系如下：
$$
\begin{aligned} v_{\pi}(s)&\doteq \mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]\\
&=\mathbb E_{\pi}\bigg[ R_t+\gamma G_{t+1}|S_t=s\bigg] \\
&=\sum_{a}\pi(a|s)\sum_{r,s'} p(r,s'|s,a)(r+\gamma v_{\pi}(s'))\end{aligned}
$$
l	**动作价值函数** 则是根据策略$\pi$，从状态$s$开始，执行动作$a$之后所有可能的决策序列的期望回报，我们用符号$q_{\pi}(s,a)$表示：
$$
\begin{aligned}q_{\pi}(s,a)&\doteq  \mathbb E_{\pi}\bigg[ G_t|S_t=s,A_t=a\bigg]\\
&=\sum_{r,s'}p(r,s'|s,a)(r+\gamma v_{\pi}(s'))

\end{aligned}
$$
​	值得注意的是回报$R_t$是一个简写，完整的表示应该是$R(S_t,A_t,S_{t+1})$。我们将智能体在策略$\pi$控制下的预期折扣收益用符号$J(\pi)$表示：
$$
J(\pi)=\mathbb E_{\pi}\bigg[ G_t\bigg]=\mathbb E_{\pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t R(S_t,A_t,S_{t+1}) \bigg]
$$
​	**公式变体**，在某些强化学习论文中，作者会为了简化公式的表达或者让某个特定的推导更清晰，而对标准符号做一些变动。如省略某些下角标或者是加上下角标变量，亦或者是对公式进行拓展，使得读者在阅读不同论文或材料时感到困惑，在本文接下来的内容中会涉及到不同的论文公式推导，为方便读者理解，笔者先列举一些公式的变体：
$$
\begin{aligned}J(\pi)&=\mathbb E_{\pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t R(S_t,A_t,S_{t+1}) \bigg] \\
&= \mathbb E_{s_0,a_0,...}\bigg[ \sum_{t=0}^{\infin}\gamma^t R(S_t,A_t,S_{t+1}) \bigg]\\
&=\mathbb E_{\tau\sim(\pi,E)} \bigg[ G(\tau)\bigg] \\
&=\sum_{\tau \sim (\pi,E)} P(\tau|\pi)G(\tau)\end{aligned}
$$
​	公式的改写本质上是对同一个问题的不同表示方式，如上式，第一二行通常用于标准的动态规划方法或分析中，直接表示在时间步$t$处从状态$S_t$开始采取动作$A_t$所获得的累积奖励的期望。第三四行则是从策略 $\pi$和环境$E$生成的轨迹序$\tau$的角度出发。同样，状态价值函数与动作价值函数也可以进行适当改写：
$$
\begin{aligned} v_{\pi}(s)&\doteq \mathbb E_{\pi}\bigg[ G_t|S_t=s\bigg]\\
&=\mathbb E_{\pi}\bigg[ R_t+\gamma G_{t+1}|S_t=s\bigg] \\
&=\mathbb E_{S_t=s,A_t,...} \bigg[ \sum_{t'=t}^{\infin}\gamma^{t'-t}R(S_{t'}=s,A_{t'},S_{t'+1})\bigg],\\
q_{\pi}(s,a)&\doteq  \mathbb E_{\pi}\bigg[ G_t|S_t=s,A_t=a\bigg] \\
&=\mathbb E_{S_t=s,A_t=a,...}\bigg[\sum_{t'=t}^{\infin}\gamma^{t'-t}R(S_{t'}=s,A_{t'}=a,S_{t'+1}) \bigg]\end{aligned}
$$
​	**优势函数**，

$$
\begin{aligned}A_{\pi}(s,a)=\underbrace{q_{\pi}(s,a)}_{\text{On-Policy} \newline }-\underbrace{v_{\pi}(s)}_{\text{On-Policy}}\end{aligned}
$$

# 2.RewardModeling



# 3.策略梯度算法

​	我们现在来看一下什么是策略梯度算法，以及其背后的动机与直觉。我们希望我们的策略能够使得回报的期望最大，动作的轨迹是在策略$\pi_{\theta}$的控制下产生的，因此我们的目标函数可以表达如下：
$$
\begin{aligned}J(\pi_{\theta})&=\mathbb E_{\tau \sim(\pi_{\theta},E)}\bigg[G(\tau)\bigg] \\
&=\sum_{\tau \sim (\pi_{\theta},E)} P(\tau|\theta)G(\tau)\end{aligned} \tag{2.1}
$$
​	其中$E$是环境，这个目标函数衡量了在指定环境下从我们的策略中采样的轨迹的理论收益。如果我们想找到最大化这个目标函数的参数$\theta$，我们可以通过梯度上升的方式不断迭代更新$\theta$。更新过程表示如下：
$$
\theta_{t+1}=\theta_t+\alpha \nabla_{\theta} J(\pi_{\theta})|_{\theta_t} \tag{2.2}
$$
​	其中$\nabla_{\theta} J(\pi_{\theta})|_{\theta_t}=\nabla_{\theta_t} J(\pi_{\theta_t})$，也就是所谓的**策略梯度**。现在的问题在于，我们如何计算$\nabla_{\theta} J(\pi_{\theta})|_{\theta_t}$​？利用策略梯度则需要一个明确的可计算的表达式。接下来我们进一步探索如何找到这个明确的表达式并加以运用。

​	**1.轨迹的概率**，给定策略$\pi_{\theta}$下轨迹$\tau=(s_0,s_1,...,s_{T+1})$出现的概率如下：
$$
P(\tau|{\theta})=\rho(s_0)\prod_{t=0}^{T}P(s_{t+1}|s_t,a_t)\pi_{\theta}(a_t|s_t) \tag{2.3}
$$
​	**2.Log-Derivative Trick**，利用$\log$求导的技巧$\part_x \log(f(x))=\frac{\partial_x f(x)}{f(x)}$，对轨迹概率关于$\theta$求导，我们有：
$$
\nabla_{\theta}P(\tau|{\theta})=\nabla_{\theta}\log {P(\tau|\theta)}\cdot P(\tau|\theta) \tag{2.4}
$$
​	**3.轨迹的对数概率**，$P(\tau|\theta)$​的对数概率如下：
$$
\log P(\tau|\theta)=\underbrace{\log(\rho(s_0))}_{与\theta 无关}+\sum_{t=0}^{T}\underbrace{\log {P(s_{t+1}|s_t,a_t)}}_{与\theta 无关}+\log(\pi_{\theta}(a_t|s_t)) \tag{2.5}
$$
​	**4.轨迹对数概率的梯度**，由于部分项与$\theta$无关，所以$\nabla_{\theta}\log P(\pi|\theta)$表达如下：
$$
\begin{aligned}\nabla_{\theta}\log P(\tau|\theta)&=\nabla_{\theta}\sum_{t=0}^{T}\log (\pi_{\theta}(a_t|s_t)) \\&=\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t))\end{aligned} \tag{2.6}
$$
​	将上述式子结合，求出$\nabla_{\theta}J(\pi_{\theta})$​则有：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})&=\nabla_{\theta}\sum_{\tau \sim (\pi_{\theta},E)}P(\tau|\theta)G(\tau) \\
&=\sum_{\tau \sim (\pi_{\theta},E)}\nabla_{\theta}P(\tau|\theta)G(\tau) \\
&=\sum_{\tau \sim (\pi_{\theta},E)}\nabla_{\theta}\log P(\tau|\theta) P(\tau|\theta)R(\tau) \\
&=\sum_{\tau \sim (\pi_{\theta},E)}\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t)) P(\tau|\theta)G(\tau) \\
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t))G(\tau)] \end{aligned} \tag{2.7}
$$
​	也就是说，我们可以根据采样到的轨迹来计算策略梯度，即如果收集到了一系列由$\pi_{\theta}$产生的轨迹$D=\set{\tau_i}_{i=1}^{N}$，我们可以利用如下公式估计策略梯度：
$$
\hat {\nabla_{\theta}}J(\pi_{\theta})=\frac{1}{|D|}\sum_{i=1}^{N}\sum_{t=0}^{T}\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t))G(\tau) \tag{2.8}
$$
​	如上便是策略梯度最简单的形式。如果我们能够在环境中运行策略来收集轨迹数据集，我们可以计算策略梯度并采取更新步骤。

​	**Don't Let the Past Distract You**.策略梯度的公式告诉我们对于一个轨迹$\tau=(s_0,s_1,...,s_T)$，每一个时刻的动作执行后对应的策略梯度都会乘以一个因子，即回报$R(\tau)$，但这并没有什么意义，因为是否强化智能体当前的决策动作只因和当前动作执行后产生的影响有关，如果当前动作执行后的收益低则不应当强化当前动作，反之亦然，而不能受先前因素的影响。因此，对于回报$R(\tau)$我们可以做出适当改进，不再考虑当前时刻$t'$之前的收益，则策略梯度可以重写为：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}(\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t) \sum_{t=t'}^{T}R(s_{t'},a_{t'},s_{t'+1}))] \end{aligned} \tag{3.10}
$$
​	在这个形式下，动作只基于在采取行动后获得的奖励而得到强化。我们将这种形式称为 **“Reward-to-go”**，因为回报为在轨迹的一个点之后奖励的总和。

​	策略梯度算法有一些变种形式，这些变种都与我们先前所学习的内容有所关联，接下来我们学习一些变种算法从而对策略梯度算法有更深入的理解。

​	**Baseline in Policy Gradients.** 使用策略梯度面临着一个问题——准确地估算出梯度需要大量的样本轨迹，使用 EGLP 引理，我们可以证明**Reward -to-go**——尽管没有改变政策梯度的期望值——减少了我们估计的方差，因此减少了估计策略梯度所需的轨迹总数。而EGLP的直接结果是对于任何直接依赖于状态的函数$b$，我们有：
$$
\begin{aligned} \mathbb E_{a_t \sim \pi_{\theta}} [\nabla_{\theta}\log(\pi_{\theta}(a_t|s_t)b(s_t)]=0\end{aligned} \tag{3.11}
$$
​	这使得我们可以在不改变期望值的情况下在策略梯度表达式上加上或减去任意项：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} [\sum_{t=0}^{T}\bigg(\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t) \big(\sum_{t=t'}^{T}R(s_{t'},a_{t'},s_{t'+1})-\underbrace{b(s_t)}_{\text{Baseline}}\big)\bigg)] \end{aligned}
\tag{3.12}
$$
​	在这个表达式中，任何函数$b$都称作$\text{baseline}$。最常见的基线选择是状态价值函数$V^{\pi}(s_t)$。回想一下，这是一个智能体从状态开始，然后在其剩余生命周期内按照策略行事时获得的平均回报。从**经验上**讲，这种选择具有减少策略梯度样本估计方差的效果，使得策略梯度学习更快更稳定。从直观上将，假设在某个状态$s_t$下，智能体采取了一个动作 $a_t$，并得到了一个回报。如果这个回报远高于该状态的预期回报（即 $V^{\pi}(s_t)$），我们认为这个动作“更好”；如果低于预期回报，我们认为该动作“不好”，通过减去$V^{\pi}(s_t)$​，我们将焦点集中在“超出预期的部分”（即优势）上。这就像在与基准相比时，我们只关注某个动作相对于平均策略的改进或削弱。

> [!NOTE]
>
> 值得注意的是，$V^{\pi}(s_t)$并不能准确的计算，所以需要近似计算，通常我们用一个神经网络$V_{\phi}^{\pi}(s_t)$估计价值函数，它与策略网络同时更新，而$V_{\phi}^{\theta}$的优化目标通常是最小均方误差（包括$VPG,TRPO,PPO$等），即：
> $$
> \phi_k= \arg\min_{\phi} \mathbb E_{s_t,\hat{R_t}\sim \pi_k}\big[ \big( V_{\phi}(s_t)-\hat{R_t}\big)^2 \big] \tag{3.13}
> $$

​	我们可以以一种更一般地形式写出策略梯度：
$$
\begin{aligned} \nabla_{\theta}J(\pi_{\theta})
&=\mathbb E_{\tau \sim (\pi_{\theta},E)} \bigg[ \sum_{t=0}^{T}\bigg(\nabla_{\theta}\log (\pi_{\theta}(a_t|s_t) \Psi(t)\bigg) \bigg] \end{aligned} \tag{3.14}
$$
​	$\Psi(t)=R(\tau)$时$\nabla_{\theta}J(\pi_{\theta})$是基础，$\Psi(t)=\sum_{t=t'}^{T}R(s_{t'},a_{t'},s_{t'+1})$时是**Reward-to-go**，$\Psi(t)=\sum_{t=t'}^{T}R(s_{t'},a_{t'},s_{t'+1})-b(s_t)$是**Reward-to-go with baseline**。此外，$\Psi(t)$的选择还可以是动作价值函数$Q^{\pi}(s_t,a_t)$，优势函数$A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)$，利用优势函数的策略梯度的公式化极为常见，并且不同算法使用的优势函数有许多不同的估计方法。

## 3.1TRPO

​	```为保证上下文符号一致，笔者在本章节推导上的符号并未遵循原论文， 进行了一定改动。```

​	一个策略$\tilde \pi$关于另一个策略$\pi$的预期收益在累计时间步上的优势为：
$$
J(\tilde \pi)-J(\pi)=\mathbb E_{\tau\sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(s_t,a_t)\bigg]\tag{3.15}
$$
​	证明如下：
$$
\begin{aligned} \mathbb E_{\tau \sim \tilde \pi}&\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(s_t,a_t)\bigg] \\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t(q_{\pi}(s_t,a_t)-v_{\pi}(s_t))\bigg] \\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^t(R(s_t,a_t,s_{t+1})+\gamma v_{\pi}(s_{t+1})-v_{\pi}(s_t))\bigg]\\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tR(s_t,a_t,s_{t+1})+\gamma^{t+1}v_{\pi}(s_{t+1})-\gamma^t v_{\pi}(s_t)\bigg]\\
&=\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tR(s_t,a_t,s_{t+1})\bigg]+\mathbb E_{\tau \sim \tilde \pi}\bigg[\sum_{t=0}^{\infin}\gamma^{t+1}v_{\pi}(s_{t+1})-\gamma^t v_{\pi}(s_t)\bigg]\\
&=J(\tilde \pi)+\mathbb E_{\tau \sim \tilde \pi}\bigg[\sum_{t=1}^{\infin}\gamma^{t}v_{\pi}(s_{t})-\sum_{t=0}^{\infin}\gamma^t v_{\pi}(s_t)\bigg]\\
&=J(\tilde \pi)-\mathbb E_{\tau \sim \tilde \pi}\bigg[v_{\pi}(s_0)\bigg]\\
&=J(\tilde \pi)-J(\pi)\end{aligned}\tag{3.16}
$$

​	如果能保证$J(\tilde \pi)-J(\pi)$大于$0$，则能说明更新后的策略一直在进步，而优势函数这一项又可以改写成：
$$
\begin{aligned}\mathbb E_{\tau \sim \tilde \pi}&\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(s_t,a_t)\bigg] \\
&=\sum_{t=0}^{\infin}\sum_{s}p(s_t=s|\tilde \pi)\sum_a \tilde \pi(a_t=a|s)\gamma^tA_{\pi}(s,a)\\
&=\sum_{t=0}^{\infin}\sum_{s}\gamma^tp(s_t=s|\tilde \pi)\sum_a \tilde \pi(a_t=a|s)A_{\pi}(s,a)\\
&=\sum_{s}\sum_{t=0}^{\infin}\gamma^tp(s_t=s|\tilde \pi)\sum_a \tilde \pi(a_t=a|s)A_{\pi}(s,a)\\
&=\sum_{s}\rho_{\tilde \pi}(s)\sum_a \tilde \pi(a_t=a|s)A_{\pi}(s,a)\end{aligned}\tag{3.17}
$$
​	其中，$\rho_{\tilde \pi}(s)=p(s_0=s|\tilde \pi)+\gamma p(s_1=s|\tilde \pi)+,...，$$\tilde \pi$是之前的策略$\pi$更新后的新策略，上述式子中涉及到$p(s_t=s|\tilde \pi)$与$\tilde \pi(a_t=a|s)$，即我们要按照新的策略与环境交互才能得到轨迹，先确定新的策略$\tilde \pi$并得到一定量的样本才能求解，并计算是否满足$\mathbb E_{\tau \sim \tilde \pi}\bigg[ \sum_{t=0}^{\infin}\gamma^tA_{\pi}(s_t,a_t)\bigg]\gt0$。$TRPO$利用函数$\mathcal L_{\pi}(\tilde \pi)$代替原始目标函数：
$$
\mathcal L_{\pi}(\tilde{\pi})=J(\pi)+\sum_{s} \rho_{\pi}(s) \sum_{a} \tilde{\pi}(a \mid s) A_{\pi}(s, a) .\tag{3.18}
$$
​	只要策略更新的幅度不大，就可以用$\mathcal L_{\pi}(\tilde \pi)$近似原本的$J(\tilde \pi)$，所以那我们**怎么来保证其更新幅度不要太大呢**？为了解决这个求解信任区域的问题，文中引入了Kakade&Langford（2002）的结论——Conservative policy iteration：

$$
\begin{aligned}\pi_{\text {new }}(a \mid s)&=(1-\alpha) \pi_{\text {old }}(a \mid s)+\alpha \pi^{\prime}(a \mid s) \\
\eta\left(\pi_{\text {new }}\right) & \geq L_{\pi_{\text {old }}}\left(\pi_{\text {new }}\right)-\frac{2 \epsilon \gamma}{(1-\gamma)^{2}} \alpha^{2} \\
& \text { where } \epsilon=\max _{s}\left|\mathbb{E}_{a \sim \pi^{\prime}(a \mid s)}\left[A_{\pi}(s, a)\right]\right|\end{aligned}\tag{3.19}
$$
​	有了这个下界表达式，我们可以利用**minorization-maximization**算法通过$\mathcal L_{\pi_{old}}(\pi_{new})$迭代$J(\pi_{new})$​。该算法具体细节不在本文涉及范围内，值得注意的是，该原始结论只适合混合策略，但实际应用中的混合策略很少使用，因此作者将该结论拓展到了一般随机策略[x]。最终的优化目标变成：

$$
\underset{\pi_{\theta}}{\operatorname{maximize}}\left[\mathcal L_{\pi_{\theta_{\text {old }}}}(\pi_{\theta})-C D_{\mathrm{KL}}^{\max }\left(\pi_{\theta_{\text {old }}}, \pi_{\theta}\right)\right] .\tag{3.20}
$$
​	其中：
$$
\mathcal L_{\pi_{old}}({\pi}_{\theta})=J(\pi)+\sum_{s} \rho_{\pi_{\theta_{old}}}(s) \sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a) .
$$
​	由于$\pi_{\theta}(a|s)$​与新策略有关，无法对其直接采样，因此我们通过重要性采样的方式进行采样，我们将式子右边项进行变体：
$$
\begin{aligned}\sum_{s} &\rho_{\pi_{\theta_{old}}}(s) \sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\\
&=\sum_{s}\sum_{t} \gamma^t p(s_t=s|\pi_{\theta_{old}})\sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\\
&=\sum_{t}\gamma^t \sum_{s} p(s_t=s|\pi_{\theta_{old}})\sum_{a} {\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\\
&=\sum_{t}\gamma^t \mathbb E_{s\sim\rho_{old}} \bigg[ \sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\bigg]\\
&=\frac{1}{1-\gamma} \mathbb E_{s\sim\rho_{old}} \bigg[ \sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)\bigg]\end{aligned}\tag{3.21}
$$
​	$\sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)$​可以通过重要性采样的方式重新表述成：
$$
\begin{aligned}\sum_a{\pi_{\theta}}(a \mid s) A_{\pi_{\theta_{old}}}(s, a)&=\mathbb E_{a \sim q}\bigg[ \frac{{\pi_{\theta}}(a \mid s)}{q_(a \mid s)} A_{\pi_{\theta_{old}}}(s, a)\bigg]\\
&\mapsto\mathbb E_{a \sim \pi_{\theta_{old}}}\bigg[ \frac{{\pi_{\theta}}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A_{\pi_{\theta_{old}}}(s, a)\bigg]\end{aligned}
$$
​	故最终的优化目标为：
$$
\begin{aligned} \arg\max_{\theta}&\mathbb E_{s\sim \rho_{old},a\sim \pi_{old}}\bigg[ \frac{{\pi_{\theta}}(a \mid s)}{\pi_{\theta_{old}}(a \mid s)} A_{\pi_{\theta_{old}}}(s, a)\bigg] \\
&{\operatorname {subject to}}{\text{ }} \mathbb E_{s\sim \rho_{old}}\bigg[ D_{KL}(\pi_{\theta_{old}}(\cdot|s)||\pi_{\theta}(\cdot|s))\bigg] \leq \delta
\end{aligned}\tag{3.22}
$$

## 3.2 PPO 

​	我们现在将深入理解为语言模型对齐奠定基础的算法 —— $PPO$(近端策略优化算法)，$PPO$是一种基于策略梯度的强化学习算法，$PPO$的核心思想是通过在每次更新时保持策略的“平稳性”或“稳定性”，避免过度优化，从而减少策略更新过程中的波动性，$PPO$算法的优化目标如下：
$$
\left.J(\theta)=\frac{1}{G} \sum_{i=1}^{G} \min \left(\frac{\pi_{\theta}\left(a_{t} \mid s\right)}{\pi_{\theta_{o l d}}\left(a_{t} \mid s_t\right)} A_{t}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(a_{t} \mid s_t\right)}{\pi_{\theta_{o l d}}\left(a_{t} \mid s_t\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{t}\right)\right)
$$
​	注意到$\min$函数会使优化目标有不同取值的选择，我们来解释不同情况下的取值情况，首先定义策略比率如下：
$$
R(\theta)=\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$
​	当优势是正数且策略比率超过$1+\varepsilon$时，意味着新的策略更可能采取$a_t$行动，$\operatorname{clip(\cdot)}$通过裁剪比率防止策略更新过大，并限制$R(\theta)$的变化范围，此时优化目标：
$$
J(\theta)=\min(R(\theta),1+\varepsilon)A_t=(1+\epsilon)A
$$
​	意味着优化目标会增加策略比率，使得动作更可能发生，当优势是正数而策略比率小于$1-\varepsilon$​时，优化目标变成：
$$
J(\theta)=\min(R(\theta),1-\varepsilon)A_t=R(\theta)A_t
$$
​	意味着优化目标会减小策略比率，使得动作没那么可能发生。同理，如果策略比率本身介于$(1-\varepsilon,1+\varepsilon)$之间，则有：
$$
J(\theta)=\min(R(\theta),R(\theta))=R(\theta)
$$
​	没有任何影响，当优势是负数时且策略比率$R(\theta)\lt 1-\varepsilon$时，有：
$$
J(\theta)=\min(R(\theta)A_t,(1-\varepsilon)A_t)=(1-\varepsilon)A_t
$$
​	其他情况同理，在信任区域内的优化目标与策略梯度是一致的。

接下来我们将结合部分的代码深入理解$PPO$算法如何在$RLHF$中大展身手，如何完成大语言模型对齐的任务。

### 3.2.1  DeepSpeed-Chat 源码解析

![image-20250217111051594](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250217111051594.png)	奖励模型和评判模型的均源自于$\mathrm{RewardModel}$这个类别，接下来分别详细解读$\mathrm{forward}$与$\mathrm{forward\_value}$​函数的实现细节。

![image-20250217134640823](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250217134640823.png) 	将采样阶段模型的输入与输出$token$拼接后输入奖励模型，由$\mathrm{self.v\_head}$将奖励模型每一个位置输出的$\mathrm{logits} \in \mathbf R^{|V|\times 1}$映射成一维标量$\mathrm {rewards_i}\in\mathbf R,i=1,....,|S|$。代码第$66$行的变量$\mathrm {rewards}$其实就是一个形状为$[2B，S]$的矩阵，前$B$个对应$\mathrm{chosen}$样本，后$B$个对应$\mathrm{rejected}$样本。接着通过$\text{for}$循环的方式遍历所有样本将损失函数逐个累加...........。

​	在$\mathrm{training/step3\_rlhf\_finetuning/main.py}$下的$for$循环内有两个比较重要的函数，第一个是第$537$行的$\mathrm{generate\_experience}$，另一个是第$553$行的$\mathrm{train\_rlhf}$。分别用于生成轨迹以及计算损失函数。

![image-20250217155740680](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250217155740680.png)

​	先看第一个函数$\text{generate\_experience}，返回的字典中包括了$actor$和$reference$的对数几率，并且还有$actor$生成的序列的奖励以及每一个$token$对应的价值。价值计算由$$\text{critic\_model.forward\_value()}$产生，对应实现在dschat/utils/model/reward_model.py。

![image-20250217135919357](assets\image-20250217135919357.png)

​       在$\text{forward\_value}$函数中，目的是为了拿到模型输出的奖励，因此需要找到整个序列中只属于$answer$的这部分$token$，第$166$行得到的$c\_inds$是模型输出的回答部分结束的位置索引，第$168-169$行是拿到最后一个位置前的输出的$value$，并返回$value$序列以及最后一个位置获得的奖励。在拿到了生成的轨迹和奖励与价值后，继续往下看到$for$循环内的第$564$行代码，在$\text{train\_rlhf}$中完成了损失函数的计算。

![image-20250217165451472](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250217165451472.png)

​	右边子图是该方法的具体实现，其中又涉及到了四个重要的函数，第一个函数是计算$actor$在策略$\pi_{\theta}$下获得的奖励，第二个函数是计算策略$\pi_{\theta}$下的优势，第三个函数则是计算$PPO$损失，第四个函数则是计算$critic$损失，

### 3.3.2 



## 3.3 DPO

​	强化学习的优化目标可以用如下一般形式的公式进行概述：
$$
\max _{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\left[r_{\phi}(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_{\theta}(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right],\tag{3.3.1}
$$
​	即让新策略在不偏离原始策略太多的情况下尽可能地生成奖励模型认为较好的内容，$DPO$将此优化目标进行变体：
$$
\begin{aligned}\arg\max_{\theta}&\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\left[r_{\phi}(x, y)\right]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi_{\theta}(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]\\
&=\arg\max_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\left[r_{\phi}(x, y)\right]-\beta \pi_{\theta}(y \mid x)\log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm {ref}}(y \mid x)}  \\
&\propto \arg\max_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[[\frac{1}{\beta}r_{\phi}(x, y)]- \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm {ref}}(y \mid x)} \bigg] \\
&=\arg\min_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm {ref}}(y \mid x)}-\frac{1}{\beta}r_{\phi}(x, y) \bigg]\end{aligned}\tag{3.3.2}
$$
​	式子$\frac{1}{\beta}r_{\phi}(x,y)$和$\theta$无关，我们可以将其适当地改造变形，即：
$$
\begin{aligned} \frac{1}{\beta}r_{\phi}(x,y)=\log e^{r_{\phi}(x,y)/\beta}\end{aligned}\tag{3.3.3}
$$
​	故优化目标变为：
$$
\begin{aligned}&\arg\min_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm {ref}}(y \mid x)}-\frac{1}{\beta}r_{\phi}(x, y) \bigg] \\
&=\arg\min_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm {ref}}(y \mid x)e^{r_{\phi}(x,y)/\beta}}\bigg] \end{aligned}\tag{3.3.4}
$$
​	此时，优化目标中$\log$的分母已经不再是一个概率分布了，我们可以对分母进行归一化：
$$
\begin{aligned}Z(x)&=\sum_y \pi_{\mathrm {ref}}(y \mid x)e^{r_{\phi}(x,y)/\beta} ,\\
\arg\min_{\theta}&\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm {ref}}(y \mid x)e^{r_{\phi}(x,y)/\beta}}\bigg]\\
&=\arg\min_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\big(\pi_{\mathrm {ref}}(y \mid x)e^{r_{\phi}(x,y)/\beta}/Z(x)\big)Z(x)}\bigg] \\
&=\arg\min_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\pi^*(y\mid x)}-\log Z(x)\bigg]\\
&\Leftrightarrow\arg\min_{\theta}\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(y \mid x)}\bigg[\log \frac{\pi_{\theta}(y \mid x)}{\pi^*(y\mid x)}\bigg]\\
&=\arg\min_{\theta}\mathbb E_{x\sim \mathcal D}\bigg[ \mathbb D_{\mathrm {KL}}(\pi_{\theta}(y \mid x)||\pi^*(y\mid x))\bigg]\end{aligned}\tag{3.3.5}
$$
​	当$\pi_{\theta}(y\mid x)=\pi^*(y \mid x)$时有最小值，即最优概率分布为$\pi^*$​。而$\pi^*(y\mid x)$和$r(x,y)$的关系为：
$$
r(x, y)=\beta \log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)\tag{3.3.6}
$$
​	我们希望对于两个分得出好坏的回答$y_1\succ y_2$，奖励模型的输出有$r(x,y_1)\gt r(x,y_2)$，Bradley-Terry模型只考虑两个回答得分的差值，即希望$r(x,y_1)\gt r(x,y_2)$，且好的回答大于坏的回答的**几率**有：
$$
\begin{aligned}p^*(y_1 \succ y_2\mid x)&^=\frac{1}{1+\exp {(r(x,y_1)-r(x,y_2))}}\\
&=\frac{1}{1+\exp {(\beta \log \frac{\pi^*(y_1 \mid x)}{\pi_{\mathrm{ref}}(y_1 \mid x)}-\beta \log \frac{\pi^*(y_2 \mid x)}{\pi_{\mathrm{ref}}(y_2 \mid x)})}}\end{aligned}\tag{3.3.7}
$$
​	我们可以通过极大似然的方式来优化策略$\pi_{\theta}$，即：
$$
\begin{aligned}\mathcal L(\pi_{\theta};\pi_{\mathrm {ref}})&=-\log {\prod_{x,y_w,y_l \sim\mathcal D}}p^*(y_w \succ y_l\mid x)\\
&=-\mathbb E_{x,y_w,y_l \sim \mathcal D}\bigg[\log\sigma {(\beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}-\beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)})}\bigg]\end{aligned}\tag{3.3.8}
$$
​	此时，我们只需要两个模型了，一个$actor$和一个$reference$。原来的强化学习过程就转化成了SFT的形式。

### 3.3.1 DeepSpeedChat源码解析

​	这里笔者着重讲解框架中的损失函数计算部分，分为如下三部分

#### 3.3.1.1  构建label mask

​	![image-20250214172456512](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250214172456512.png)

​	如果开发者将batchsize设置成$B$，由于每一个$prompt$对应了一好一坏的回答，则一共会有$2B$个样本在一次迭代$step$中，前$B$个样本是好的回答，后$B$个样本是被拒绝的回答。而$label\_mask$则用于选择对应$token$位置的奖励，即$label\_mask=1$的位置会被考虑，其他为$0$的位置不会被纳入计算，由于奖励只是针对模型生成内容的部分，且对于模型的$prefilling$而言，前$k$个$token$一致时每个位置输出的$\mathrm {logits}$都一致，所以要找到$chosen$与$reject$的第一个顺序遇到不同的$token$的位置索引，将$label\_mask$该索引前元素都置为$0$，实现对应于代码第$465$-$473$行。第$463$行先是把不为$<pad>$的部分置$1$，再接着把$prompt$和$answer$中前$k$个相同$token$对应位置的$label\_mask$置$0$。

#### 3.3.1.2 计算对数概率

​	详见代码第$474-485$行与函数$\mathrm {def\_batch\_logps}$。![image-20250214174114978](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250214174114978.png)

​	我们需要先将$label$与$label\_mask$对应位置元素相乘，只剩下需要考虑的位置的$token$，因此将$\mathrm{logits}$序列对应$token$位置的$\mathrm {logit} \in \mathbf R^{|V|\times 1}$向量中该$token$的索引的分量，即$\pi_{\theta}(y_j\mid x),j=1,...,|y|$取出，通过$\mathrm {torch.gather}$函数实现，最终再相加作为整句话的奖励，笔者该处的符号表述多了一个下角标$j$，和$DPO$论文中的符号不一致，原因是Deepspeed-chat框架中奖励模型不是最朴素的奖励模型，而是$\text{Outcome Reward}$模型，即考虑$token$位置的奖励而非只考虑整句话最后一个位置的奖励。

#### 3.3.1.3 计算损失函数

![image-20250214180146011](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250214180146011.png)

​	$actor$与$reference$的对数概率变量$\mathrm {logps}$和$\mathrm {ref\_logps}$都计算完成后就可以根据公式$3.3.8$带入进行损失函数计算，$\sigma$函数内的变量对应图中代码的第$487-489$行。此外，框架中还对损失函数进行了平滑操作：
$$
\begin{aligned}\log\sigma(x) \mapsto (1-\alpha)\log\sigma(x)+\alpha \log\sigma(-x)\end{aligned}\tag{3.3.9}
$$
​	函数$(1-\alpha)\log\sigma(x)+\alpha \log\sigma(-x)$的图像如下：	![image-20250214151223507](E:\Study\gitpro\knowledge-planet\RL系列\assets\image-20250214151223507.png)

​	当$\alpha=0$时，为$\log\sigma(x)\in(-\infin,0)$，而随着$a$的增大，函数右边有往下的趋势，函数左边有往上抬的趋势，当$\alpha=1$时，函数和$\log\sigma(x)$图像完全反了过来（关于$y$轴对称）。即在训练中，虽然我们希望$\begin{aligned}  \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}-\beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}\end{aligned}$大于$0$，但实际情况可能是小于$0$的，越远离$0$则损失函数越大，可能导致训练不稳定，因此可以通过平滑系数对原函数进行趋势的控制。

​	如上便是Deepspeed-Chat框架的DPO实现代码，关键部分相较于$PPO$便于理解，实现起来也较为简单，当然，对奖励模型损失函数的选择，并不局限于$\log\mathrm {sigmoid}$函数，其是无界的，即便通过平滑项进行趋势控制也难以彻底解决遇到异常样本时梯度过大导致训练不稳定的问题，我们可以选择一个有界函数使得训练过程更稳定，如保真度损失，亦或者截断梯度。

## 3.4 GRPO





















# 参考文献

[[X]Policy Gradients: The Foundation of RLHF](https://cameronrwolfe.substack.com/p/policy-gradients-the-foundation-of)

[[X]Proximal Policy Optimization (PPO): The Key to LLM Alignment](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo)

[[X](WIP) A Little Bit of Reinforcement Learning from Human Feedback](https://rlhfbook.com/book.pdf)

[[x]Policy Gradient Algorithms,Weng, Lilian,liliangweng.github.io,2018](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

[[x]深度强化学习（三）：TRPO（Trust Region Policy Optimization ，信赖域策略优化）,Dreammaker](https://zhuanlan.zhihu.com/p/605886935)
