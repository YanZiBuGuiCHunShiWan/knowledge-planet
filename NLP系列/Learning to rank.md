# Learning to Rank

# 1.信息检索

## 1.1 排序学习



# 2.稠密检索技术

## 2.1Sentence-BERT



## 2.2SimCSE

### 

# 3.Ranking



## 3.1 Point-Wise



##3.2 Pair-Wise

### 3.2.1 RankSVM&IRSVM



### 3.2.2 RankNet& lambda Rank

​	RankNet 的核心思想是使用 **成对比较（pairwise approach）** 来学习一个排序函数，该函数可以根据文档对$(doc_i,doc_j)$的相关性预测它们相对于查询$q$的排序顺序。RankNet 的损失函数可以表示为：	
$$
\begin{align}
            C=\frac{1}{2}\left(1-S_{i j}\right) \sigma\left(s_{i}-s_{j}\right)+\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)
\end{align}
$$
​	对于给定的查询$Q$，$S_{ij}\in\{−1,0,1\}$取值如下：$S_{ij}=1$：如果 $doc_i$ 比 $doc_j $更相关；$S_{ij}=0$：如果 $doc_i$ 和 $doc_j$ 相关性相同；$S_{ij}=−1$：如果 $doc_i$ 比更不相关。$s_i$ 和$s_j$分别表示文档$doc_i$和$doc_j$的相关性评分。$\sigma$是一个超参数，用于缩放$s_i-s_j$的值。

### 3.2.4 Adaptive Schema and unified perspective



##3.3 List-Wise

​	listwise方法可以分为两类，位置有关的指标优化与位置无关的指标优化[x]。和位置有关的衡量指标有$MRR$,$MAP$,$NDCG$等，而模型参数关于这些衡量指标不可导，我们通常采用函数近似的方式构造一个可导函数作为优化目标，从而实现模型参数的更新。首先，我们需要明确的是，什么衡量指标/算子是不可导的？

### 3.3.1 不可导算子的可导近似

​	学高等数学的时候我们知道导数是指函数描述的是函数在某一处的变化率，可导描述的就是指导数在某一处的变化率是否存在，常见的可到操作有：加减乘除、平方、对数、指数、线性变化、切片等。而不可导就是指函数在某些点处的导数不存在，或者不具备可微性，常见的不可导操作有：阶跃函数、$\arg\max$​、$\max$、指示函数、排序、采样。

​	在深度学习中，训练神经网络时由于策略的选择原因，标准的优化目标可能涉及到不可导算子，因此通常需要找一个可导算子进行近似。常见的可导算子不可导近似如下：

|                    不可导操作                    |                       可导近似                        |
| :----------------------------------------------: | :---------------------------------------------------: |
|                      $\max$                      |             $\operatorname{log sum exp}$              |
|                    $\arg\max$                    | $\sum_{i=1}^{n}i*\operatorname{softmax}(\mathbf x)_i$ |
| $\operatorname{Indicator function}$:$I(s_i>s_j)$ |                     $p(s_i>s_j)$                      |
|              $\operatorname{sort}$               |          $\operatorname{Sinkhorn Operator}$           |
|            $\operatorname{sampling}$             |            $\operatorname{Gumbel-softmax}$            |

​	以表格中的$\max$不可导算子为例，$\operatorname{max}$ 算子的作用是从一个向量中获得最大值，如$v=(2,3,4,1,4,5)^{\top}$的最大值是$5$，则$\max v=5$，其近似如下：
$$
\max(x_1,x_2,...,x_n)\approx\lim_{K\rightarrow \infin}\frac{1}{K}\log\sum_{i=1}^{n}\exp(Kx_i)
$$
​	$K$越大则近似越好，当$K$取$1$时则$\max$算子的近似就是$\operatorname{log sum exp}$​。$\operatorname{sort}$算子和采样算子笔者将会在接下来的章节详细介绍。

### 3.3.2 SoftRank

​	SoftRank的思想是过文档的得分和排名进行概率建模，实现了对NDCG等指标的可微近似，从而使得梯度下降等优化方法得以应用。$NDCG$指标计算依赖于$DCG$和$IDCG$，但是$DCG$这个指标中涉及到了$\mathrm {sort}$的操作是不可导算子，因此训练时没法直接反向传播，如果将$DCG$和$IDCG$有一个平滑点的可导函数近似，那$NDCG$自然也就可导了。

​	具体地，假设当前查询$query$有$k$个候选文档集合$\set{doc_j}_{j=1}^{k}$。将$query$与$doc_j$拼接后得到的文本对$\mathbf x_j$送入一个$\mathrm{Encoder}$，得到神经网络输出的$k$个分数$f(\mathbf \theta,\mathbf x_j),j=1,..,k$，SoftRank假设当前文本对$\mathbf x_j$的输出分数$s_j$不再是确定的，而是服从于高斯分布：
$$
\begin{aligned} s_j \sim \mathcal N(s_j|f(\mathbf \theta,\mathbf x_j),\sigma_s^2)\end{aligned}
$$
​	如果给定两个文档$doc_i$与$doc_j$，从各自的高斯分布中采样得到的分数为$S_i,S_j$,我们想判断谁和$query$更加相似，那么就可以判断$S_i$与$S_i$谁大，但是由于分数是一个随机变量，因此我们看的是一个概率$P(S_i>S_j)$，即$\mathrm{Pr}(S_i-S_j)>0$，而服从高斯分布的随机变量之差仍然是高斯分布，我们定义文档$i$打败文档$j$的概率$\pi_{ij}$：
$$
\begin{aligned} \pi_{ij} :=  \operatorname{Pr}(S_i-S_j>0)=\int_{0}^{\infin} \mathcal N(s|\bar {s_i}-\bar{s_j},2\sigma_s^2) \operatorname {d}s \end{aligned}
$$
![image-20250325182632385](assets\Gaussian-area.png)	

​	我们可以基于成对比较的方式来近似排序，其背后直觉如下：如果文档$doc_j$的排名比较靠后，说明其和其他文档在对比时都被打败了，具体地：假设共计$5$个文档，$doc_2$的排名为$4$（从$0$开始排），说明$doc_2$在和其他四个比较时都被打败了，如果其他文档$doc_j$打败$doc_2$的概率较大，则说明$\pi_{i2},i\neq2$较大，当$\bar{s_i}-\bar{s_2}$较大时高斯分布大于$0$部分的面积接近$1$，此时有$r_{doc_2}=4\approx \sum_{i=1,i\neq 2}\pi_{i2}$。因此任意文档$j$的排序$r_j$的期望可以表示如下：
$$
\begin{aligned} \mathbb E\big[r_j\big]= \sum_{i=1,i\neq j}^N\pi_{ij}=\sum_{r=0}^{N-1}rP(r_j=r)\end{aligned}
$$
​	我们可以将上述式子看作一个$N-1$次的相互独立的伯努利实验，$r_j\sim\operatorname{Bernoulli}(\pi_{ij}),j\neq i$。但整体而言与标准的二项分布有所不同，标准的二项分布的概率质量函数有一个明确的解析式：
$$
P(X=k)=\begin{pmatrix} n \\ k\end{pmatrix}p^k(1-p)^{n-k}
$$
​	现在，文档$doc_j$的位置可以看成一个随机变量的期望，将$DCG$指标中的$D(r_j)$用$\mathbb E\big[D(r_j)\big]$替代，则我们可以得到一个可导的计算指标$\operatorname{SoftNDCG}$，即：
$$
\begin{aligned}DCG&=\sum_{i=1}^{N}g(j)D(r_j)\\
&\approx \sum_{i=1}^{N}g(j)\sum_{r=0}^{N-1}D(r_j)P(r_j=r)\end{aligned}
$$
​	我们只要$P(r_j=k)$知道就能计算$DCG$，文档$j$的排序位置$r_j$的取值可能为$0,...,N-1$，但是我们会发现情况有点复杂，即$P(r_j=k)$的解析式表达起来很繁琐，当$r_j$取值为$0$时虽然有$P(r_j=0)=\prod_{i=1,i\neq j}^N(1-\pi_{ij})$，但是当$r_j=1$时可能是$N-1$种情况，即:
$$
P(r_j=1)=\sum_{k=1,k\neq j}^{N}\big(\pi_{kj}\prod_{i=1,i\neq j，i\neq k}^{N}(1-\pi_{ij})\big)
$$
​	当$r_j=2$时，则有$\begin{pmatrix} N-1 \\ 2\end{pmatrix}$种情况，$r_j=3$时有$\begin{pmatrix} N-1 \\ 3\end{pmatrix}$种情况，更一般的，我们可以将$P(r_j=k)$表达如下：
$$
P(r_j=k)=\sum_{\substack{E\subseteq\{1,2,...,N\} \setminus \{j\}\\|E|=K}}^{}\big(\prod_{e\in E}\pi_{ej}\prod_{i=1,i\neq j，i\neq e}^{N}(1-\pi_{ij})\big)
$$
​	该概率质量函数虽然是解析式，但计算时需遍历子集，属于“非封闭形式”（因涉及到组合爆炸）。通常我们需要迭代进行求解，现在来思考一下不同视角下的$P(r_j=k)$的表达方式，从一开始，假设只有一个文档$doc_j$，那么第一次排序其排在位置$0$的概率必然是1，现在有第二个文档$doc_i,i\neq j$进来，我们需要确认文档$doc_j$排在$0$还是$1$，这种情况下如果$doc_j$仍然排在$0$的概率是$1-\pi_{ij}$，排在$1$的概率是从$0$位置跌落一名$\pi_{ij}$。假设有第三篇文档进来，则文档$doc_j$的位置排名只可能出现排序不变及往下跌落一位的情况，不可能上升，如果我们将文档$doc_j$的排序位置视作一个状态，则这个状态只与前一个状态有关，且只可能保持不变或者由前一个状态转移到下一个相邻的状态（第3个时刻的位置3只能转移到第四个时刻的位置3或者位置4，不可能转移到位置2或位置5），整体情况如下图所示：

![image-20250326165450215](assets\image-20250326165450215.png)

​	用一个类似于状态转移矩阵的方式刻画（图中灰色圆圈代表文档处于该位置的概率为0），将$P^{(i)}_j(r=k)$记作排序$i,i=1,..j-1,j+1..,N$篇文档时，文档$doc_j$排在位置$k$的概率，则我们可以写一个递推公式：
$$
P_j^{(i)}(r=k)=P_j^{(i-1)}(r=k-1)\pi_{ij}+P_j^{(i-1)}(r=k)\big(1-\pi_{ij}\big)
$$
​	最终计算得到$P_j^{(N)}(r=k):=P(r_j=k)$，针对所有的$j=1,...,N$,我们都会利用上述公式进行迭代计算得到一个位置分布向量$\mathbf P(r_j)=(P_j(0),P_j(1),....,P_j(N-1))^{\top}$，再基于公式x求得最终的$SoftNDCG$​​，作为损失函数进行反向传播。

### 3.3.3Approximate Rank & SmoothRank

​	Approximate Rank认为$NDCG$指标不连续的根本原因在于排序的位置关于排序的得分是一个不可导的映射，因此将排序位置用排序分数近似是一个非常直接的想法，具体地，$DCG=\sum_{i=1}^{N}g(j)D(r_j)=\sum_{i=1}^{N}g(j)/\log(1+\pi(\mathbf x_j))$，而$\pi(\mathbf x_j)$是文档在按照模型预测的相关性分数排序后的列表中的位置，从分数到位置的这个操作是不可导的，我们可以把$\pi(\mathbf x_j)$用$s_j=f(\theta,\mathbf x_j)$进行近似：
$$
\begin{aligned}\pi(\mathbf x_j)&=1+\sum_{i=1,i\neq j}^N\operatorname I\set{s_i>s_j}\\
&\approx 1+\sum_{i=1,i\neq j}^NP(s_i-s_j>0)\end{aligned}
$$
​	其中指示函数$\mathbf I$可以用概率近似，Approximate Rank提出以如下方式近似$\pi(\mathbf x_j)$：
$$
\begin{aligned}\hat \pi(\mathbf x_j)&=1+\sum_{i=1,i\neq j}^NP(s_i-s_j>0)\\
&=1+\sum_{i=1,i\neq j}\frac{\exp(-\alpha(s_j-s_i))}{1+\exp(-\alpha(s_j-s_i))}\end{aligned}
$$
​	故最后的损失函数是：
$$
\begin{aligned}L(f;\mathbf x,\mathbf y)&=1-G_{\max}^{-1}\sum_{j=1}^{N}g(j)/\log(1+\hat\pi(\mathbf x_j))\end{aligned}
$$
​	SmoothRank的思想与Approximate Rank类似，都是基于近似$\pi(\mathbf x_j)$的思想，区别在于近似时的概率质量函数不同，SmoothRank的具体近似公式如下：

​	



​	xxxxxxxx

### 3.3.4 ListNet&ListMLE

​	ListNet将排序视作一个概率分布，用交叉熵损失优化排序网络。具体地，ListNet先介绍了排列概率$\text{(Permutation Probability)}$，假设$\pi$是一个关于$n$个物品的排列，$\Phi$是一个严格单调增函数，给定一个分数列表$\mathbf s$，则排列$\pi$出现对应的概率定义为：

$$
P_{\mathbf s}(\pi):=\prod_{j=1}^n\frac{\Phi(\mathbf s_{\pi(j)})}{\sum_{k=j}^{n}\Phi(\mathbf s_{\pi(k)})}
$$
​	其中$\mathbf s_{\pi(j)}$是排序$\pi$中位置$j$的得分，假设有一个文档$\set{1,2,3}$的分数列表$\mathbf s=\set{s_1,s_2,s_3}$，则排序$\pi=\langle2,3,1\rangle$出现的概率：
$$
\begin{aligned}P_{\mathbf s}(\pi)=\frac{\Phi(s_2)}{\Phi(s_2)+\Phi(s_3)+\Phi(s_1)}\frac{\Phi(s_3)}{\Phi(s_3)+\Phi(s_1)}\frac{\Phi(s_1)}{\Phi(s_1)}\end{aligned}
$$
​	上述公式只是一个定义式，因为客观世界里排列实际出现的概率和分数列表没有关系，如假设有三个水果分别是“桃子、梨子、苹果”，实际的排列情况有$6$种可能性，每一个排列是等概率的。但是假设将这个水果的排列交给一个人来排列，则有可能人会根据自己的喜好将偏爱的水果放在前面，如果让一个人来排列多次这三个水果，很有可能是喜好的水果在前的次数较多，而排列概率模拟了人类排列物品的过程，但人为定义的排列概率是否满足概率分布的要求我们还需要证明，文中给出了引理$2$[x]：

> **Lemma 2** *The permutation probabilities $P_{\mathbf s}(\pi)$, $\pi \in \Omega_n$form a probability distribution over the set of permuta*
>
> *tions, i.e., for each* $\pi \in \Omega_n$， we have $P_{\mathbf s}(\pi)$, and$\sum_{\pi \in \Omega_n}P_{\mathbf s}(\pi)=1$.

​	给定两个长度皆为$n$分数列表$\mathbf s_1,\mathbf s_2$，则我们可以计算得到两个排列分布向量$\begin{pmatrix}\pi_{\mathbf s_1}^1\cdots\pi_{\mathbf s_1}^{n!}\end{pmatrix}^\top$与$\begin{pmatrix}\pi_{\mathbf s_2}^1\cdots\pi_{\mathbf s_2}^{n!}\end{pmatrix}^\top$。我们可以用一个度量概率分布差异的指标作为损失函数。在实际计算中，由于排列$n$个物品有$n!$种可能性，计算过于复杂，因此我们只考虑物品$j$被排在第一个位置的概率——Top1 Probability。ListNet定义物品$j$被排序在第一个位置的概率公式：
$$
\begin{aligned}P_{\mathbf s}(j)=\sum_{\pi(1)=j,\pi\in\Omega_n}P_{\mathbf s}(\pi)\end{aligned}
$$
​	我们希望分数越大的物品被排在第一个位置的概率越高，只要计算每一个物品被排在第一个位置的概率$P_{\mathbf s}(k),k=1,...,n$。但即便如此，根据公式计算概率也几乎不可能，因为直剩下的$n-1$个物品排序仍然有$n-1!$种可能性，文中的定理$6$则明确告诉我们可以通过如下式子计算$P_{\mathbf s}(j)$：
$$
P_{\mathbf s}(j)=\frac{\Phi(\mathbf s_j)}{\sum_{k=1}^n\Phi(\mathbf s_k)}
$$
​	此外，我们仍需确保$P_{\mathbf s}(j)$也是符合概率分布的，文中引理7：

> Top one probabilities  $P_{\mathbf s}(j)$ forms a probability distribution over the set of n objects.

​	通过Top1概率，给定一个真实标签的概率分布$\mathbf P_{\mathbf y}^{(i)}$和模型输出的概率分布$\mathbf P_{\mathbf z}^{(i)}$，我们就可以用一个度量分布的指标作为损失函数，这里笔者沿用论文中的符号，查询$q^{(i)}$对应的候选文档集合为$\mathbf d^{(i)}=\set{d^{(i)}_{1},...,d_{n^{(i)}}^{(i)}}$，查询$q^{(i)}$对应文档集合的人工标记相关性分数向量记作$\mathbf y^{(i)}=(y^{(i)}_{1},...,y^{(i)}_{n^{(i)}})$，模型预测的输出为$\mathbf z^{(i)}=(z^{(i)}_{1},...,z^{(i)}_{n^{(i)}})$，我们看一下不同标注方式下的ListNet模型的损失函数：

![image-20250327155924152](E:\Study\gitpro\knowledge-planet\NLP系列\assets\image-20250327155924152.png)

​	**方式一**即上图的左半部分，假设在标注阶段的每一个文档的相关性分数都是确切的，专家关注每一篇文档的得分，查询$q^{(i)}$标签的概率分布记作$\mathbf P_{\mathbf y}^{(i)}=(P_{y^{(i)}}(1),...,P_{y^{(i)}}(n))^{\top}$，模型输出的概率分布记作$\mathbf P_{\mathbf z}^{(i)}=(P_{z^{(i)}}(1),...,P_{z^{(i)}}(n))^{\top}$，前者是目标分布，后者是真实分布，我们可以找一个度量分布的函数作为损失函数，KL散度。若采用KL散度作为损失，则$\operatorname{D}_{KL}(\mathbf P_{\mathbf y}^{(i)}||\mathbf P_{\mathbf z}^{(i)})$表达如下：
$$
\begin{aligned}\operatorname{D}_{KL}(\mathbf P_{\mathbf y}^{(i)}||\mathbf P_{\mathbf z}^{(i)})&=\sum_{k=1}^{n^{(i)}} P_{y^{(i)}}\log \frac{P_{y^{(i)}}}{P_{z^{(i)}}}\\
&=C-\sum_{k=1}^{n^{(i)}} P_{y^{(i)}}\log {P_{z^{(i)}}}\\
&=C+H(\mathbf P_{\mathbf y}^{(i)},\mathbf P_{\mathbf z}^{(i)})\end{aligned}
$$
​	由于标签是固定的，即$C$一直不变，采用KL散度作为损失函数等价于用交叉熵作为损失函数。所以ListNet的损失函数就是Cross Entropy Loss。ListMLE[[x]]()则采用了一个更加直接的方式，以真实的标签顺序排列作为目标，以极大似然估计的思想设计损失函数。
$$
xxxx
$$
假设有一个文档$\set{1,2,3,4}$的分数列表$\mathbf s=\set{4,2,3,1}$，则排序$\pi=\langle1,3,2,4\rangle$，直接优化该排列序列出现的概率，利用公式$()$​：
$$
-\log P(\hat {\mathbf s})=-\log \frac{\exp (4)}{\exp (4)+\exp (2)+\exp (3)+\exp (1)}\frac{\exp (3)}{\exp (3)+\exp (2)+\exp (1)}\frac{\exp (2)}{\exp (2)+\exp (1)}
$$
​	然而，ListNet与ListMLE这类排序模型的优化目标与位置无关，用IR的衡量指标如$NDCG$来衡量排序好坏时有不一致的矛盾，如下案例[[x]](https://auai.org/uai2014/proceedings/individuals/164.pdf)能给出一个直观解释，假设真实的最优排序是$\mathbf y=(1,2,3,4,5)$，给定一个分数列表$\mathbf s_1=(\ln4,\ln5,\ln3,\ln2,\ln1)$和$\mathbf s_2=(\ln5,\ln4,\ln1,\ln2,\ln3)$，依据ListMLE的损失函数，则有：
$$
\begin{aligned}
L(f_1,\mathbf x,\mathbf y)&=-\log(\frac{4}{4+5+3+2+1}\cdot\frac{5}{5+3+2+1}\cdot\frac{3}{3+2+1}\cdot\frac{2}{2+1})\\&=1.213\\
L(f_2,\mathbf x,\mathbf y)&=-\log(\frac{5}{5+4+3+2+1}\cdot\frac{4}{4+3+2+1}\cdot\frac{1}{1+2+3}\cdot\frac{2}{2+3})\\&=1.213
\end{aligned}
$$
​	从NDCG指标来衡量则发现结果相反，即$NDCG@5(f_1,\mathbf x,\mathbf y)\lt NDCG@5(f_1,\mathbf x,\mathbf y)$:
$$
\begin{aligned}
 N D C G @ 5\left(f_{1}, \mathbf{x}, \mathbf{y}\right) &=  \frac{1}{N_{5}}\left(\left(2^{5}-1\right) \log \frac{1}{3}+\left(2^{4}-1\right) \log \frac{1}{2}+\left(2^{3}-1\right) \log \frac{1}{4}\right) \\
& +\frac{1}{N_{5}}\left(\left(2^{2}-1\right) \log \frac{1}{5}+\left(2^{1}-1\right) \log \frac{1}{6}\right), \\
&=xxxx \\
 N D C G @ 5\left(f_{2}, \mathbf{x}, \mathbf{y}\right) &= \frac{1}{N_{5}}\left(\left(2^{5}-1\right) \log \frac{1}{2}+\left(2^{4}-1\right) \log \frac{1}{3}+\left(2^{3}-1\right) \log \frac{1}{6}\right) \\
& +\frac{1}{N_{5}}\left(\left(2^{2}-1\right) \log \frac{1}{5}+\left(2^{1}-1\right) \log \frac{1}{4}\right),\\
&=xxxxx
\end{aligned}
$$
​	NDCG这样的评估指标反映用户会更关注排序靠前的结果，因此排序列表中若错误地排错了靠前的物品会比错误排序靠后的物品更加严重，而ListMLE则无法捕捉这样的位置信息。 

### 3.3.5  Bolztman Rank

​	Bolztman Rank的思想与ListNet也类似，定义一个排序概率，考虑给定分数$\mathbf s$下目标性能度量的期望值，并将该期望作为优化指标，受统计物理学中玻尔兹曼分布的启发，给定分数列表$\mathbf S^{(f)}=\set{s_1,...,s_m}$和排序$\mathbf R=\set{r_1,...,r_m}$，Boltzman Rank先定义了一个给定$\mathbf s$下$R$出现的能量作为：
$$
E(R|\mathbf S)=\frac{2}{m(m-1)}\sum_{r_j>r_k}g_q(r_j-r_k)(s_j-s_k)
$$
​	其中$g_q$可以是任意的符号保持函数，顾名思义，是指 **在输入值正负不变的情况下，输出值的正负也保持不变的函数**。换句话说，如果输入的两个值的相对大小关系是确定的，那么它们经过该函数变换后的相对大小关系仍然保持不变。如$g_q(x)=a_qx$，其他的函数还有仿射函数$f(x)=kx+b$，指数函数$e^x$。从公式中我们可以知道，当$s_j\gt\gt s_k$时，说明$R$与$\mathbf s$不太匹配，会获得一个较大的能量，当$s_j<<s_k$时，则会获得比较少的能量。而在物理系统中，**较低的能量通常表示更稳定的状态**，较高的能量表示系统处于更不稳定、不自然的状态，系统会容易发生改变，因此，较低的能量说明$R$与$\mathbf s$​排序关系较为一致，反之则不匹配。利用能量函数，我们现在可以通过指数化和归一化来定义文档排列上的条件玻尔兹曼分布：
$$
\begin{aligned}P(\mathbf R|\mathbf S)=\frac{1}{Z(\mathbf S)}\exp{(-E(\mathbf R|\mathbf S))} \\
Z(\mathbf S)=\sum_{\mathbf R} \exp{(-E(\mathbf R|\mathbf S))}\end{aligned}
$$
​	$\mathbf S$是神经网络的输出，即我们希望在给定网络预测$\mathbf S$的情况下，某个具体的排列$\mathbf R$出现的概率最大，但是$\mathbf R$出现的排列可能性是$m!$，因此分母项$\mathbf Z(\mathbf S)$只能近似计算。如SoftRank中近似计算$P(r_j=k)$，在SoftRank中文档$doc_j$被文档$doc_i$打败的概率为高斯分布：
$$
s_j\sim\mathcal N(\bar {s_i}-\bar{s_j},2\sigma_s^2)
$$
​	在Boltzman Rank中借用Bradley Terry模型定义文档$doc_j$被文档$doc_i$​打败的概率如下：
$$
\pi_{ij}=P(s_i>s_j)=\frac{\exp{(-k*s_i)}}{\exp{(-k*s_i)}+\exp{(-k*s_i)}}
$$
​	此外，Boltzman Rank认为打分函数$f$（神经网络）由两部分构成：（1）单点打分的函数$\phi$，无需考虑文档对之间的关系。（2）成对打分函数$\psi$。任何给定文档$d_j$的最终分数计算如下：
$$
f(d_j|q,D)=\phi(d_j)+\sum_{k,k\neq j}\psi(d_j,d_k)
$$
​	xxxx

### 3.3.6 Neural Sort&Neural NDCG

​	在上述的ListWise形式的排序中，由于$NDCG$​指标的计算关于神经网络的输出是一个不可导的操作，因此不可直接优化，可以通过函数近似替代的方式或者与位置无关的损失函数来优化网络，那有没有研究是找到一个离散的排序的可导近似呢？——NeuralSort就是一种“连续松弛”，是排序操作的可导近似。

​	Neural Sort的目标是通过反向传播的方式优化包含$\operatorname{sort}$算子的优化目标，即如下形式：
$$
\begin{aligned} L(\theta,\mathbf s)=f(P_{\mathbf z};\theta)\\
\text{where } \mathbf z=\operatorname{sort}(\mathbf s)\end{aligned}
$$
​	其中，$\mathbf s\in\mathbb R^{n}$是一个$n$元实值向量，$\mathbf z$是一个由$\mathbf s$排序后的置换向量，$P_{\mathbf z}$是一个置换矩阵。在上文中，笔者罗列了不可导算子的可导近似，其中$\operatorname{sort}$算子的可导近似是$\operatorname{Sinkhorn}$算子，Sinkhorn Operator 是一种 **将非可导的排序操作（如 permutation matrix）变成可导形式** 的方法，它常用于 **可微排序（differentiable sorting）** 或 **可微分配（differentiable assignment）** 的场景中。它的关键是将 **离散的置换矩阵（permutation matrix）** 近似为 **可导的双随机矩阵（doubly stochastic matrix）**。置换矩阵$P_{\mathbf z}$是一个特殊的方阵，用于对向量或矩阵进行排序，一个$n\times n$的矩阵$P$​称之为置换矩阵，当且仅当：

1. 每个元素$P_{ij}\in\set{0,1}$。
2. 每行与每列之和都为$1$:

$$
\sum_{j=1}^nP_{ij}=1\text{	}\forall i,\text{}\sum_{i=1}^nP_{ij}=1\text{ }\forall j
$$

​	给定一个$n$维的置换向量$\mathbf z=(z_1,z_2\cdots z_n)^{\top}\in\mathbb R^n$，$z_i\in\set{1,2\cdots n}$且两两不同，对应的置换矩阵$P_{\mathbf z}$满足：
$$
P_{\mathbf z}=\left\{\begin{array}{ll}
1 & \text { if } j=z_{i} \\
0 & \text { otherwise }
\end{array}\right.
$$
​	假设一个输入向量$\mathbf s=(9,1,5,2)^{\top}$，且**定义**$\operatorname{sort}:\mathbb R^n\rightarrow\mathcal Z_n$算子是一个将$n$维实值向量输入映射到一个降序排列的置换向量的操作，则$\mathbf s$经过$\operatorname{sort}$作用后对应的置换向量是$\mathbf z=\operatorname{sort}({\mathbf s})=(1,4,2,3)^{\top}$，即第$1$个元素排第一位，第$2$个元素排第$4$位，第$3$个元素排第$2$位，第$4$个元素排第$3$位，置换向量对应的置换矩阵$P_{\mathbf z}$：
$$
P_{\mathbf z}=\left[\begin{array}{llll}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
0 & 1 & 0 & 0
\end{array}\right],P_{\mathbf z}\cdot \mathbf s=(9,5,2,1)^{\top}
$$

> [!IMPORTANT]
>
> 给定一个输入向量$\mathbf s$，其对应一个置换向量$\mathbf z$和置换矩阵$P$，则置换矩阵$P$的元素$P[i,j]=1$时的含义是输入向量$\mathbf s_j$一定是第$i$大的。如上案例，$P[1,1]=1$，$\mathbf s_1=9$是第$1$大的。$P[2,3]=1$，$\mathbf s_3= 5$是第$2$大的，$P[3,4]=1$，$\mathbf s_4=2$是第$3$大的。即如果输入$\mathbf s_j$是第$i$大的，则矩阵第$j$列第$i$行为$1$。

​	给定任意$\mathbf s$我们需要先找其和到$P_{\operatorname{sort}(s)}$明确的数学表达关系，我们知道的是$P_{\operatorname{sort}(s)}[i,j]=1$一定代表$\mathbf s_j$第$i$大，第$1$大可以用$\max$，最小可以用$\min$，但是第$i$大这个该如何通过数学公式描述？我们需要借用这样一个引理：

> For an input vector $\mathbf s = [s_1, s_2, \cdots,s_n] ^{\top}$ that is sorted as $s[1] ≥ s[2] ≥\cdots≥s[n]$ , we have the sum of the $k$-largest elements given as:
> $$
> \sum_{i=1}^{k} \mathbf s_{[i]}=\min _{\lambda \in\left\{s_{1}, s_{2}, \ldots, s_{n}\right\}} \lambda k+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda, 0\right)
> $$

​	这个引理的证明很简单：
$$
\begin{aligned} \sum_{i=1}^{k} \mathbf s_{[i]}&=\sum_{i=1}^{k} \mathbf s_{[i]}-\lambda+\lambda k \\
&\leq\lambda k+\sum_{i=1}^{k} \mathbf \max(s_{[i]}-\lambda,0)\\
&\leq \lambda k+\sum_{i=1}^{n} \mathbf \max(s_i-\lambda,0)\end{aligned}\tag{3-x}
$$
​	当$\lambda$比$\mathbf s_{[k]}$大时，$\max$算子是生效的，当$\lambda$等于$\mathbf s_{[k]}$时，$\max$算子不生效，有：
$$
\begin{aligned} \lambda k+\sum_{i=1}^{n} \mathbf \max(s_{i}-\lambda,0)&=\lambda k +\sum_{i=1}^{n} \mathbf s_{i}-\lambda\\
&=k\mathbf s_{[k]}+\sum_{i=1}^{k} \mathbf s_{[i]}-s_{[k]}\\
&=\sum_{i=1}^{k} \mathbf s_{[i]}\end{aligned} \tag{3-x}
$$
​	更具体地，等式$(3-x)$成立的条件是$\mathbf s_{[k]}\geq \lambda \geq \mathbf s_{[k+1]}$。通过控制$\lambda$的大小，我们可以得到$\mathbf s$的前$k$个最大值之和。而第$k$大的值$\mathbf s_{[k]}$表明上可以通过下式得到：
$$
\begin{aligned}\mathbf s_{[i]}&=\sum_{i=1}^{k} \mathbf s_{[i]}-\sum_{i=1}^{k-1} \mathbf s_{[i]}\\
&=\min _{\lambda \in\left\{s_{1}, s_{2}, \ldots, s_{n}\right\}} \lambda k+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda, 0\right)\\
&-\big(\min _{\lambda'\in\left\{s_{1}, s_{2}, \ldots, s_{n}\right\}} \lambda'(k-1)+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda' 0\right) \big)\\
&=\min_{\lambda}F_{k}(\lambda)-\min_{\lambda'}F_{k-1}(\lambda')\end{aligned}
$$
​	我们会发现这是一个差分$\min$运算，会让优化问题变得复杂，不便于接下来的推导。为此，我们必须想一种方式让优化目标只有一个$\min$。等式$\sum_{i=1}^{k} \mathbf s_{[i]}=\lambda k+\sum_{i=1}^{n} \mathbf \max(s_{i}-\lambda,0)$的成立条件是$\mathbf s_{[k]}\geq \lambda \geq \mathbf s_{[k+1]}$，我们思考是否可以再构造一个优化目标使得$\mathbf s_{[k-1]}\geq \lambda \geq \mathbf s_{[k]}，$这样就可以通过夹逼的方式强迫$\lambda=\mathbf s_{[k]}$，优化$\lambda$使得目标最小就得到了最终的$\mathbf s_{[k]}$。换个角度想，前$k$大其实等价于后$n-k+1$小，如果将$\mathbf s$取负即令$\mathbf t=-\mathbf s$，则$\mathbf t$的前$n-k+1$大等价于$\mathbf s$的后$n-k+1$小等价于$\mathbf s$的前$k$​大。如下是一个直观的例子：

![image-20250620110756812](assets\image-20250620110756812.png)

​	因此可以同样使用引理2把$\mathbf t$的前$n-k+1$大的和写成：
$$
\begin{aligned} \sum_{i=1}^{n-k+1}\mathbf t_{[i]}&=\min_{\lambda\in\mathbf t=-\mathbf s}\big[ \lambda(n-k+1)+\sum_{i=1}^{n}\max(\mathbf t_i-\lambda,0)\big]\\
&st.\mathbf t_{[n-k+1]}\geq \lambda \geq \mathbf t_{[n-k+2]}\end{aligned}
$$
​	再令$\lambda=-\lambda$​，则有：
$$
\begin{aligned} \sum_{i=1}^{n-k+1}\mathbf t_{[i]}&=\min_{\lambda\in\mathbf -t=\mathbf s}\big[ -\lambda(n-k+1)+\sum_{i=1}^{n}\max(\mathbf \lambda-s_i,0)\big]\\
&st.\mathbf t_{[n-k+1]}\geq -\lambda \geq \mathbf t_{[n-k+2]}\equiv \mathbf s_{[k]}\leq\lambda \leq \mathbf s_{[k-1]}\end{aligned}
$$
​	但本质上我们是想求得$\lambda$，因此我们合并两个$\arg\min_{\lambda}$使得$ \mathbf s_{[k]}\leq\lambda \leq \mathbf s_{[k-1]}$与$\mathbf s_{[k]}\geq \lambda \geq \mathbf s_{[k+1]}$同时成立，即$\lambda^*=\mathbf s_{[k]}$，最终合并两式和，有关于$\lambda$的优化目标为：
$$
\begin{aligned} \lambda^*=\mathbf s_{[k]}&=\arg \min _{\lambda \in \mathbf{s}}\left(\lambda k+\sum_{i=1}^{n} \max \left(\mathbf s_{i}-\lambda, 0\right)+\lambda(k-1-n)+\sum_{i=1}^{n} \max \left(\lambda-\mathbf s_{i}, 0\right)\right) \\
&=\arg \min _{\lambda\in\mathbf s}\lambda(2k-1-n)+\sum_{i=1}^{n}\max(\mathbf s_i-\lambda,0)+\max(\mathbf \lambda-\mathbf s_i,0)\\
&=\arg \min _{\lambda\in\mathbf s}\lambda(2k-1-n)+\sum_{i=1}^{n}|\mathbf s_i-\lambda|\\
&=\arg \max _{\lambda\in\mathbf s}\lambda(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\lambda|\end{aligned}
$$

​	我们看这个等式，并把它展开：
$$
\begin{aligned} \lambda^*&=\arg \max _{\lambda\in\mathbf s}\lambda(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\lambda|\\
&=\arg \max _{\lambda\in\mathbf s}\begin{pmatrix}
\mathbf s_1(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_1| \\
\mathbf s_2(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_2|\\
\vdots \\
\mathbf s_n(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_n|
\end{pmatrix}\\
&=\arg \max _{\lambda\in\mathbf s}[(n+1-2k)\mathbf s-\mathbf A_i\mathbb 1]\end{aligned}
$$
​	该式子的含义是遍历$\lambda \in \mathbf s$使得$\lambda(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\lambda|$达到最大，如果抛去$\begin{aligned}\arg\max_{\lambda}\end{aligned}$的下角标$\lambda$，则：
$$
\begin{aligned} &\arg \max \begin{pmatrix}
\mathbf s_1(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_1| \\
\mathbf s_2(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_2|\\
\vdots \\
\mathbf s_n(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_n|
\end{pmatrix}\\
&=\arg \max [(n+1-2k)\mathbf s-\mathbf A_i\mathbb 1]\end{aligned}
$$
​	作用是找到该向量中分量值最大元素对应的索引。若结果是索引$i$，则说明$\mathbf s_i=\mathbf s_{[k]}$，理清了这层关系，我们开始构造置换矩阵$P$。我们回顾一下之前所述：置换矩阵$P$的元素$P[i,j]=1$时的含义是输入向量的分量$\mathbf s_j$一定是第$i$大的，且置换矩阵每一行每一列和为$1$，因此构造置换矩阵的列从左往右看就需要知道$\mathbf s_1,\mathbf s_2,...,\mathbf s_n$分别第几大。构造一个$n\times n$的置换矩阵时，逻辑如下，我们从行开始构造，选定第$i$行，先构造左边第一列，那么我们就判断$\mathbf s_1=\mathbf s_{[k]}$时的$k$是多少，则$P[k,1]=1$，第$k$行之外的其他行$P[i,1]=0,i\neq k$​。那$\mathbf s_i$第几大我们要一一判断，从$k=1,2,...,n$，对于第$1$列而言，只要：
$$
\begin{aligned} \arg \max \begin{pmatrix}
\mathbf s_1(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_1| \\
\mathbf s_2(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_2|\\
\vdots \\
\mathbf s_n(n+1-2k)-\sum_{i=1}^{n}|\mathbf s_i-\mathbf s_n|
\end{pmatrix}=1，\text{for k in }1,2,...,n\end{aligned}
$$
​	便能说明$\mathbf s_1=\mathbf s_{[k]}$，因此，对于第一列我们可以写成：
$$
\begin{aligned}P[i,1]==\left\{\begin{array}{ll}
1 & \text { if } 1=\arg \max [(n+1-2i)\mathbf s-\mathbf A_i\mathbb 1] \\
0 & \text { otherwise }
\end{array}\right.\end{aligned}
$$
​	同理，第二列我们可以写成：
$$
\begin{aligned}P[i,2]==\left\{\begin{array}{ll}
1 & \text { if } 2=\arg \max [(n+1-2i)\mathbf s-\mathbf A_i\mathbb 1] \\
0 & \text { otherwise }
\end{array}\right.\end{aligned}
$$
​	以此类推，第$j$列我们可以写成，这便是一个统一的形式：
$$
\begin{aligned}P[i,j]==\left\{\begin{array}{ll}
1 & \text { if } j=\arg \max [(n+1-2i)\mathbf s-\mathbf A_i\mathbb 1] \\
0 & \text { otherwise }
\end{array}\right.\end{aligned}
$$
​	当然，$\arg\max$算子是不可导的，$P$也是不可导的，$P$的每一行是一个$\operatorname{one-hot}$向量，我们找一个可导近似来近似这个$\operatorname{one-hot}$向量，最简单的，可以用算子$\operatorname{softmax with temperature}$近似，即：
$$
\lim _{\tau \rightarrow 0^{+}} \widehat{P}_{\operatorname{sort}(\mathbf{s})}[i,:](\tau)=P_{\operatorname{sort}(\mathbf{s})}[i,:] \quad \forall i \in\{1,2, \ldots, n\}
$$
​	温度系数$\tau$越小则该分布越接$\operatorname{one-hot}$向量，$\tau$越大则越接近平稳分布。接下来我们再思考如何构造优化目标，在很多排序任务中，**目标排序是唯一的**，比如给定一个数字列表$[3,2, 1, 4]$，我们知道升序结果是 $[1, 2,3, 4]$。但在**模型学习排序函数**时，并不是直接输出这个固定结果，而是输出一个分数向量，然后间接地产生排序。一个分数向量可能对应多个潜在的排列结果，如果只用一个排列，那么信号太稀疏了，不够全面，网络在学习时可能会记住某个输入对应的一种排列情况而不是学到通用规律。而列举分数向量$\mathbf s$​所有排列情况显然也不现实，因此我们需要采样，即从所有可能性结果中选出一部分具有代表性的排列，然后评估这些排序的表现，用这些反馈去优化分数向量生成器，从而提高神经网络的泛化能力。

> [!NOTE]
>
> 不采样就对应了确定性Neural Sort，采样并重参数化就对应了随机Neural Sort。

​	直接采样排列$z\sim \operatorname{Plackett-Luce}(\mathbf s)$这个操作也是不可导的，最常见的解决方式之一是重参数化，将离散采样过程转化为**确定性函数+噪声扰动**，使梯度能通过连续变量传递而$\operatorname{Gumbel Softmax}$就是代表性的重参数化方法。重参数化方法是处理如下优化目标的一种方法：
$$
\begin{aligned} L_{\theta}=\mathbb E_{z\sim P_{\theta}(z)}\big[ f(z)\big]\end{aligned}
$$
​	由于采样操作不可导，因此没有办法写一个精确的$L_{\theta}$，而$z$从$p_{\theta}(z)$中采样会失去关于参数$\theta$​的梯度信息，重参数方法则将采样变化成“固定随机数 + 可导变换”的方式使得反向传播可以用于训练目标涉及到采样操作的神经网络。其具体数学形式如下：设一个随机变量 $z∼p(z∣θ)z$，其中 $\theta$ 是模型参数，例如均值 $\mu$、标准差 $\sigma$等，目标关于参数的梯度为：
$$
\nabla_{\theta}\mathbb E_{z\sim p_{\theta}(z)}\big[ f(z)\big]
$$
​	引入一个可以重参数化的随机变量$\epsilon\sim p(\epsilon)$，使得：
$$
z=g_{\theta}(\epsilon)
$$
​	其中，$g$是一个确定性的可导分布，$\epsilon$的分布和$\theta$无关，则有：
$$
\begin{aligned}\nabla_{\theta}\mathbb E_{z\sim p_{\theta}(z)}\big[ f(z)\big]&=\nabla_{\theta}\mathbb E_{\epsilon\sim p(\epsilon)}\big[ f(g_{\theta}(\epsilon))\big]\\
&=\mathbb E_{\epsilon\sim p(\epsilon)}\big[ \nabla_{\theta}f(g_{\theta}(\epsilon))\big]\end{aligned}
$$
​	以常见的高斯分布为例，假设随机变量$z\sim \mathcal N(z;\mu_{\theta},\sigma_{\theta})$，令$z=\sigma_{\theta}\epsilon+\mu_{\theta}$，其中$\epsilon \sim \mathcal N(\epsilon;0,1)$，则有：
$$
\begin{aligned}\nabla_{\theta}\mathbb E_{z\sim p_{\theta}(z)}\big[ f(z)\big]&=\nabla_{\theta}\mathbb E_{\epsilon\sim \mathcal N(\epsilon;0,1)}\big[ f(\sigma_{\theta}\epsilon+\mu_{\theta})\big]\\
&=\mathbb E_{\epsilon\sim \mathcal N(\epsilon;0,1)}\big[ \nabla_{\theta}f(\sigma_{\theta}\epsilon+\mu_{\theta})\big]\end{aligned}
$$
​	在离散情况，将随机变量$z$用$y$代替，以从类别分布中采样为例：
$$
\mathbf p_{\theta}=[\mathbf p_{\theta1},\mathbf p_{\theta2},...,\mathbf p_{\theta k}]
$$
​	现在$y\sim \operatorname{Categorical(\mathbf p_{\theta})}$，我们需要找一个确定性的可导分布$y=g_{\theta}(\epsilon)$使得采样的随机性转移到随机变量$\epsilon$上，而$\operatorname{Gumbel Max}$提供了一种从类别分布中采样的方法（本节暂不对此进行深入的原理解析）：
$$
\begin{aligned} \arg\max_i &\big(\log \mathbf p_{\theta i} -\log(-\log\epsilon_i)\big)^k \\
\epsilon_i &\sim U[0,1]\end{aligned}
$$
​	即先从均匀分布中先采样得到$k$个随机数，然后再计算每一个$\big(\log \mathbf p_{\theta i} -\log(-\log\epsilon_i)\big)^k$，找到最大值对应的索引：
$$
\arg\max_i\begin{pmatrix}
\big(\log \mathbf p_{\theta 1} -\log(-\log\epsilon_1)\big)^k\\ 
\big(\log \mathbf p_{\theta 2} -\log(-\log\epsilon_2)\big)^k\\
\vdots \\
\big(\log \mathbf p_{\theta k} -\log(-\log\epsilon_k)\big)^k
\end{pmatrix}
$$
​	前面讲过$\arg\max$算子不可导，通常用算子$\operatorname{softmax with temperature}$近似，因此得到了$\operatorname{Gumbel Max}$的可导近似$\operatorname{Gumbel Softmax}$，回到主题上，值得注意的是我们并不是从类别分布中采样，我们是重参数化$\mathbf z\sim \operatorname{Plackett-Luce}(\mathbf s)$采样，因此不是直接套用$\operatorname{Gumbel Softmax}$，而是采用$\operatorname{Gumbel Sinkhorn}$，具体流程如下：



​	







# 4.Ranking skills in Direct Preference Optimization

## 4.1 Preference Feedback

### 4.1.1 Pair-Wise Feedback

​	成对反馈侧重于比较成对问答的偏序关系，即$(x_i,y^{(i)}_j)$与$(x_i,y^{(i)}_k)$，从而判断回答的相对好坏。DPO在RLHF的理论框架基础上，利用成对偏好数据实现了这一范式，从而拟合了隐式奖励模型[[x]]()。

### 4.1.2 List-Wise Feedback

​	



## 4.2 Preference Granularity



# 5.参考文献

[[1]]()

[[2]]()

[[3]]()

[[4]]()

[[5]]()

[[6]Grover,Wang,Zweig,et al.Stochastic Optimization of Sorting Networks via Continuous Relexations[J].International Conference on Learning Representations,2019.](https://arxiv.org/abs/1903.08850)

[[x]A Survey of Direct Preference Optimization](https://arxiv.org/abs/2503.11701)

[[x]**LLM Alignment as Retriever Optimization: An Information Retrieval Perspective**](https://arxiv.org/abs/2502.03699)
