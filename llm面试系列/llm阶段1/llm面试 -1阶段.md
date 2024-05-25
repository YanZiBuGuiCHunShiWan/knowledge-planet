### 阶段一

$\text{1.谈谈ChatGPT的优缺点}$

​	1.资源耗用角度：ChatGPT在预训练阶段需要海量的文本数据，在对齐阶段首先需要高质量的人类标注语料告诉模型该如何输出人类想要的答案，其次还需要通过人类反馈强化学习算法让模型学习到人类的偏好，提升输出的质量。这些过程及其消耗人力资源和计算资源，

​    2.使用角度：ChatGPT擅长于处理一些繁琐、但不会特别有深度、专业性的问题，帮助提高日常办公效率。比如写一篇文案、写一段简单的数据处理的代码、总结一段文本等等。但是ChatGPT其生成的内容受prompt的影响有可能包含政治敏感、违法犯罪等内容，还需要人们去校验与后处理。且输出质量高度依赖于用户的输入质量，模糊或不明确的问题可能没法引导GPT生成令人满意的回答。此外，其内部的知识在没有调用外部向量库的情况下没法做到实时更新，即知识有时效性；其还有幻觉现象，会一本正经地胡说八道，当外行人请教其一个专业点的知识时，根本不知道其说的到底可不可靠。

$\text{2.请简述下Transformer基本流程}$

​	Transformer由一个编码器和一个解码器构成，编码器是堆叠的transformer block构成。首先是最底层的输入，文本先会被分词成token序列，$\text{token1,token2,...,tokenk}$，然后经过$\text{embedding layer}$得到$(\mathbf e_1,\mathbf e_2,...,\mathbf e_k)$，此外每一个token还有其对应的位置编码帮助模型编码位置信息，记作$(p_1,p_2,...,p_k)$，二者相加作记作$\mathbf X=(\mathbf x_1,\mathbf x_2,....,\mathbf x_k)$为最底层transformer block的输入。

​	编码器中的transformer block由多头自注意力机制$\text{multihead self-attention}$（编码器的自注意力机制没有掩码）和$\text{add \&layer normalization}$还有$\text{feed-forward neural network}$构成。多头自注意力机制帮助模型关注不同的语义信息，编码器的输出是序列化的语义向量。在多头自注意力机制中，有三个参数矩阵$W^Q,W^K,W^V$，$\mathbf X$分别乘以三个参数矩阵得到$Q,K,V$,然后reshape成[Batch_size,seq_len,num_heads,head_dim]的形状，通过公式$softmax(\frac{QK^T}{\sqrt d})$得到每一个头的注意力矩阵，再由注意力矩阵乘以对应的value得到融合上下文信息的语义向量，然后再将得到的多个语义向量reshape成[Batchsize,seqlen,hidden_dim]，最后再经过线性层映射后进入PosistionwiseFeedFowrad层。同时有残差链接机制，缓解梯度爆炸或消失问题，如此重复计算N个Block就得到了编码器输出的语义向量。

​    解码器同样也是由transformer block堆叠而成，其最底层的输入是target token（目标语言）的词嵌入和位置编码，和编码器不同之处在于，其注意力机制是交叉注意力机制，即注意力机制的query来自解码器的语义表征，而key和value来自于编码器输出的语义表征。并有注意力机制掩码将对角线右上方的区域置为0，让模型不能获取未来时刻的信息，防止信息泄露。

$\text{3.为什么基于Transformer的架构需要多头注意力机制?}$

​	直观上讲，多头的注意力有助于网络捕捉到更丰富的特征/信息，就像CNN中不同的卷积核可以关注到不同的特征。对于Transformer中的自注意力机制，有Attention矩阵等于$softmax(\frac{QK^T}{\sqrt d})$，$QK^T \in \mathbf R^{k \times k}$，方针对角线上的元素是token和自己的相似度，如果方阵对角线上的元素过大，那么模型在对当前位置的信息进行编码时过度将注意力集中于自身，而忽略其他位置。 而采用多头注意力机制，可以克服模型在对当前位置信息编码时将注意力过度集中自身的问题，使得整个模型表达能力更强，提升对于注意力权重的合理分配。

$\text{4.编码器，解码器，编解码LLM模型之间的区别是什么?}$

​	Encoder主要任务是从文本中提取特征和上下文信息，以BERT为例，编码器通过上下文来预测被掩码掉的token来学习文本语义表征。此外，编码器通常不直接产生人类可读的输出，而是生成序列化、连续数值的语义向量，这需要进一步的处理才能完成文本分类、实体识别等其他NLP任务，解码器LLM的注意力矩阵不会被掩码掉任何区域。

  Decoder在NLP任务中主要负责生成响应输出，在机器翻译、摘要生成等任务中基于编码器提高的语义信息生成人类可理解的文本，解码器可以兼顾自然语言理解、自然语言生成任务。

​	对于编解码LLM而言，虽然其能够兼顾自然语言理解和生成，但它的性能极大地依赖于编码器提供的信息质量。如果编码器未能准确理解输入数据，解码器生成的文本可能不准确或不相关。

  

$\text{5.你能解释在语言模型中强化学习的概念吗?它如何应用于ChatGPT?}$

​		强化学习的最终目的是让智能体在与环境的交互中做出最优的决策，让累计奖励最大化。而在大语言模型训练中，强化学习则是通过奖励机制来指导模型的学习，让模型更好地理解和输出自然语言。

​		既然是强化学习，那么肯定得有学习策略(Policy)，在RLHF中的策略是两个模型（以PPO为例），分别是Actor（演员）和Critic（评论家）,Actor是负责决策执行动作的，而Critic则总结演员的表现。而演员就是我们最终想要训练出来的大模型，他根据上文来预测下一个token的概率分布，评论家则根据一段上文来计算下一个token的收益。在ChatGPT中，我们可以将强化学习分成三个步骤看。分别是采样、奖励、学习。

​		$\text{1.采样：}$采样就是从一堆prompt池中选择一些prompt，然后模型基于这些prompt推理得到token的概率分布进行token采样得到最后的response，在这一步，我们不只需要采样到的response，采样这个环节可以理解为学生在黑板上答题，学生只留下了答案，还未得到老师的反馈和指导。

​		$\text{2.反馈：}$反馈就相当于老师检查学生答案的过程，会给学生的回答(response)进行打分，即有一个奖励模型(Reward Model)用于衡量prompt和Actor模型输出的response的匹配程度，prompt和response会拼起来给奖励模型打分得到一个标量score。但这个分数不是最终的奖励，因为score只能说明结果的好坏不能说明过程的合理性，那怎样的过程是合理的？1.循规蹈矩至少不会太差，即和SFT阶段已经训练好的模型（Reference Policy）差不多，因此按照这个规矩来可以得到少量奖励 2. 给予最终的奖励前，对Actor的标新立异给予少量的惩罚（认为他在挑战权威）。通过KL散度可以衡量两个分布间的相似度，我们希望Actor的策略和Reference 相差Policy不要太大，因此通过KL散度来约束Actor策略。反馈环节就是老师在检查学生的答案并给出评价，学生们就可以了解他们的表现如何从而进行学习和改进。模型也需要通过学习阶段来提升其生成内容的质量。

​		$\text{3.学习：}$ **学习**就是学生根据反馈总结得失并自我改进的过程，或者说是强化优势动作的过程。而优势就是**实际获得的收益超出预期的程度**。举个例子：小明数学其中考了80分，数学老师预估他在期末考试大约也在80分左右，但是小明期末考了120分，小明超出老师预期的程度为$120-80=40$，所以相对于预期获得了40分的优势。所以$\text{优势=实际收益-预期收益}$。

​		对于语言模型而言，生成第 $i $个 token 的实际收益就是：从生成第 $i $个 token 开始到生成第$N $个 token 为止，所能获得的所有奖励的总和。$\begin{aligned} \operatorname{return}[i]=\sum_{j=i}^{N} \operatorname{reward}[j]=\operatorname{reward}[i]+\ldots+\operatorname{reward}[N]\end{aligned}$。（不考虑贴现和广义优势估计），我们将Actor 生成第$i$个token获得的优势定义为：$\begin{aligned} \operatorname{a[i]}=\operatorname{return[i]}-\operatorname{values[i]}\end{aligned}$，其中$\operatorname{values[i]}$是Critic预测的收益，而PPO的学习阶段就是**强化优势动作**，在语言模型中，根据上下文生成一个 token 就是所谓的“动作”，**强化优势动作**就是如果$p(token|context)$获得的优势很高，那么就应当增加生成该token的概率。我们可以设计一个损失函数，通过优化损失函数来实现优势动作强化，比如：

​															$\begin{aligned}\text { actor\_loss }=-\frac{1}{N} \sum_{i=1}^{N} a[i] \times \frac{p(\operatorname{token}[\mathrm{i}] \mid \text { context })}{p_{\text {old }}(\operatorname{token}[\mathrm{i}] \mid \text { context })}\end{aligned}$

​		此外，评论演员的评论家也需要学习，它会预测Actor生成response时每一个token的预期收益，我们希望它预测的越准越好，即预期收益与真实的回报差距越小越好，因此我们可以设计如下损失：

​                                                            $\begin{aligned} \text { critic } \_ \text {loss }=\frac{1}{2 N} \sum_{i=1}^{N}(\operatorname{values}[i]-\operatorname{returns}[i])^{2}\end{aligned}$

​		最终的损失就是$\text { actor\_loss }+\beta \times \text { critic\_loss }$。通过这样的方式，可以使得语言模型不断强化自身的优势动作，让输出质量更加符合人类意图，更加安全可靠。



6.在GPT模型中，什么是温度系数?

​		在GPT推理的过程中，为了结果的多样性，通常会选择采样的方式从词表分布中选择一个词作为当前时刻的预测结果。而温度系数$\tau$是控制采样分布的超参数。假设Decoder-only Transformer的输出对应的logits是$(h_1,h_2,...,h_{|V|})$，那么经过$\text{Softmax}$以后的概率分布则可以记作$(\begin{aligned}\frac{e^{h_1}}{\sum_{i}e^{h_i}},\frac{e^{h_2}}{\sum_{i}e^{h_i}},...,\frac{e^{h_{|V|}}}{\sum_{i}e^{h_{|V|}}})\end{aligned}$，而加上温度系数$\tau$以后概率分布会变成$(\begin{aligned}\frac{e^{h_1/\tau}}{\sum_{i}e^{h_i/\tau}},\frac{e^{h_2/\tau}}{\sum_{i}e^{h_i/\tau}},...,\frac{e^{h_1/\tau}}{\sum_{i}e^{h_{|V|}/\tau}})\end{aligned}$。以极限的思想去分析，假设第$k$个位置的token概率$\frac{e^{h_k/\tau}}{\sum_i e^{h_i/\tau}}$是最大的，当$\tau \rightarrow 0^{+}$时$\begin{aligned}\lim_{\tau \rightarrow 0^{+}}\frac{e^{h_k/\tau}}{\sum_{i}e^{h_i/\tau}}=\frac{1}{1+\sum_{i\not = k}e^{(h_i-h_k)/\tau}}\end{aligned}$，因为$h_k$大于所有的$h_i$，且$\tau \rightarrow 0^{+}$，故有$\begin{aligned}\lim_{\tau \rightarrow 0^+} e^{(h_i-h_k)/\tau}=0 \end{aligned}$，$\begin{aligned}\frac{1}{1+\sum_{i\not = k}e^{(h_i-h_k)/\tau}}=1\end{aligned}$，说明原本概率大的token在$\tau$的作用下分布更加尖锐了，因此$\tau \rightarrow 0^{+}$时，原本概率较大的那些token更加容易被采样到，生成的文本就没那么多变数，少一点创造性。

​		同理，$\tau \rightarrow \infin$时，有$\lim_{\tau \rightarrow \infin}e^{h_i/\tau}\rightarrow 0,i=1,2...,|V|$，原本的概率分布会变得非常的扁平，所有的token分布在$\tau$的作用下几乎都一样了，没啥显著区别，因此$\tau$越大token分布的差异就越小，这个时候会增加采样到原本分布概率低的那些token的概率，因此生成的文本更具创造性。



$\text{7.什么是旋转位置编码(ROPE)?}$

​	    对于Transformer而言纯靠自注意力机制没有融合token的位置信息，所以需要位置编码补充token的位置信息。而加入位置编码可以有两种方式：

1. 从输入层面改造，$\mathbf {x_i}$在经过$W^Q,W^K,W^V$之前加上位置编码，可以是一个自定义的绝对位置编码也可以是可学习的位置编码。
2. 从注意力层面改造，在$\mathbf x_i$经过$W^Q,W^K,W^V$作用时融入位置信息。

​		而位置编码应当考虑第$i$个位置和第$j$个位置的相对的距离信息，从注意力层改造，那么一个大致思路是$Q_iK_j^T=f(\mathbf x_i,\mathbf x_j,i-j)$，现在要找到函数$f$。对于一个矩阵$X$而言，右乘一个矩阵$\mathbf A$就是对$X$的个行向量施加变换，而我们知道$\begin{aligned} R(\theta)=\begin{pmatrix} \cos {\theta} & \sin {\theta} \\ -\sin {\theta} & \cos {\theta} \end{pmatrix}\end{aligned}$是一个旋转变换，一个行向量$\mathbf x_i$右乘以$R(\theta)$相当于这个行向量逆时针旋转角度$\theta$，记一个二维行向量$\mathbf x=(x_1,x_2)$，用极坐标表示就是$\mathbf x=\rho(\cos \phi,\sin\phi)$，具体证明如下。

​																$\begin{aligned}\mathbf x R(\theta)&=\rho(\cos \phi, \sin \phi)\left(\begin{array}{cc}
\cos \theta & \sin \theta \\
-\sin \theta & \cos \theta
\end{array}\right) \\
&=\rho(\cos \phi \cos \theta-\sin \phi \sin \theta, \cos \phi \sin \theta+\sin \phi \cos \theta) \\
&=\rho(\cos (\phi+\theta), \sin (\phi+\theta))
\end{aligned}$

​		此外，$R(\theta)$的性质还有两点：

1. $R(\theta)^T=R(-\theta)$      
1. $R(\theta_1)R(\theta_2)=R(\theta1+\theta_2)$

以上是二维向量在二维空间中的旋转，对于高纬空间而言，可以先假设空间是偶数维的，那么有：

- $\Theta= (\theta_1,\theta_2,...,\theta_{D/2})$
- $\begin{aligned} R(\Theta) =  \left(\begin{array}{ccccccc}
  \cos \theta_{1} & \sin \theta_{1} & 0 & 0 & 0 & 0 & 0 \\
  -\sin \theta_{1} & \cos \theta_{1} & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & \cos \theta_{2} & \sin \theta_{2} & 0 & 0 & 0 \\
  0 & 0 & -\sin \theta_{2} & \cos \theta_{2} & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & \cdots & 0 & 0 \\
  0 & 0 & 0 & 0 & \cdots & \cos \theta_{D / 2} & \sin \theta_{D / 2} \\
  0 & 0 & 0 & 0 & \cdots & -\sin \theta_{D / 2} & \cos \theta_{D / 2}
  \end{array}\right) = \left(\begin{array}{cccc}
  R\left(\theta_{1}\right) & 0 & 0 & 0 \\
  0 & R\left(\theta_{2}\right) & 0 & 0 \\
  0 & 0 & \cdots & 0 \\
  0 & 0 & 0 & R\left(\theta_{D / 2}\right)
  \end{array}\right)\end{aligned}$​ 
- $\begin{aligned} R(\Theta) = \hat R(\theta_1) \hat R(\theta_{2}) ....\hat R(\theta_{D/2}) \end{aligned}$，$\begin{aligned} \hat R(\theta_k)=\left(\begin{array}{ccccc}
  1 & 0 & 0 & 0 & 0 \\
  0 & \ddots & 0 & 0 & 0 \\
  0 & 0 & R\left(\theta_{k}\right) & 0 & 0 \\
  0 & 0 & 0 & \ddots & 0 \\
  0 & 0 & 0 & 0 & 1
  \end{array}\right) \end{aligned}$
- 给定一个高维矩阵$\mathbf X=(\mathbf X_1,\mathbf X_2,...,\mathbf X_{D/2})$，右乘以$R(\Theta)$那么有：

​																				$\begin{aligned}\mathbf XR(\Theta)= \left(\begin{array}{cccc}\mathbf X_1R\left(\theta_{1}\right) & 0 & 0 & 0 \\
0 & \mathbf X_2R\left(\theta_{2}\right) & 0 & 0 \\
0 & 0 & \cdots & 0 \\
0 & 0 & 0 & \mathbf X_{D/2}R\left(\theta_{D / 2}\right)
\end{array}\right)=\mathbf X_1\hat R(\theta_1) \mathbf X_2\hat R(\theta_{2}) ....\mathbf X_{D/2}\hat R(\theta_{D/2}) \end{aligned}$

​		物理含义是不同的矩阵在独立的二维子空间上做不同角度的旋转。

​		介绍完了大致的知识，那么现在来讲一下RoPE的动机：我们希望$Q_iK_j^T=f(\mathbf x_i,\mathbf x_j,i-j)$ ，先考虑简单的二维情形：假设$Q_i,K_j$是二维向量，$i,j$分别是位置，$\eta_i,\eta_j$分别是$Q_i,K_j$的角度。而$\begin{aligned}Q_iK_j^T = \lVert Q_i\rVert  \lVert K_j\rVert \cos(\eta_i-\eta_j)\end{aligned}$，若把两个向量各自按照$i,j$角度来旋转后再来计算点积，即$\begin{aligned}Q_iR(i)(K_jR(j))^T \end{aligned}$，可以得到$\begin{aligned}Q_iR(i)R(-j)K_j^T =\cos(\eta_i-\eta_j+(i-j)) \end{aligned}$，可以观察到成功融入了位置的相对信息。

​		所以在输入$\mathbf x_i$经过投影矩阵映射后的向量再由旋转矩阵作用后进行运算是可以融入相对位置信息的，

​																		$\begin{aligned}
Q_{i} & =\mathbf x_i W_{Q} R(i \theta) \\
K_{j} & =\mathbf x_j W_{K} R(j \theta) \\
Q_{i} K_{j}^{T} & =\mathbf x_i W_{Q} R(i \theta) R(j \theta)^{T} W_{K}^{T}\mathbf x_j^{T} \\
& =\mathbf x_i W_{Q} R(i \theta) R(-j \theta) W_{K}^{T}\mathbf x_j^{T} \\
& =\mathbf x_i W_{Q} R((i-j) \theta) W_{K}^{T} \mathbf x_j^{T} \\
& \sim f\left(\mathbf x_i, \mathbf x_j, i-j\right)
\end{aligned}$

​		拓展到高维空间：
​																			$\begin{aligned}
Q_{i} & =\mathbf x_i W_{Q} R(i \Theta) \\
K_{j} & =\mathbf x_j W_{K} R(j \Theta) \\
Q_{i} K_{j}^{T} & =\mathbf x_i W_{Q} R(i \Theta) R(j \Theta)^{T} W_{K}^{T}\mathbf x_j^{T} \\
& =\mathbf x_i W_{Q} R(i \Theta) R(-j \Theta) W_{K}^{T}\mathbf x_j^{T}  \\ 
& = \mathbf x_i W_{Q}\hat R(i\theta_1) \hat R(i\theta_{2}) ....\hat R(i\theta_{D/2}) \hat R(-j\theta_1) \hat R(-j\theta_{2}) ....\hat R(-j\theta_{D/2}) W_{K}^{T}\mathbf x_j^{T}    \\
& = \mathbf x_i W_{Q}\hat R((i-j)\theta_1) \hat R((i-j)\theta_{2}) ....\hat R((i-j)\theta_{D/2})  \\
& =\mathbf x_i W_{Q} R((i-j) \Theta) W_{K}^{T} \mathbf x_j^{T} \\& \sim f\left(\mathbf x_i, \mathbf x_j, i-j\right)
\end{aligned}$

​		所以RoPE就是在多个Transformer Block中，Q,K进行矩阵乘法前，每一个位置的$Q_i,K_j$先各自右乘一个旋转矩阵$R(i\Theta),R(j\Theta)$融入相对位置信息。相当于对$Q_i,K_j$的每一个二维子空间右乘以一个二维的基准旋转矩阵，即按照特定的维度进行旋转。对于位置$i$而言，其对应的旋转角度序列和旋转矩阵就是$i(\Theta)=(i\theta_1,i\theta_2,...,i\theta_{D/2})$和$R(i\Theta)$。

​		而随着维度的增大，还要考虑到旋转角度是否会重复的问题，即每一个子维度的$\theta_k$的取值该才比较怎样合适？对于任意第$k$个子空间，其出现周期性重复的条件是：$(i-j)\theta_k=2N\pi,\theta_k=\frac{2N\pi}{i-j}$，所以只要$\theta_k$不包含$\pi$，那么基本上是不可能会有重复的情况的，而实际取值是$\theta_k=1/10000^{k/D}$，无需担心此问题。RoPE不仅可以考虑到相对位置信息，其还能通转阵来生成超过预训练长度的位置编码。这样可以提高模型的泛化能力和鲁棒性。这一点是其他固定位置编码方式（如正弦位置编码、训练的位置编码等）所不具备的，因为它们只能表示预训练长度内的位置，而不能表示超过预训练长度的位置。



$\text{8.为什么现在的大模型大多是decoder-only的架构?}$

​		Encoder-only比如像BERT，预训练阶段的任务是MLM和NSP，编码器主要通过上下文来预测被掩码掉的token来学习文本语义表征，不适合做生成类的任务，其更适合做自然语言理解类的任务，但即使做自然语言理解类的任务也需要下游数据微调。

​		对于Decoder-only的模型而言，其在预训练阶段的任务是next token prediction，兼顾了理解和生成，参数量大的时候在各种下游任务上的Zero Shot与Few Shot的泛化能力也比较好。当然，现在也有Encoder-Decoder结构的大语言模型如T5、FLAN的Few Shot能力也很不错。

​		但理论上Encoder的双向注意力会存在低秩问题，这可能会削弱模型表达能力，就生成任务而言，引入双向注意力并无实质好处。Decoder-only不一样，他的causal attention是下三角矩阵，必然是满秩的，同等参属下的表达能力更强。而Encoder-Decoder架构之所以能够在某些场景下表现更好，大概只是因为它多了一倍参数，在同等参数量、同等推理成本下，Decoder-only架构就是最优选择了。

$\text{9.ChatGPT的训练步骤有哪些?}$

​	三步走：1.pretraining 2.SFT 3.RLHF。第一个阶段是预训练，openai用了海量的互联网数据进行预训练，预训练任务是next token prediction, 可以认为这一个阶段给模型注入了大量的知识，模型可以根据输入预测下一个token。但是预训练完了以后GPT还不算很好用，他只会根据已有的上文预测接下来出现概率最大token，并没法理解人类想要什么样的回答，因此要通过有监督微调告诉模型人类想要的答案是什么样的。2.有监督微调阶段，数据形式是人类标注的一问一答的语料，而预训练阶段则无此要求。通过有监督微调，模型能够学到人类想要的回答。3.人类反馈强化学习阶段。人类不可能穷举各种各样的问题来做SFT，因此，引入了一个奖励模型来模仿人类偏好，奖励模型可以对GPT的输出（question+answer拼起来）打分，分数越高说明质量越好，反之越差。接着通过强化学习的方式PPO训练GPT。因此，ChatGPT训练步骤分为预训练+有监督微调+人类反馈强化学习。

$\text{10.为什么transformers需要位置编码?}$

​	 	Transformers模型和RNN不同，RNN结构是串行的，训练的时候$t$时刻的隐状态依赖于前一个时刻，因此比较慢，但是其结构蕴含了位置信息。transformer虽然对所有token进行了并行处理，但是无法感知token在句子中的相对和绝对位置信息。

​		记输入序列的每一个token embedding是$\mathbf x_i \in \mathbf R^{768 \times 1}$，token embedding 矩阵记作$\begin{aligned}\mathbf X = \begin{pmatrix} \mathbf x_1^T \\ {\vdots} \\ \mathbf x_k^T \end{pmatrix}\end{aligned}$，如果没有positional embedding，那么有：

$\begin{aligned} Q=\begin{pmatrix} \mathbf x_1^T \\ {\vdots} \\ \mathbf x_k^T\end{pmatrix}W^{Q},K=\begin{pmatrix} \mathbf x_1^T \\ {\vdots} \\\mathbf x_k^T\end{pmatrix}W^K,V=\begin{pmatrix} \mathbf x_1^T \\ {\vdots} \\\mathbf x_k^T\end{pmatrix}W^V\end{aligned}$，故$\begin{aligned}QK^T=\left( \begin{array} { c c c c } { \mathbf x_1^TW^Q{W^K}^T\mathbf x_1 } & { \cdots } & { \mathbf x_1^TW^Q{W^K}^T\mathbf x_k }  \\ { \vdots } & { } & { \vdots } \\ { \mathbf x_k^TW^Q{W^K}^T\mathbf x_1 } & { \cdots } & { \mathbf x_k^TW^Q{W^K}^T\mathbf x_k} \end{array} \right)\end{aligned}$，对于双向语言模型如BERT，$\begin{aligned}softmax(\frac{QK^T}{\sqrt d})V\end{aligned}$的第$i$个位置的语义向量的具体公式是：

​																						$\begin{aligned}\frac{1}{C} \sum_{j=1}^{k}e^{\frac{ \mathbf x_i^TW^Q{W^K}^{T}\mathbf x_j}{\sqrt d}}\mathbf x_j^TW^V\end{aligned}$

其中$\begin{aligned}C=\sum_{j}^ke^{\frac{\mathbf x_i^TW^Q{W^K}^{T}\mathbf x_j}{\sqrt d}}\end{aligned}$。假设原来的第$i$个位置的token是“我“，第$i+p$个token是”你“，现在这两个token换一下位置。分析第$i+p$个位置上（”我“）的语义向量。现在有$\mathbf x_{i+p}^{new}=\mathbf x_i^{old},\mathbf x_i^{new}=\mathbf x_{i+p}^{old}$，以下将$\mathbf x_i^{old}记作\mathbf x_i$，记新的$Q,K,V$分别为$Q',K',V'$，那么$Q'K'^T$的第$i+p$行就是：

​								$\begin{aligned}&\left( \begin{array} { c c c c } {{\mathbf x_{i+p}^{new}}^T}W^Q{W^K}^T{\mathbf x_1},...,{{\mathbf x_{i+p}^{new}}^T}W^Q{W^K}^T{\mathbf x_{i}^{new}},...,{{\mathbf x_{i+p}^{new}}^T}W^Q{W^K}^T{\mathbf x_{i+p}^{new}},...,{{\mathbf x_{i+p}^{new}}^T}W^Q{W^K}^T{\mathbf x_k}\end{array} \right) \\&=\left( \begin{array} { c c c c } {{\mathbf x_i}^T}W^Q{W^K}^T{\mathbf x_1},...,{{\mathbf x_i}^T}W^Q{W^K}^T{\mathbf x_{i+p}},...,{\mathbf x_i^T}W^Q{W^K}^T{\mathbf x_i},...,{\mathbf x_i^T}W^Q{W^K}^T{\mathbf x_k}\end{array} \right) \end{aligned}$

​		将$softmax(\frac{Q'K'^T}{\sqrt d})$记作$A'$，那么$A'$的第$i+p$行为$A_{[i+p,:]}$与$V'$矩阵乘法如下：

​						$\begin{aligned}\frac{1}{C}\left( \begin{array} { c c c c } {e^{\frac{{{\mathbf x_i}^T}W^Q{W^K}^T{\mathbf x_1}}{\sqrt d}}},...,{e^{\frac{{{\mathbf x_i}^T}W^Q{W^K}^T{\mathbf x_{i+p}}}{\sqrt d}}},...,{e^{\frac{{{\mathbf x_i}^T}W^Q{W^K}^T{\mathbf x_i}}{\sqrt d}}},...,{e^{\frac{{{\mathbf x_i}^T}W^Q{W^K}^T{\mathbf x_k}}{\sqrt d}}}\end{array} \right) \begin{pmatrix} \mathbf x_1^TW^V\\ {\vdots} \\\mathbf x_{i+p}^TW^V\\ {\vdots}\\\mathbf x_i^TW^V\\ {\vdots}\\\mathbf x_k^TW^V \end{pmatrix}\end{aligned}$

​		其实就等价于之前的$\begin{aligned} \frac{\sum_{j=1}^{k}e^{\frac{ \mathbf x_i^TW^Q{W^K}^{T}\mathbf x_j}{\sqrt d}}\mathbf x_j^TW^V}{C}\end{aligned}$，上述的公式表明，如果把输入序列顺序打乱，如第$i$与$i+p$位置的token交换顺序，最终交换了位置的两个token对应语义向量是不会变的（其他没有交换位置的token的按照注意力权重融合后的语义向量也不会变），即输入变量在注意力机制的作用下满足$f(\mathbf x_1,\mathbf x_2,...,\mathbf x_k)=f({permutation(\mathbf x_1,\mathbf x_2,...,\mathbf x_k)})$。比如序列$\text{“我不喜欢你”}$变成$\text{“你不喜欢我"}$，那么最终按照注意力权重融合了的语义向量的$\text{“我”}$和$\text{“你”}$中两个语义向量是完全和没打乱顺序前一样的，这并不合理。因此transformer需要位置编码，帮助模型获取token的顺序关系。

$\text{11.为什么对于ChatGPT而言，提示工程很重要?}$

​		提示工程之所以对ChatGPT至关重要，是因为它能够显著提高模型生成文本的相关性、准确性和多样性。在预训练过程中，虽然模型已经吸收了大量的信息，但它并不总是能够准确理解用户的查询意图，而在对齐阶段则是被教导如何输出与人类意图一致的高质量文本内容。对于ChatGPT这种大语言模型而言，通过提示工程技术，无需消耗大量资源微调就能被用于解决各种各样的任务。

​		提示工程通过精心设计的输入提示，可以减少模型的猜测空间，使其更专注于特定任务或问题领域，引导模型朝向更高质量的输出。常见的提示工程技术有Few Shot Prompting, Chain of Thought Prompting等。前者通过在提示中给出几个相应的示例，使模型能够快速适应新的任务或领域，不需要额外的微调就能了解人类的意图，依据参考案例学习就能提升回答内容的质量。后者通过在提示中给出思维链（即明确的推理步骤）可以显著提升大语言模型在复杂推理任务上的成绩。

​		此外，工程师可以设定特定的提示引导ChatGPT按照特定的风格或者扮演角色生成文本，也可以在提示种给出各式各样的案例让模型学习到更广泛的模式，提升面对未知输入的适应能力；还可以加入反思的机制让模型避免输出不恰当或有害的内容。

$\text{12.如何缓解 LLMs 复读机问题?}$

​		出现复读机问题可能主要原因是LLM在预训练阶段的数据影响，如果训练数据中存在大量的重复文本或者某些特定的句子或短语出现频率较高（缺乏多样性），模型在生成文本时可能会倾向于复制这些常见的模式。此外，LLM在预训练阶段是做的next token prediction。这样的训练目标可能使得模型更倾向于生成与输入相似的文本，导致复读机问题的出现。

​		缓解复读机问题的方法有如下几种：

1. 训练数据的多样性，在训练时考虑不同风格的文本数据，有助于大语言模型学习到更多样的表达方式，减少复读机情况发生。
1. 1调整Decode的超参数，如temperature和repetition_penalty。较大的temperature会增加生产文本的随机性，repetition_penalty则是调整以生成序列中单词的softmax概率来施加惩罚，如一个词已经出现过，那么计算其softmax时会除以repetition_penalty降低它再次被选中的概率。

​	
