​	

​	

# 1.Recall

## 1.1 Sentence BERT

![image-20250701150319806](E:\Study\gitpro\knowledge-planet\NLP系列\assets\image-20250701150319806.png)

##1.2 SimCSE

​	SimCSE通过对比学习的方式学习句向量的语义表征，仅用简单的Dropout技术就显著提升了句子嵌入的质量，在多个语义相似度任务上刷新了SOTA。SimCSE的核心思想是**让相似的句子在向量空间中靠近，不相似的远离**，创新点在于构造正负样本的方式，其可分为无监督和有监督两个版本，具体原理如下图：

![image-20250701150217824](E:\Study\gitpro\knowledge-planet\NLP系列\assets\image-20250701150217824.png)

### 1.2.1 无监督



### 1.2.2 有监督

​	有监督SimCSE利用自然语言推理（NLI）数据集，**正样本**：蕴涵（entailment）关系的句子对。**负样本**：矛盾（contradiction）关系的句子对。通过构造负样本，在一个Batch内让模型区分更多的句子对显著提示模型学习的语义表征质量。这个思想可以拓展到不同类型的下游任务而不仅仅是NLI数据集，一个典型的场景是文档检索场景，我们可以为查询$query$召回多个语义相似的候选文档作为负样本，具体地，我们可以将一个查询，查询对应的真实文档，相关文档三者作为一个样本对$(query,doc^+,doc_j^-)$。多个这样的样本对组成一个Batch作为神经网络输入，通过对比损失提升模型在下游任务上的性能，具体原理如下图：

![image-20250701153752592](E:\Study\gitpro\knowledge-planet\NLP系列\assets\image-20250701153752592.png)

​	首先是有监督无负样本，此时查询，查询对应的真实文档$(query_i,doc_i^+)$作为一个样本对，$N$个这样的样本对构成一个Batch,因此一个Batch有$2N个$输入，神经网络输出$2N$个对应的语义向量，这$2N$个语义向量先两两计算预先相似度得到一个相似度方阵$\mathbf S$，其中$\mathbf S_{ij}=\operatorname{Sim}(\mathbf h_i,\mathbf h_j)$，SimCSE的思想是，只比较$query_i$和其他的$doc_j^+$之间的语义相似度，不比较$query_i$和$query_j$的语义相似度，因此在方阵$\mathbf S$中只取第奇数行，第偶数列。因为第$2,4,\ldots,2n$行都是$doc_j^+$和其他比，我们不需要这部分比较的信息，其已经蕴含在$query_i$和其他比的信息中了。而第奇数列都是$query_i$和$query_j$比，因此这部分信息我们也不需要。于是我们只在第偶数行上算交叉熵损失便等价于对比损失。同样的，是一个负样本的有监督SimCSE，那么当是第$(2k),(2k+1),k=1,...,n$行不需要考虑，第$3k-2,k=1,..,n$列不需要考虑，在此基础上采用交叉熵损失等价于对比损失。此外，论文引入了温度系数$\tau$来控制负样本带来的影响，即$\operatorname{softmax with temperature}$，温度系数越大，则受负样本的影响越小，温度系数越小，则更容易受到影响，具体分析如下[[x]]()：记模型为$\mathbf E(\cdot)$，当前查询为$query_i$，对应的正样本为$doc_i^+$，负样本为$doc_{i}^-$，有$\mathbf E(query_i)=\mathbf h_i,\mathbf E(doc_i^+)=\mathbf h_i^+,\mathbf E(doc_i^-)=\mathbf h_i^-$。
$$
\begin{aligned}\lim_{\tau\rightarrow0^+}\frac{\exp(\operatorname{Sim}(\mathbf h_i,\mathbf h_j^+)/\tau)}{\sum_{k=1}^{n}\exp(\operatorname{Sim}(\mathbf h_i,\mathbf h_k^+)/\tau)+\exp(\operatorname{Sim}(\mathbf h_i,\mathbf h_k^-)/\tau)}\end{aligned}
$$