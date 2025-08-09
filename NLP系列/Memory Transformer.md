# Memory Transformer



# 1.Linear Transformer

## 1.1 Linear Transformers as RNNs.

​	给定输入$\mathbf X\in \mathbb R^{S\times H}$，自注意力机制机制通过三个投影矩阵$\mathbf W^Q,\mathbf W^K,\mathbf W^V\in\mathbb R^{D\times D} $将输入映射得到$\mathbf {Q,K,V}\in\mathbb R^{S\times D} $，然后计算注意力分数矩阵$\mathbf A$，并依据注意力分数矩阵加权得到上下文矩阵$\mathbf C$。	
$$
\begin{aligned}\mathbf Q&=\mathbf {XW}^Q\\
\mathbf K&=\mathbf {XW}^K\\
\mathbf V&=\mathbf {XW}^V\\
\mathbf C&=\underbrace{\mathrm{Softmax}(\frac{\mathbf{QK^{\mathrm{T}}}}{\sqrt D})}_{\mathbf A}\mathbf V\end{aligned}
$$
​	对于上下文矩阵$\mathbf C$中的某一行上下文向量$\mathbf c_i$则可以表示成：
$$
\begin{aligned}\mathbf c_i=\sum_{j=1}^{N}\mathbf A_{i,j}\mathbf v_j=\frac{\sum_{j=1}^{N}\mathrm{sim}(\mathbf q_i,\mathbf k_j)\mathbf v_j}{\sum_{j=1}^{N}\mathrm{sim}(\mathbf q_i,\mathbf k_j)}\end{aligned}
$$
​	其中，相似度函数$\mathrm{sim}(\mathbf q_i,\mathbf k_j)=\exp(\frac{\mathbf q_i\mathbf k_{j}^{\top}}{\sqrt D})$。计算注意力矩阵的计算复杂度为$\mathcal O(N^2D)$，随着序列长度的增加，计算开销会呈二次方增长。而借鉴核技巧的思想，可以尝试将向量内积由函数表达，即：
$$
\begin{aligned}\mathbf c_i&=\frac{\sum_{j=1}^{N}\phi(\mathbf q_i)\phi(\mathbf k_j)^{\top}\mathbf v_j}{\sum_{j=1}^{N}\phi(\mathbf q_i)\phi(\mathbf k_j)^{\top}}\\
&=\frac{\phi(\mathbf q_i)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}\mathbf v_j}{\phi(\mathbf q_i)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}}\end{aligned}
$$
​	如果是整个上下文向量$\mathbf C$，忽略下角标索引则有：
$$
\begin{aligned}\mathbf C&=
\begin{pmatrix}
\mathbf c_1 \\
\vdots\\
\mathbf c_N
\end{pmatrix}
=\begin{pmatrix}
\frac{\phi(\mathbf q_1)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}\mathbf v_j}{\phi(\mathbf q_1)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}} \\
\vdots\\
\frac{\phi(\mathbf q_N)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}\mathbf v_j}{\phi(\mathbf q_N)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}}
\end{pmatrix}=\begin{pmatrix}
{\phi(\mathbf q_1)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}\mathbf v_j}\\
\vdots\\
{\phi(\mathbf q_N)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}\mathbf v_j}
\end{pmatrix}⊙
\begin{pmatrix}
\frac{1}{\phi(\mathbf q_1)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}} \\
\vdots\\
\frac{1}{\phi(\mathbf q_N)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}}
\end{pmatrix}\\
&=\begin{pmatrix}
{\phi(\mathbf q_1)}\\
\vdots\\
{\phi(\mathbf q_N)}\end{pmatrix}\big({\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}\mathbf v_j}\big)⊙
\begin{pmatrix}
\frac{1}{\phi(\mathbf q_1)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}} \\
\vdots\\
\frac{1}{\phi(\mathbf q_N)\sum_{j=1}^{N}\phi(\mathbf k_j)^{\top}}
\end{pmatrix}\\
&=\Phi(\mathbf Q)\big(\Phi(\mathbf K)^{\top}\mathbf V\big)⊙\underbrace{\big[\Phi(\mathbf Q)\Phi(\mathbf K)^{\top}\mathbf 1_{N}\big]^{-1}}_{归一化因子\mathbf Z}
\end{aligned}
$$
​	根据这个公式，那么思路是可以先算出$\Phi(\mathbf K)^{\top}\mathbf V$，然后算出$\Phi(\mathbf K)^{\top}$在列的方向上的加和即$\Phi(\mathbf K)^{\top}\mathbf 1_{N}$。再算$\Phi(\mathbf Q)$和这两个变量的矩阵乘法后的结果，得到最终的上下文$\mathbf C$矩阵，码实现如下：

```python
def linear_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float) -> torch.Tensor:
	#Q,K,V:[Batch_size,Seq_len,Hidden_dim]
    # 将Q核K高维映射
    phi_Q = nn.ELU(Q) + 1
    phi_K = nn.ELU(K) + 1
    # 计算 K 和 V 的乘积
    KV = torch.matmul(phi_K.transpose(-1, -2), V) #[Batch_size,Hidden_dim,Hidden_dim] $\Phi(\mathbf K)^{\top}\mathbf V$
    K_sum =  phi_K.sum(dim=-2, keepdim=True).transpose(-1, -2) #[Batch_size,Hidden_dim,1] $\Phi(\mathbf K)^{\top}\mathbf 1_{N}$
    # 计算归一化因子 Z
    Z = 1.0 / (torch.matmul(phi_Q,K_sum).squeeze(-1) + eps) #[Batch,Seq_len,1] $\big[\Phi(\mathbf Q)\Phi(\mathbf K)^{\top}\mathbf 1_{N}\big]^{-1}$
    # 计算新的 V 值
    context_vector = torch.matmul(phi_Q, KV) * Z.unsqueeze(-1) #$\Phi(\mathbf Q)\big(\Phi(\mathbf K)^{\top}\mathbf V\big)⊙\big[\Phi(\mathbf Q)\Phi(\mathbf K)^{\top}\mathbf 1_{N}\big]^{-1}$
    # 重新排列输出张量的维度
    return context_vector
```

​	计算复杂度分析，我们看矩阵乘法，对$\mathbf {Q,K}$进行核函数映射，由于序列长度是$N$,维度是$D$，故计算复杂度是$\mathcal O(2ND)$。然后是$\Phi(\mathbf K)^{\top}\mathbf V={\sum_{j=1}^{N}\phi(\mathbf K_j)^{\top}\mathbf V_j}$，执行$N$次向量外积运算，计算复杂度是$\mathcal O(ND^2)$，然后是$\Phi(\mathbf K)^{\top}$在列的方向上的加和，计算复杂度仍然是$\mathcal O(ND)$，归一化因子的计算是矩阵乘以向量故结果还是一个向量，做$N$次向量内积计算，对应的计算复杂度是$\mathcal O(ND)$，最后的上下文向量在$\Phi(\mathbf Q)\big(\Phi(\mathbf K)^{\top}\mathbf V\big)$基础上乘以缩放因子，计算复杂度为$\mathcal O(N)$，故整体计算复杂度为：
$$
\mathcal O(2ND+ND^2+ND+ND+N)
$$
​	也就是当$N\gg D^2$时，Linear Attention算出上下文向量的计算复杂度是$\mathcal O(N)$。在上文中，注意力机制并没有加入因果掩码，基于自回归解码器的思想，改写后的Linear Attention的上下文向量$\mathbf C_i$:
$$
\begin{aligned}\mathbf c_i&=\frac{\phi(\mathbf q_i)\sum_{j=1}^{i}\phi(\mathbf k_j)^{\top}\mathbf v_j}{\phi(\mathbf q_i)\sum_{j=1}^{i}\phi(\mathbf k_j)^{\top}}\end{aligned}
$$
​	定义两个新变量$\mathbf s_i$和$\mathbf z_i$如下：
$$
\begin{aligned}\mathbf s_i&={\sum_{j=1}^{i}\phi(\mathbf k_j)^{\top}\mathbf v_j}\\
\mathbf z_i&=\sum_{j=1}^{i}\phi(\mathbf k_j)^{\top}\end{aligned}
$$
​	那么，有$\mathbf s_{i+1}=\mathbf s_i+{\phi(\mathbf k_{i+1})^{\top}\mathbf v_{i+1}}$，$\mathbf z_{i+1}=\mathbf z_{i}+\phi(\mathbf k_{i+1})^{\top}$。

​	也就是在解码时，系统只需要保存每个时刻的$\mathbf s_t,\mathbf z_t$，每次预测的时间和内存成本是常数。整体来说，Linear Attention结合了RNN和Transformer的全部优点：在训练时，Linear Attention可以做到像Transformer那样并行训练，从而充分利用GPU的性能。在推理时只需在内存中维护向量$\mathbf s_t,\mathbf z_t$，和标准Attention相比无需维护动态变化的KV Cache，显著减少了显存占用。而Linear Transformer也能转化成递归神经网络的形式：
$$
\begin{aligned}\mathbf s_0&=0\\
\mathbf z_0&=0\\
\mathbf s_i&=\mathbf s_{i-1}+{\phi(\mathbf k_{i})^{\top}\mathbf v_{i}}\\
\mathbf z_i&=\mathbf z_{i-1}+\phi(\mathbf k_i)^{\top}\\
\mathbf y_i&=f_l(\frac{\phi(\mathbf {x_{i}W}^Q)\mathbf s_i}{\phi(\mathbf {x_{i}W}^Q)\mathbf z_i}+\mathbf x_i)\end{aligned}
$$
​	**Linear Attention的劣势** 在



## 1.2 Forgetting Mechanism





## 1.3 Delta-Net





# 2. Learn at Test Time







# 参考文献

