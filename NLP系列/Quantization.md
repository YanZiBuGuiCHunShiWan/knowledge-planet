- # 大模型量化


# 1.基础

​	量化就是将以高精度类型如fp32,fp16存储的数据通过缩放和取整转化为低精度类型如int4,int8存储的数据，使得模型参数占用内存大幅降低，在推理时通过缩放因子反量化尽可能还原推理结果，即原始输入$\mathbf x_{f}\xrightarrow{\text{quantize}}\mathbf x_{q}\xrightarrow{\text{dequantize}}\mathbf x_{f}'$​。当然，反量化后得到的结果和原始的矩阵乘法要尽可能接近。

## 1.1常见的数据类型

​	x

## 1.2 对称量化



## 1.3 非对称量化



## 1.4  映射范围与裁剪



# 2. 训练后量化

## 2.1  LLM.int8()

​	**绝对值最大量化(Absmax quantization)**是最常用的技术手段，如果将数据量化成8bit的类型，那么将输入元素除以该张量中绝对值最大的元素，再乘以127然后四舍五入为整数，就能得到$[-127,127]$之间的整数。假设输入矩阵为$\mathbf X_{f16}\in\mathbb R^{s\times h}$，绝对值量化的过程用数学公式描述为：
$$
\mathbf{X}_{i 8}=\left\lfloor\frac{127 \cdot \mathbf{X}_{f 16}}{\max _{i j}\left(\left|\mathbf X_{f 16_{i j}}\right|\right)}\right\rceil=\left\lfloor\frac{127}{\left\|\mathbf{X}_{f 16}\right\|_{\infty}} \cdot \mathbf{X}_{f 16}\right\rceil
$$
​	其中${\left\|\mathbf{X}_{f 16}\right\|_{\infty}}$是张量中的最大绝对值，$\left\lfloor \right\rceil$表示四舍五入到的最近整数。

​	**零点量化**的目标是把浮点数据线性映射到 $[-127, 127] $的整数范围，并通过一个“零点”对齐偏移，使得所有数据都能充分利用整数范围。其具体量化过程如下，首先计算缩放因子：
$$
n d_{x_{f 16}}=\frac{2 \cdot 127}{\max _{i j}\left(\mathbf X_{f 16}^{i j}\right)-\min _{i j}\left(\mathbf X_{f 16}^{i j}\right)}
$$
​	将输入动态范围线性映射到$[-127, 127] $区间，然后再计算零点偏移量：
$$
zp_{xi8}=\left\lfloor \mathbf X_{f16}\cdot \min_{ij}(\mathbf X_{f16}^{ij})\right\rceil
$$
​	当于记录“最小值”浮点数在整数空间中应被映射到的位置，用于后续解码时恢复原始范围。最终量化公式与反量化如下：
$$
\begin{aligned}\mathbf X_{i8}&=\left\lfloor \mathbf X_{f16}\cdot nd_{x_{f16}}\right\rceil \\
\mathbf X_{f16}'&\approx\frac{\mathbf X_{i8}-zp_{x_{i8}}}{nd_{x_{f16}}}\end{aligned}
$$
​	给定两个浮点数$A_{f16}$和$B_{f16}$，两个零点量化后的数分别$A_{i8}$和$B_{i8}$，和对应的零点偏移$zp_{a_{16}}$和$zp_{b_{16}}$。理想情况下，直接通过浮点计算为：
$$
\begin{aligned}A_{f16}\cdot B_{f16}&\approx\big(\frac{A_{i8}-zp_{a_{16}}}{nd_{a_{f16}}}\big)\big(\frac{B_{i8}-zp_{b_{16}}}{nd_{b_{f16}}}\big) \\ 
{nd_{a_{f16}}}{nd_{b_{f16}}}A_{f16}\cdot B_{f16}&\approx ({A_{i8}-zp_{a_{16}}})({B_{i8}-zp_{b_{16}}})\end{aligned}
$$
​	所以我们可以先基于Int8类型的数进行计算，即先算$({A_{i8}-zp_{a_{16}}})({B_{i8}-zp_{b_{16}}})$，此时乘法运算在16为的整数空间进行（避免溢出）。理想情况下有专用指令，硬件直接支持一步完成 **“INT8加零点 + INT16乘法”**，此外，硬件加速中要避免实时减法，因此存储时我们让$-zp_{a_{16}}=zp_{a\_stored},-zp_{b_{16}}=zp_{b\_stored}$，这样减号就能转化为加号，当无专用指令的情况下则需手动展开计算，需手动展开计算：

```python
C_i32 = A_i8 * B_i8      // INT8乘法（精度低）
       + A_i8 * zp_b_stored     // INT8 * INT16（需升位）
       + B_i8 * zp_a_stored     // INT8 * INT16（需升位）
       + zp_a_stored * zp_b_stored     // INT16 * INT16（大数计算）
```

​	在计算完$	C_{i32}$后在通过${nd_{a_{f16}}},{nd_{b_{f16}}}$这两个缩放因子进行dequantize即可。

​	因此，对于fp16类型的输入$\mathbf X_{f16}\in\mathbb R^{s\times h}$，权重矩阵$\mathbf W\in \mathbb R^{h\times o}$，我们可以通过8-bit的矩阵乘法完成计算：
$$
\begin{aligned} \mathbf X_{f16}&=\begin{bmatrix}
\mathbf X_{f16}^{11} & \cdots &\mathbf X_{f16}^{1h} \\
\vdots & & \vdots\\
\mathbf X_{f16}^{s1} &\cdots & \mathbf X_{f16}^{sh}
\end{bmatrix},\mathbf W_{f16}= \begin{bmatrix}
\mathbf W_{f16}^{11} & \cdots &\mathbf W_{f16}^{1o} \\
\vdots & & \vdots\\
\mathbf W_{f16}^{h1} &\cdots & \mathbf W_{f16}^{ho}
\end{bmatrix}\\
\mathbf X_{f16}\mathbf W_{f16}&\approx \frac{1}{n d_{x_{f 16}}\cdot n d_{w_{f 16}}}\mathbf C_{i32}\\
&\approx\frac{1}{n d_{x_{f 16}}\cdot n d_{w_{f 16}}}\mathbf A_{i8}\mathbf B_{i8}=S_{f16}\cdot Q(\mathbf X_{f16})Q(\mathbf W_{f16})\end{aligned}
$$
​	而量化算法面临的主要挑战是——每一个张量如果只用一个缩放因子，那么任何一个异常值都会降低其他值的量化精度。因此每个张量都应该有多个缩放因子。LLM.int8()提出了向量级量化技术，将张量拆解为独立向量块，每个向量块独立计算缩放因子。但对于动轴上百亿参数量的模型而言，向量量化并不够迅速，因此需要**混合精度分解**，即将少量高精度特征维度（占比≈0.1%）以16位精度表示，其余99.9%的数值则采用8位精度。

### 2.1.1 向量量化

​	将矩阵乘法可以看错序列化的向量乘法，将每个序列的向量内积都对应一个缩放因子那么就可以增加矩阵乘法的缩放因子个数。给定隐状态$\mathbf X_{f16}\in\mathbb R^{b\times h}$和参数矩阵$\mathbf W_{f16}\in \mathbb R^{h\times o}$，我们可以给隐状态每一行都分配一个缩放因子$c_{x_{f16}}$，给参数矩阵每一列都分配一个缩放因子$c_w$。因此量化矩阵乘法可以写成：
$$
\begin{aligned}Q(\mathbf X_{f16})Q(\mathbf W_{f16})=\mathbf A_{i8}\mathbf W_{i8}\end{aligned}
$$
​	而反量化后的结果$\mathbf C_{f_{16}}$需要在$\mathbf A_{i8}\mathbf W_{i8}$得到的结果上乘以缩放因子dequantize回来，笔者给一个简单的示意图（为方便表示，符号进行了变动）：

![image-20250718154014364](E:\Study\gitpro\knowledge-planet\NLP系列\assets\image-20250718154014364.png)

​	其中，$c_{xi}$代表输入的第$i$行的缩放因子，$c_{wj}$代表参数矩阵第$j$列的缩放因子。基于$\mathbf A_{i8}\mathbf W_{i8}$进行dequantize就是在量化后的矩阵上除以对应的缩放因子，即量化后的矩阵为$\mathbf D_{i8}$，则其第$i$行第$j$列的元素进行dequantize 就是乘以$1/{c_{xi}c_{wj}}$。因子还原后的$C_{f_{16}}$公式如下：
$$
\begin{aligned}\mathbf{C}_{f_{16}} &\approx \frac{1}{\mathbf{c}_{x_{f 16}} \otimes \mathbf{c}_{w_{f 16}}} \mathbf{C}_{i 32}=\mathbf{S} \cdot \mathbf{A}_{i 8} \mathbf{B}_{i 8}=\mathbf{S} \cdot Q\left(\mathbf{A}_{f 16}\right) Q\left(\mathbf{B}_{f 16}\right)\\
&\approx \mathrm{Diag(\mathbf{C}_{x_{f16}}^{-1})}\mathbf{A}_{i 8} \mathbf{B}_{i 8}\mathrm{Diag(\mathbf{C}_{w_{16}}^{-1})}\end{aligned}
$$
​	那么知道了向量量化的具体过程，我们就可以找到那些存在较大值的维度，即将维度分成两部分，一部分是极大值能显著影响模型性能的，这一部分不需要量化防止模型性能降低，将异常值维度放进集合$O$，即$O=\{i|i\in \mathbb Z,0\leq i\leq h\}$​，而另一部分不存在异常值的维度则可以进行量化并储存好缩放因子，在推理时进行dequantize，整个混合精度矩阵乘法公式定义如下：
$$
\begin{aligned} \mathbf {C_{f16}} \approx \sum_{h\in O}\mathbf X_{f16}^{h}\mathbf W_{f16}^{h}+\mathbf S_{f16}\cdot\sum_{h\notin O}\mathbf X_{i8}^{h}\mathbf W_{i8}^{h}\end{aligned}
$$
​	那异常值维度如何找到？我们定义一个特征维度为异常值维度如果满足以下条件：（1）至少在$25\%$的序列维度中；（2）至少出现在$50\%$层中；（3）该维度幅值大于等于$6$。也就是说每一层的异常值检测 仅依赖于当前层的输入 （即上一层的输出），而不需要预知后续层的激活状态。





​	**显存减少的来源：**（1）激活值。（2）权重。

## 2.2 SmoothQuant

​	动机是什么？xx现象。

​	阐明现象的体现以及造成该现象的原因。

​	解决方案。

​	潜在缺陷。





## 2.3 AWQ













