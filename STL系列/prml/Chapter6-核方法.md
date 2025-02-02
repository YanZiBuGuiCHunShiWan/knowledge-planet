# 核方法

# 相关概念

现代数学的一个特点就是以集合为研究对象，这样的好处就是可以将很多不同问题的本质抽象出来，变成同一个问题，当然这样的坏处就是描述起来比较抽象，很多人就难以理解了。

## 1.1 线性空间

### 1.1.1 向量空间

主要研究集合的描述，关注的是向量的位置，直观地说就是如何告诉别人这个集合的样子，为了描述清楚，就引入了“基”的概念，对于向量空间，集合元素只需要知道在给定基下 的坐标就能描述出位置。

### 1.1.2 赋范空间

向量空间中的元素没有长度，为了量化向量空间中的元素，引入了特殊的长度“范数”，赋予了范数的向量空间又称赋范空间。

### 1.1.3 度量空间

度量空间就是指定义了度量函数的空间，对属于$L$的两个任意元素$\mathbf x,\mathbf y$，存在一个函数$d$满足如下性质：

（1）$d$是实值的、有限的、非负的（2）$d(\mathbf x,\mathbf y)=d(\mathbf y,\mathbf x)$（3）$d(\mathbf x,\mathbf y)\leq d(\mathbf x,z)+d(z,\mathbf y),\mathbf x,\mathbf y,z\in L  $

### 1.1.4 空间的完备性

属于空间$L$中的任意柯西列都收敛，且收敛到$L$内，柯西列的定义：设$\left\{ \mathbf x_n\right\}$是度量空间$R$中的点列，如果它满足柯西准则：即若对于任意的$\epsilon >0$，存在一个数$N_{\epsilon}$使得所有的$\mathbf x_{n^{'}}>N_{\epsilon},\mathbf x_{n^{''}}>N_{\epsilon},d(\mathbf x_{n^{'}},\mathbf x_{n^{''}})<{\epsilon}$，则称为柯西列。即离得越来越近的点列，且后面两点之间的距离无穷小。完备性就是指$n \rightarrow \infin$时，数列$\left\{ \mathbf x_n\right\}$极限存在，且也属于$L$。

若一个空间可以定义度量函数，且它是完备的，那么他就是完备度量空间（判断柯西列是否收敛需要计算两点间的距离，因此需要定义度量函数。）**如果一个空间既是线性的，又是赋范的，还是完备的，则是一个巴拿赫空间**。前面所述空间关系如下。

### 1.1.5内积空间

赋范空间中的元素没有角度的概念，因此又引入了内积的概念。在线性空间上引入范数和内积，就是内积空间。

### 1.1.6 希尔伯特空间

因为有度量，所以可以在度量空间、赋范线性空间以及**内积空间中**引入极限，但抽象空间中的极限与实数上的极限有一个很大的不同就是，极限点可能不在原来给定的集合中，所以又引入了完备的概念。完备的**内积空间**就是**希尔伯特空间**。

总结：**((向量空间+范数=赋范空间+线性结构)+内积=内积空间)+完备性=希尔伯特空间**，整体关系如下图所示：

![space2](src\kernel mothods\space2.png)

![RKHS-space](src\kernel mothods\RKHS-space.png)



## 1.2 线性算子与线性泛函



# 2 核函数

## 2.1 再生核

​	**首先介绍核函数的定义：**给定$\mathbf x$和$k$，在希尔伯特空间中存在一个内积和映射$\Phi: \mathbf x \rightarrow \mathcal H$使得:
$$
K(\mathbf x,\mathbf y)=\langle\phi(\mathbf x),\phi(\mathbf y)\rangle_{\mathcal H} ，对于所有的\mathbf x,\mathbf y\in \mathcal H
$$
​	**求值泛函：**求值泛函是定义在函数空间上的函数，设有集合$L$及希尔伯特空间$\mathcal H$，且$\mathcal H$内的元素为函数$f:L \rightarrow R$，则对于任意$L$上的元素$\mathbf x \in L$,求值泛函为函数$\delta_\mathbf x:\mathcal H \rightarrow R$且对于任意$\mathbf x$有$\delta_\mathbf x(f)=f(\mathbf x)$。在点$\mathbf x$处的狄拉克求值泛函$\delta_{\mathbf x}$记作。根据[Riesz representation theorem]()，如果$\phi$是一个在$\mathcal H$上的有界线性泛函，则存在一向量$u \in \mathcal H$使得$\phi f=\langle f,u\rangle,\mathbf x \text{for all }f\in \mathcal H$，将这个理论放到当狄拉克求值泛函的情境下翻译，意思是对于每一个狄拉克求值泛函$\delta_\mathbf x$，存在一个唯一的向量（记作$k_\mathbf x$）$k_\mathbf x\in \mathcal H$，使得:[[4]](http://users.umiacs.umd.edu/~hal/docs/daume04rkhs.pdf)
$$
\delta_\mathbf xf=f(\mathbf x)=\langle f,k_\mathbf x\rangle_{\mathcal H} \tag {2.1}
$$
​	由于$k_\mathbf x$是唯一的，对于每一个$\mathbf x$，都能在$\mathcal H$中找到一个这样的$k_\mathbf x$，不同的点$\mathbf x$与$\mathbf x'$对应的$k_\mathbf x$与$k_\mathbf x'$是不同的，但它们都属于$\mathcal H$，可以通过内积$\langle ,\rangle_{\mathcal H}$进行运算，更进一步，我们可以**定义**$K(\mathbf x,\mathbf x')=\langle k_\mathbf x,k_\mathbf {x'}\rangle_{\mathcal H}$，其中$k_\mathbf x$与$k_\mathbf x'$分别是$\delta_\mathbf x$与$\delta_{\mathbf x'}$的唯一表示。现在我们来深入理解为什么可以这样定义，以及什么是核的可再生性。

​	根据利兹表示理论，我们知道$f(\mathbf x)=\langle f,k_\mathbf x\rangle_{\mathcal H},\forall f\in \mathcal H$，取$f=k_{\mathbf x'} \in \mathcal H$的特殊情况，则有$k_{\mathbf x'}(\mathbf x)=\langle k_{\mathbf x'},k_\mathbf x \rangle_{\mathcal H}$，此时我们将$\mathbf x'$视作一个常量，$\mathbf x$视作一个变量，根据核函数的定义，$K(\mathbf x,\mathbf x')$是$k_{\mathbf x'}(\mathbf x)$的具体表示形式：
$$
\begin{aligned} K(\mathbf x,\mathbf x')=k_{\mathbf x'}(\mathbf x)= \langle k_\mathbf x ,k_{\mathbf x'} \rangle_{\mathcal H}, \forall \mathbf x,\mathbf x' \in \mathcal H \end{aligned} \tag{2.2}
$$
​	将$K(\mathbf x,\mathbf x')$视作一个两个变量的函数，并记$K(\cdot,\mathbf x')=k_{\mathbf x'}$代表固定住变量$\mathbf x'$的情况下的函数本身，即$k_\mathbf x$定义为$k_\mathbf x:\mathbf y\mapsto K(\mathbf x,\mathbf y)$即$k_\mathbf x(\mathbf y)=K(\mathbf x,\mathbf y)$，结合$\forall f \in \mathcal H$，$f(\mathbf x)=\langle f,k_\mathbf x\rangle_{\mathcal H}$，有：
$$
\begin{aligned} f(\mathbf x')&=\langle f,k_{\mathbf x'}\rangle =\langle f,K(\cdot,\mathbf x') \rangle \\ 
f(\mathbf x) &=\langle f,k_{\mathbf x}\rangle =\langle f,K(\cdot,\mathbf x) \rangle \\
K(\mathbf x,\mathbf y)&=\langle k_\mathbf x,k_{\mathbf y}\rangle=\langle K(\mathbf x,\cdot ),K(\mathbf y,\cdot)\rangle  \end{aligned} \tag{2.3}
$$
​	**总结** 核的可再生性是指对于 $\mathcal{H}$ 中的任意函数$f$和任意 $\mathbf x\in \mathcal X$，核函数$K$满足：
$$
f(\mathbf x)=\langle f,K(\mathbf x,\cdot)\rangle_{\mathcal H} \tag{2.4}
$$
即函数值$f(\mathbf x)$可以通过 $\mathcal{H}$中$f$与核函数的内积的特殊形式计算出来。

## 2.2 RKHSs的构造

​	实际上，任何的正定函数都是某个$RKHS$中的再生核，接下来我们根据一个给定的正定核，一步步解释如何构造一个对应的再生核希尔伯特空间$\mathcal {H_K}$。有如下关键点需要注意：

- 空间需要包含核函数所定义的所有性质，还要使得核函数$K$成为这个空间的再生核。即核函数在这个空间中满足再生性质：
  $$
  f(\mathbf x)=\langle f,K(\mathbf x,\cdot)\rangle_{\mathcal {H_k}}
  $$

​	我们首先可以先定义$\mathcal H$下的元素空间$V$，再定义点积和范数，然后引入完备性来构造一个$RKHS$使得$K$是该空间的再生核。（这几个性质可参考章节1中的空间关系图理解）我们首先定义空间$V$，方法是从集合$S=\set {k_\mathbf x:\mathbf x \in \mathbf x}$开始构造，其中$k_\mathbf x(\mathbf y)=K(\mathbf x,\mathbf y)$可以理解为一个固定住了变量$\mathbf x$的情况下的二元函数，接着定义$V$是集合$S$中所有元素的线性组合，即$V$中的所有元素都可以写成$\sum_{i} \alpha_ik_{\mathbf x_i}$，接着我们可以定义$\mathcal{H_K}$点积如下：
$$
\langle k_\mathbf x,k_\mathbf y\rangle_{\mathcal {H_K}}=\langle\sum_i \alpha_i k_{\mathbf x_i},\sum_j\beta_jk_{\mathbf x_j}\rangle_{\mathbf x} \tag{2.5}
$$
​	由于核函数$K$的性质，我们可以将上述公式写成：
$$
\langle k_\mathbf x,k_\mathbf y\rangle_{\mathcal {H_K}} = \sum_i \sum_j \alpha_i\beta_jK({\mathbf x_i},{\mathbf y_j}) \tag{2.6}
$$
​	现在空间$V$并不是完备的，完备性是指一个空间中所有的柯西序列（Cauchy sequence）都必须收敛于该空间中的某个点，但现在存在某些柯西序列的极限点并不在$V$中，不过我们可以**将存在极限点**的柯西序列添加到空间$V$中形成一个更大的空间$V'$，扩展后的空间$V'$就满足了柯西序列的完备性，同时保持希尔伯特空间的性质。	

​	**柯西序列**（Cauchy sequence）是一个序列，其元素逐渐“聚集”在一起，无论它是否收敛于某个极限点。即一个数列$\set {\mathbf x_n}$被称为**柯西序列**，如果对于任意给定的正数$\epsilon >0$，都存在一个正整数$N$，使得当$n,m>N$时$d(\mathbf x_n,\mathbf x_m)<\epsilon$，如数列$\set{1/n},n=1,...,\infin$就是一个柯西序列。在上一步我们将存在极限点的柯西序列添加到了$V$，我们可以函数点到点差异来验证这些存在极限点的柯西序列的点对点之间的差异是有界的：
$$
\left| f_n(\mathbf x)-f_m(\mathbf x)\right|=\left|\langle f_n-f_m,K(\mathbf x,\cdot) \rangle\right|\leq K(\mathbf x,\mathbf x)\lVert f_n-f_m \rVert_2 \tag{2.7}
$$
​	单纯的差异被控制并不能直接推出柯西序列的收敛，但它确保了点对点的函数值是受控的，不会无界发散。总而言之，即使 $V$中本身的柯西序列不收敛，我们可以通过引入所有柯西序列的极限点来构造一个完备空间，使得所有柯西序列都能在扩展后的空间中收敛。

​	现在拓展后的空间$V'$是完备的，**不禁引出一个问题**：即原来的点积操作在空间拓展以后是否是有效的。因为添加的存在极限的柯西序列并不一定能由原始的$V$中的元素的线性组合即$S=\set{k_\mathbf x:=\sum_i \alpha_ik_{\mathbf x_i}}$表达。我们是否需要更进一步完善原有的点积的定义，确保内积操作在扩充后的$\mathcal {H_K}$？

​	答案是否定的，我们不需要重新定义内积，首先，$RKHS$中的点积$\langle f,h \rangle_{\mathcal {H_K}}$已经在$\mathcal V$中定义了，可以利用柯西序列的极限将内积从$\mathcal V$中的有限线性组合延展到扩展后的空间$\mathcal {V'}$中：若$\set{f_n}$和$\set{g_n}$是存在极限的柯西序列，其极限分别是$f,g$，那么有：
$$
\langle f,g\rangle_{\mathcal {H_K}}:=\lim_{n\rarr \infin,m \rarr \infin} \langle f_n,g_m \rangle_{\mathcal {H_K}} \tag{2.8}
$$
​	对于任何$ f,g \in V$延拓后的内积值与原始定义值一致，对于拓展后的空间，他们的内积值是通过序列一致极限定义的，与原始内积有连贯性。

### 2.2.1 特征函数

​	一个函数可以视作一个无穷维的向量，而对于有两个变量的和函数而言，其可以看作一个无穷维度的矩阵。在线性代数中，一个矩阵$M$的特征向量$v$使得$Mv=\lambda v$，$\lambda$是相应的特征值。对于函数而言，同样也有其对应的特征函数，如果$K$是一个核函数，则$\phi $是该核函数的特征函数，如果满足如下条件：
$$
\int K(\mathbf x,\mathbf x')\phi(\mathbf x')\operatorname{d}\mathbf x'=\lambda\phi(\mathbf x) \tag{2.9}
$$
​	写成函数内积的形式，即$\langle K(\mathbf x,\cdot),\phi\rangle=\lambda\phi(\mathbf x)$，[Mercer-Hilbert-Schmit theorems[7]]()说的是，如果$K$是一个正定核，那么关于$K$存在一个无穷维的特征函数序列$\set {\phi_i}_{i=0}^{\infin}$和特征值序列$\set {\lambda_{i}}_{i=0}^{\infin},(\lambda_1\ge \lambda_2 \ge,...)$可以将$K$表示成：
$$
K(\mathbf x,\mathbf x')=\sum_{i}^{\infin}\lambda_i \phi_i(\mathbf x)\phi_i(\mathbf x') \tag{2.10}
$$
​	这就像矩阵由其特征值和特征值向量的表达一样，只不过维度拓展到了无穷维。和不同下标特征向量的互相正交一样，我们有$\langle \phi_i,\phi_j\rangle=0,\text{for }i\neq j$。因此，$\set {\phi_i}_{i=0}^{\infin}$是函数空间的一组正交基。

### 2.2.2 重新定义内积

​	现在假设对于一个核$K$，我们有特征函数序列$\set {\phi_i}_{i=0}^{\infin}$和特征值序列$\set {\lambda_{i}}_{i=0}^{\infin}$，从$L_2$空间出发，对于任意函数$f \in L_2$，我们将用其在特征函数下的系数表示：
$$
f_i=\langle f,\phi_i\rangle_{L_2}=\int f(\mathbf x)\phi_i(\mathbf x)\operatorname{d}\mathbf x \tag{2.11}
$$
​	而在$K$对应的空间下，函数本身$f\in \mathcal V'$可以表示成$\begin{aligned}f=\sum_{i=0}^{\infin} f_i\sqrt{\lambda_i}\phi_i\end{aligned}$。而核函数固定变量$\mathbf x$时有$\begin{aligned}K(\mathbf x,\cdot)=k_{\mathbf x}=\sum_{i=0}^{\infin}\lambda_i \phi_i(\mathbf x)\phi_i\end{aligned}$，我们可以定义$RKHS$下的内积运算：
$$
\langle f,f'\rangle_{\mathcal H_{K}}:=\sum_{i=0}^{\infin}\frac{f_if_i'}{\lambda_i} \tag{2.12}
$$
​	这里重申一下为什么要再次定义内积，以及为什么这么定义。首先：之前的点积定义（例如在Hilbert空间中）是一般的，但在RKHS中，点积需要与核函数$K$的特定属性一致。因此，需要重新定义点积以满足这些要求。再生核希尔伯特空间的核心在于满足以下两个关键性质：

1. 再生性：对任意函数$f \in \mathcal H$和任意$\mathbf x \in \mathcal X$，有$f(x)=\langle f,k_{\mathbf x}\rangle_{\mathcal H}$。
2. 核的内积性质：核函数本身的值通过点积表达：$K(\mathbf x,\mathbf x')=\langle k_\mathbf x,k_{\mathbf x'}\rangle_{\mathcal H}$。

​	而现在我们需要验证一下公式$2.12$的定义是否能满足如上两个关键点，即内积的定义能使得运算满足公式$2.10$，证明如下：将$k_{\mathbf x}$与$k_{\mathbf {x'}}$展开到特征基$\set{\phi_i}_{i=0}^{\infin}$上：
$$
k_{\mathbf x}=\sum_{i=0}^{\infin}\lambda_i \phi_i(\mathbf x)\phi_i,\text{ } k_{\mathbf {x'}}=\sum_{i=0}^{\infin}\lambda_i \phi_i(\mathbf x')\phi_i .\tag{2.13}
$$
​	代入公式$2.12$点积的定义，则点积计算的结果为：
$$
\begin{aligned}\langle k_{\mathbf x},k_{\mathbf x'}\rangle_{\mathcal {H_K}}&=\sum_{i=0}^{\infin} {\lambda_i}^2\phi_i(\mathbf x)\phi(\mathbf x')/\lambda_i \\
&=\sum_{i=0}^{\infin} {\lambda_i}\phi_i(\mathbf x)\phi(\mathbf x') \\
&=K(\mathbf x,\mathbf x')\end{aligned}
$$
​	因此，符合核函数值通过点积来表达的条件。再生性质能直接推导出内积性质，内积性质也能推导出再生性质，因此满足了内积性质也就满足了再生核希尔伯特空间的核心条件。

### 2.2.3 特征空间

​	我们已经给出了一个详细的构造过程，表明给定一个核函数$K(\mathbf x,\mathbf y),\mathbf {x,y} \in \mathcal X$，我们能找到一个希尔伯特空间，使得核函数成为其的再生核，更进一步地说，我们可以找到一个特征函数$\Phi:\mathcal X \rarr \mathcal H$使得：
$$
K(\mathbf x,\mathbf {x'})=\langle \Phi(\mathbf x),\Phi(\mathbf x')\rangle_{\mathcal {H}} \tag{2.14}
$$
​	上述公式其实就是我们最想要的结果。这个公式的含义是给定一个对称的正定核函数$K$，存在一个映射$\Phi$使得核函数在点$\mathbf x$与$\mathbf x'$处的值等于希尔伯特空间中的两个核向量（可能是无穷维的）$\Phi(\mathbf x)$与$\Phi(\mathbf x')$的内积。我们让$\mathcal {H_K}$视作特征空间，并简单定义$\Phi(\mathbf x)=K(x,\cdot)$，利用核的可再生性，我们有：
$$
\langle \Phi (\mathbf x),\Phi(\mathbf x')\rangle_{\mathcal {H_K}}=\langle K(\mathbf x,\cdot),K(\mathbf {x'},\cdot)\rangle_{\mathcal {H_K}}
$$
​	完美符合$\Phi$的要求，而$\begin{aligned}K(\mathbf x,\cdot)=k_{\mathbf x}=\sum_{i=0}^{\infin}\lambda_i\phi_i(\mathbf x)\phi_i\end{aligned}=(\lambda_0\phi_0(\mathbf x),...,\lambda_k\phi_k(\mathbf x),...)^T$，带入2.12定义的内积公式，有：				
$$
\begin{aligned}\langle \Phi (\mathbf x),\Phi(\mathbf x')\rangle_{\mathcal {H_K}}&=\langle K(\mathbf x,\cdot),K(\mathbf {x'},\cdot)\rangle_{\mathcal {H_K}} \\ 
&=\langle k_{\mathbf x},k_{\mathbf {x'}}\rangle_{\mathcal {H_K}} \\
&=\sum_{i=0}^{\infin} \lambda_i^2/\lambda_i \phi_i(\mathbf x)\phi_i(\mathbf {x'}) \\
&=K(\mathbf x,\mathbf {x'})\end{aligned}
$$
​	如此一来，我们便得到了所谓的”核技巧“，我们并不需要找到具体的映射函数，就可以计算出核函数在点$\mathbf x$与$\mathbf {x'}$处的值。如下给出一个关于“异或”问题的例子：在二维平面中有四个散落的点，其中$(0,0)$与$(1,1)$是$\times$，$(0,1)$与$(1,0)$是$\operatorname{O}$，显然在二维平面中无法通过一条直线将图中的红色点与蓝色点完全分割，现在我们借助之前的结论，考虑一个特征映射如下：
$$
\begin{aligned} \Phi &: \mathbb R^2 \rightarrow \mathbb R^3 \\
\mathbf x = \begin{bmatrix} 
x_1 \\ 
x_2 
\end{bmatrix} &\mapsto \Phi(\mathbf x) = \begin{bmatrix} 
x_1 \\ 
x_2 \\
(x_1-x_2)^2
\end{bmatrix}
\end{aligned}
$$
![image-20250131111318353](C:\Users\13664\AppData\Roaming\Typora\typora-user-images\image-20250131111318353.png)

​	此时升维后的两组点分别变成了$(0,0,0),(1,1,0)$与$(1,0,1),(0,1,1)$，在这个三维空间中我们可以通过一个超平面$z=0.5$轻易地将这两组点分割。我们现在为$\mathbf x$定义一个函数$f(\mathbf x)$：
$$
f(\mathbf x)=ax_1+bx_2+c(x_1-x_2)^2
$$
​	这个函数将输入从二维空间映射到一维标量，我们可以将函数本身记作：
$$
f(\cdot)=\begin{bmatrix} 
a \\ 
b \\
c 
\end{bmatrix}
\cdot
$$
​	一般用$f$表示，$f(\mathbf x)\in \mathbb R$的意思就是函数在点处的取值，可以通过内积计算得出：
$$
\begin{aligned}f(\mathbf x)&=f(\cdot)^{\top}\Phi(\mathbf x) \\
&:=\langle f,\Phi(\mathbf x)\rangle_{\mathcal H} \end{aligned}
$$
> [!IMPORTANT]
>
> **1.希尔伯特空间和再生核希尔伯特空间的不同？**
>
> ​	再生核希尔伯特空间是希尔伯特空间的基础上引入一个核函数。这个核函数$K(\mathbf x,\mathbf y)$是正定与对称的，使得函数$f(x)$的值可以由$f$与希尔伯特空间中的核向量的内积再生出来，即$f(x)=\langle f,K(\mathbf x,\cdot)\rangle_{\mathcal H}$。也就是说普通的希尔伯特空间只给了一些向量，并没有特殊的工具帮助你计算函数在某一点的值。

## 2.3 傅里叶变换

### 2.3.1  基

​	对于两个向量$A$与$B$而言，向量内积可以反应两个向量的相似度，同时也代表 $A$在$B$方向上的投影长度与$B$的长度的乘积：
$$
\langle A,B\rangle=|A||B|\cos\theta
$$
​	而$A$在$B$方向上的投影的长度可以表示为：
$$
|A|\cos \theta=\frac{|A|\langle A,B \rangle}{|A||B|}=\frac{\langle A,B \rangle}{|B|}
$$
​	$A$在$B$方向上的投影$A'$则为B按照一个比例因子缩放，这个比例因子就是$\frac{|A|\cos\theta}{|B|}$：
$$
A'=\frac{|A|\cos\theta}{|B|}B=\frac{\langle A,B \rangle}{\langle B,B\rangle}B
$$
​	如果$B$是单位向量，则向量内积就是$A$在$B$上投影的坐标。因此，给定一组正交基向量$\set{\mathbf e_i}_{i=0}^{n}$，则基向量可以通过线性组合张成一个空间，这个空间中的任意向量$\mathbf x$都可以被表示成：
$$
\mathbf x=\sum_{i=0}^{n}\langle \mathbf x,\mathbf e_i\rangle \mathbf e_i
$$
​	拓展到更一般的情况，当$|\mathbf e_i|\ne 1$时，有：
$$
\mathbf x=\sum_{i=0}^{n}\frac{\langle \mathbf x,\mathbf e_i\rangle}{\langle \mathbf e_i,\mathbf e_i\rangle} \mathbf e_i
$$
​	函数也有基，可以是正交的或非正交的。如果我们定义了函数基的集合，空间中的任何函数都可以分解成若干基函数的组合。函数的内积也可以用来计算某个特定基上的系数。然而，由于函数的维度是连续的，因此可能有无穷多个基函数，对于一组基函数$\set{\phi_i}$,如果$\langle \phi_i,\phi_j\rangle=0$,$|\phi_i|=1$，任何由该基函数张成的空间中的函数都可以表示成：
$$
\begin{aligned}f&=\sum_{i=0}^{\infin} f_i\phi_i=\sum_{i=0}^{\infin}\langle f,\phi_i\rangle\phi_i \\
&=\sum_{i=0}^{\infin}\int f(\mathbf x)\phi_i(\mathbf x)\operatorname{d}\mathbf x \text{ }\phi_i
\end{aligned}
$$
​	同理，当$|\phi_i|\neq1$时：
$$
\begin{aligned}f&=\sum_{i=0}^{\infin} \frac{f_i\phi_i}{\langle\phi_i,\phi_i\rangle} =\sum_{i=0}^{\infin} \frac{\langle f,\phi_i\rangle\phi_i}{\langle\phi_i,\phi_i\rangle} \\
&=\sum_{i=0}^{\infin}\frac{\int f(\mathbf x)\phi_i(\mathbf x)\operatorname{d}\mathbf x \text{ }}{\int \phi_i(\mathbf x)\phi_i(\mathbf x)\operatorname{d}\mathbf x }\phi_i
\end{aligned}
$$
​	傅里叶级数（Fourier Series）是一个数学工具，用于将周期函数分解为一系列简单的正弦和余弦函数的和。其在信号分析、音频处理、图像处理等领域都有广泛应用。傅里叶级数的基本概念如下，假设有一个周期为$T$的函数$f(x)$，其可以被展开成无穷个三角函数的和的形式：
$$
f(x)=a_0+\sum_{i=0}^{\infin}(a_n\cos{\frac{2\pi nx}{T}}+b_n\sin{\frac{2\pi nx}{T}})
$$
​	其中：
$$
\begin{aligned}a_0&=\frac{2}{T} \int_{0}^{T} f(x)\operatorname{d}x \\
a_n&=\frac{2}{T} \int_{0}^{T} f(x)\cos({\frac{2\pi nx}{T}})\operatorname{d}x \\
b_n&=\frac{2}{T} \int_{0}^{T} f(x)\sin({\frac{2\pi nx}{T}})\operatorname{d}x
\end{aligned}
$$
​	借助欧拉公式$e^{ix}=\cos x+i\sin x$，傅里叶级数也可以写成:
$$
\begin{aligned}f(x)&=\sum_{\neg \infin}^{\infin}c_ne^{i\frac{2\pi n}{T}x}\\
c_n&=\frac{1}{T}\int_{0}^{T} f(x)e^{-i\frac{2\pi n}{T}x}\operatorname{d}x
\end{aligned}
$$
​	我们可以将$\set{\phi_{n}(x)}=\set{e^{i2\pi nx}}_{n=\neg\infin}^{\infin}$视作基函数集合，可以简单证明，任意两个基函数是正交的：
$$
\begin{aligned}\langle \phi_j,\phi_k\rangle&=\int_{0}^{T}\phi_j (x) \bar{\phi_k}(x)\operatorname{d}x \\
&=\int_{0}^{T}e^{\frac{i2\pi(j-k)}{T}}{\operatorname{d}x} \\
&=\int_{0}^{T}\cos{\frac{2\pi(j-k)}{T}}{\operatorname{d}x} \\
&=0\end{aligned}
$$
​	由$\phi_n$张成的函数空间中的函数$f$可以表示成$\phi_n$的线性组合：
$$
f(x)=\sum_{n}c_n\phi_n(x)=\sum_{n}c_ne^{i\frac{2\pi nx}{T}}
$$
​	则系数$c_n$可以表示为：
$$
\begin{aligned}c_n&=\frac{\langle f,\phi_n\rangle}{\langle \phi_n,\phi_n \rangle}=\frac{\int_{0}^{T} f(x)\bar{\phi_n(x)}\operatorname{d}x}{\int_{0}^{T}{\phi_n(x)} \bar{\phi_n(x)} \operatorname{d}x} \\
&=\frac{\int_{0}^{T}f(x)e^{-i\frac{2\pi nx}{T}}\operatorname{d}x}{\int_{0}^{T}1\operatorname{d}x} \\
&=\frac{1}{T}\int_{0}^{T}f(x)e^{-i\frac{2\pi nx}{T}}\operatorname{d}x\end{aligned}
$$


# 3.支持向量机

xxxxxxx






# 4.高斯过程

xxxxxxxx



# 参考文献

[[1]PRML Page-by-Page](https://www.bilibili.com/video/BV1uL411Z7pW/?spm_id_from=333.999.0.0&vd_source=e50d01fc812aa71407304f50a37ed296)

[[2]A story of basis and kernel](http://songc\mathbf y.net/posts/stor\mathbf y-of-basis-and-kernel-part-2/?utm_source=wechat_session&utm_medium=social&utm_oi=904795905495035904)

[[3]Introduction to RKHS, and some simple kernel algorithms](https://www.avishek.net/assets/papers/lecture4_introToRKHS.pdf)

[[4]From Zero to Reproducing Kernel Hilbert Spaces in Twelve Pages or Less](http://users.umiacs.umd.edu/~hal/docs/daume04rkhs.pdf)

[[5]Kernel methods in machine learning](https://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2021mva/inde\mathbf x.html)

[[6]A Story of Basis and Kernel - Part II: Reproducing Kernel Hilbert Space](http://songc\mathbf y.net/posts/stor\mathbf y-of-basis-and-kernel-part-2/)

[[7]https://en.wikipedia.org/wiki/Fourier_series](https://en.wikipedia.org/wiki/Fourier_series)

# 附录-证明

## A.Reize representation theorem



## B.Mercer-Hilbert-Schmit theorems







