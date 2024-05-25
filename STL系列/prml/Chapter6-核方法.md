# $\text{核方法}$

参考内容:[PRML Page-by-Page](https://www.bilibili.com/video/BV1uL411Z7pW/?spm_id_from=333.999.0.0&vd_source=e50d01fc812aa71407304f50a37ed296)，[A story of basis and kernel](http://songcy.net/posts/story-of-basis-and-kernel-part-2/?utm_source=wechat_session&utm_medium=social&utm_oi=904795905495035904)，[Introduction to RKHS, and some simple kernel algorithms](https://www.avishek.net/assets/papers/lecture4_introToRKHS.pdf)，[From Zero to Reproducing Kernel Hilbert Spaces in Twelve Pages or Less](http://users.umiacs.umd.edu/~hal/docs/daume04rkhs.pdf)，[Kernel methods in machine learning](https://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2021mva/index.html)

## 5.1 前言



## 5.2 相关概念

​		现代数学的一个特点就是以集合为研究对象，这样的好处就是可以将很多不同问题的本质抽象出来，变成同一个问题，当然这样的坏处就是描述起来比较抽象，很多人就难以理解了。

### 5.2.1 向量空间

​		主要研究集合的描述，关注的是向量的位置，直观地说就是如何告诉别人这个集合的样子，为了描述清楚，就引入了“基”的概念，对于向量空间，集合元素只需要知道在给定基下 的坐标就能描述出位置。

### 5.2.2 赋范空间

​		向量空间中的元素没有长度，为了量化向量空间中的元素，引入了特殊的长度“范数”，赋予了范数的向量空间又称赋范空间。

### 5.2.3 度量空间

度量空间就是指定义了度量函数的空间，对属于$L$的两个任意元素$x,y$，存在一个函数$d$满足如下性质：

（1）$d$是实值的、有限的、非负的（2）$d(x,y)=d(y,x)$（3）$d(x,y)\leq d(x,z)+d(z,y),x,y,z\in L  $

### 5.2.4 空间的完备性

​	属于空间$L$中的任意柯西列都收敛，且收敛到$L$内，柯西列的定义：设$\left\{ x_n\right\}$是度量空间$R$中的点列，如果它满足柯西准则：即若对于任意的$\epsilon >0$，存在一个数$N_{\epsilon}$使得所有的$x_{n^{'}}>N_{\epsilon},x_{n^{''}}>N_{\epsilon},d(x_{n^{'}},x_{n^{''}})<{\epsilon}$，则称为柯西列。即离得越来越近的点列，且后面两点之间的距离无穷小。完备性就是指$n \rightarrow \infin$时，数列$\left\{ x_n\right\}$极限存在，且也属于$L$。

若一个空间可以定义度量函数，且它是完备的，那么他就是完备度量空间（判断柯西列是否收敛需要计算两点间的距离，因此需要定义度量函数。）**如果一个空间既是线性的，又是赋范的，还是完备的，则是一个巴拿赫空间**。前面所述空间关系如下。

### 5.2.5内积空间

赋范空间中的元素没有角度的概念，因此又引入了内积的概念。在线性空间上引入范数和内积，就是内积空间。

### 5.2.6 希尔伯特空间

因为有度量，所以可以在度量空间、赋范线性空间以及**内积空间中**引入极限，但抽象空间中的极限与实数上的极限有一个很大的不同就是，极限点可能不在原来给定的集合中，所以又引入了完备的概念。完备的**内积空间**就是**希尔伯特空间**。

总结：**((向量空间+范数=赋范空间+线性结构)+内积=内积空间)+完备性=希尔伯特空间**，整体关系如下图所示：

![space2](src\kernel mothods\space2.png)

![RKHS-space](src\kernel mothods\RKHS-space.png)

### 5.2.7 核函数与再生核希尔伯特空间(Reproducing Kernel Hilbert Space)

**首先介绍核函数的定义：**给定$\mathcal X$和$k$，在希尔伯特空间中存在一个内积和映射$\Phi:\mathcal X \rightarrow \mathcal H$使得:

​																			$k(x,y)=<\phi(x),\phi(y)>_{\mathcal H}$，对于所有的$x,y \in \mathcal H$

在点$x$处的狄拉克求值泛函$\delta_{x}$记作$\delta_{x} \in \mathcal{H}$使得$\delta_{x}f=f(x)$,$\delta_{x}$是将函数$f$映射成$f(x)$对应的值。$\kappa$

**再生核希尔伯特空间的定义**：一个再生核希尔伯特空间是建立在希尔伯特空间$\mathcal H$上的，并且$\mathcal H$中所有的狄拉克求值泛化都是连续且有界的。

**求值泛函：**求值泛函是定义在函数空间上的函数，设有集合$L$及希尔伯特空间$\mathcal H$，且$\mathcal H$内的元素为函数$f:L \rightarrow R$，则对于任意$L$上的元素$x \in L$,求值泛函为函数$\delta_x:\mathcal H \rightarrow R$且对于任意$x$有$\delta_x(f)=f(x)$。**核的可再生性：** $Riesz~representation~theorem$理论说的是如果$\phi$是在$\mathcal H$上的(满足狄拉克求值泛函)有界的线性泛函，则有唯一的向量$u \in \mathcal H$使得$\phi(f)=<f,u>_{\mathcal H}$对于所有的$f \in \mathcal{H}$。将这个理论放到当狄拉克求值泛函的情境下翻译，那么就是对于每一个$\delta_x$存在一个$K_x \in \mathcal H$使得$\delta_x(f)=<f,K_x>_{\mathcal H}$。**再生性就是指用$\mathcal H$中的内积重新表达了求值泛函。**

![RKHS-space2](src\kernel mothods\RKHS-space2.png)

举例：设$x=\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$,$\mathcal H$中有点$f=(a,b,c)^T$使得求值泛函为:

​			                  		$\delta_x(f)=f(x)=ax_1+bx_2+cx_1x_2=(a,b,c)^T\begin{pmatrix} x_1 \\ x_2 \\x_1x_2 \end{pmatrix}=<f(.),\phi(x)>_{\mathcal H}$，



