

# 多模态特征融合



# 1. Introduction



# 2.Attentional Feature Fusion



# 3.perceiver

​		$\text{PerceiverIO}$就是$\text{Perceiver}$的进阶版本，其比在原有的编码器结构上做出了一定的调整，最大的变化是多了一个由$\text{cross attention}$构成的$\text{Perceiver}$结构用于解码编码器的语言信息。其结构如下图：

![PerceiverIO](src\Transformer\PerceiverIO.png)

​		为了更具体地帮助理解输入数据经过$\text{PerceiverIO}$形状是如何改变的，接下来给出一段矩阵公式推导。在开始之前，先约定好数据的形状，符号分别如下：

​       		$\begin{aligned}\text{Input array}:&\mathbf X=\begin{pmatrix} \mathbf x_1^T \\ {\vdots} \\ \mathbf x_k^T \end{pmatrix}  \text{      Latent array:}\mathbf Z=\begin{pmatrix} \mathbf z_1^T \\ {\vdots} \\ \mathbf z_m^T \end{pmatrix} \text{      Output query array:}\mathbf Z'=\begin{pmatrix} \mathbf {z'_1}^T \\ {\vdots} \\ \mathbf {z'_n}^T \end{pmatrix} \\&\mathbf x_i \in \mathbb R^{kv_{dim}\times 1} ,\mathbf z_i \in \mathbb R^{q_{dim}\times 1},\mathbf {z'_i} \in \mathbb R^{kv_{dim}\times 1} \\ &\mathbf W^Q \in \mathbb R^{q_{dim} \times qk_{channels} }, \mathbf W^K \in \mathbb R^{kv_{dim} \times qk_{channels} }, \mathbf W^V \in \mathbb R^{kv_{dim} \times v_{channels} } \\\end{aligned}$

​		下标$dim$对应了矩阵的行，下标为$channels$对应矩阵的列。因为$k,v$实际上是同一个东西,所以用$kv_{dim}$表示$\mathbf W^Q$的行维度,而$query$的维度和$k,v$其实没有关系,所以用$q_{dim}$表示$\mathbf W^Q$的列维度。由于$QK^T$计算，所以需要保证$\mathbf W^Q$和$\mathbf W^K$的列数相同，所以用二者的列数都记作$qk_{channels}$，最后$\mathbf V$的维度没有显示地要求，所以$\mathbf W^V$的列数用$v_{channels}$表示。

​	$\text{step1}.$输入矩阵和隐状态矩阵$\text{Latent array}$通过$\text{cross attention}$进行语义融合。输入$\mathbf X$经过三个$\mathbf {W^Q,W^K,W^V}$投影矩阵得到${Q,K,V}$:

​                                      $\begin{aligned}&Q=\mathbf {ZW^Q}\in\mathbb R^{m \times qk_{channels}} \\ &K =\mathbf {XW^K}\in \mathbb R^{k\times qk_{channels}}\\&V =\mathbf {XW^V}\in \mathbb R^{k\times v_{channels}} \\ &QK^T=\mathbf{ZW^Q{W^K}^T X}\in \mathbb R^{m\times k}\\ \text{cross att output}=&\operatorname{softmax(QK^T/\sqrt d)V \in \mathbb R^{{m \times v_{channels}}}} (\text{此处省略multi-head}操作)\end{aligned}$

​        从最终输出的结果可以看到，$\text{Cross attention}$的输出在$\text{seqlen} (\text{time index})$和$\text{Latent array(Query)}$保持一致，在$\text{hidden dim}$上会和$v_{channels}$保持一致($v_{channels}可以自己设置$)。

​       $\text{step2.}$而接下来的自注意力模块的$Q,K,V$都来自$\text{Cross attention}$的输出,形状也不会改变.所以整个Encoder 模块的输出的形状就是$\text{Output}_{Encoder}\in \mathbb R^{m\times v_{channels}}$。

​		我们来看一下源码的具体实现:$\text{(transformers/models/perceiver/modeling\_perceiver.py) }$对于最开始的$\text{Cross attention}$,初始化时要指定$q_{dim},kv_{dim}$和$qk_{channels},v_{channels}$。而$\text{PerceiverLayer}$类的实现最终会定位到$\text{PerceiverSeflAttention(transformers/models/perceiver/modeling\_perceiver.py)}$。对应的代码是:

​		$\text{step3.}$我们不妨来进行模型推理验证:  可以看到 输入$\mathbf X$形状是$[16,175,768]$,$\text{Latent array}$形状是$[16,256,512]$,$\text{cross attention}$ 和$\text{Encoder}$最终的输出形状都是$[16,256,512]$.注意:实际上$v_{channels}得和{q_{dim}}$是保持一致的，虽然表明上没有显示地说明，但是因为残差链接的原因$Q+\text{cross att output}$操作需要二者维度一直。所以$\text{Encoder}$最终的输出形状和$\text{Latent array}$是一致的。

​		$\text{step4.}$同样,对于$\text{Decoder}$而言,其输出在交叉注意力机制的影响下会和一致(具体流程和$\text{Encoder}$中一样,故次省略).所以,如果是用于$\text{Classification}$任务,那么$\text{Output Lantent array}$的形状就应该是:$[\text{Batch},1,\text{hidden dim}]$.这样$\text{Decoder}$的输出就会变成$[\text{Batch},1,\text{hidden dim}]$,squeeze掉第1维以后再接上用于分类的线性得到最终的输出形状就是$[\text{Batch},\text{num classes}]$​。











