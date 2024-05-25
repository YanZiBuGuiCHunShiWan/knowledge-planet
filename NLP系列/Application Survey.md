

## $\text{perceiverIO}$

​		在开始之前，先约定好数据的形状。符号分别如下：

​								$\begin{aligned}\text{Input array}:&\mathbf X=\begin{pmatrix} \mathbf x_1^T \\ {\vdots} \\ \mathbf x_k^T \end{pmatrix}  \text{      Latent array:}\mathbf Z=\begin{pmatrix} \mathbf z_1^T \\ {\vdots} \\ \mathbf z_m^T \end{pmatrix} \text{      Output query array:}\mathbf Z'=\begin{pmatrix} \mathbf {z'_1}^T \\ {\vdots} \\ \mathbf {z'_n}^T \end{pmatrix} \\&\mathbf x_i \in \mathbb R^{kv_{dim}\times 1} ,\mathbf z_i \in \mathbb R^{q_{dim}\times 1},\mathbf {z'_i} \in \mathbb R^{kv_{dim}\times 1}\end{aligned}$

​		因为$k,v$实际上是同一个东西,所以用$kv_{dim}$表示其维度,而query的维度和$k,v$其实没有关系,所以用$q_{dim}$表示.

​	$\text{step1}.$输入矩阵和隐状态矩阵通过$\text{cross attention}$进行语义融合。首先要初始化三个$\mathbf {W^Q,W^K,W^V}$初始化矩阵:

​							$\begin{aligned} \mathbf W^Q \in \mathbb R^{q_{dim} \times qk_{channels} }, &\mathbf W^K \in \mathbb R^{kv_{dim} \times qk_{channels} }, \mathbf W^V \in \mathbb R^{kv_{dim} \times v_{channels} } \\&Q=\mathbf {ZW^Q}\in\mathbb R^{m \times qk_{channels}} \\ &K =\mathbf {XW^K}\in \mathbb R^{k\times qk_{channels}}\\&V =\mathbf {XW^V}\in \mathbb R^{k\times v_{channels}} \\ &QK^T=\mathbf{ZW^Q{W^K}^T X}\in \mathbb R^{m\times k}\\ &\operatorname{softmax(QK^T/\sqrt d)V \in \mathbb R^{{m \times v_{channels}}}} (\text{此处省略multi-head})\end{aligned}$

​        从最终输出的结果可以看到，$\text{Cross attention}$的输出在$\text{seqlen} (\text{time index})$和$\text{Latent array(Query)}$保持一致,在$\text{hidden dim}$上会和$v_{channels}$保持一致($v_{channels}可以自己设置$).而接下来的自注意力模块的$Q,K,V$都来自$\text{Cross attention}$的输出,形状也不会改变.所以整个Encoder 模块的输出的形状就是$\text{Output}_{Encoder}\in \mathbb R^{m\times v_{channels}}$.

​		我们来看一下源码的具体实现:(transformers/models/perceiver/modeling_perceiver.py) 

![image-20240513134513822](C:\Users\13664\AppData\Roaming\Typora\typora-user-images\image-20240513134513822.png)

​			对于最开始的$\text{Cross attention}$,初始化时要指定$q_{dim},kv_{dim}$和$qk_{channels},v_{channels}$.而PerceiverLayer类的实现最终会定位到PerceiverSeflAttention(transformers/models/perceiver/modeling_perceiver.py).对应的代码是:

```python
class PerceiverSelfAttention(nn.Module):
    """Multi-headed {cross, self}-attention. Can be used both in the encoder as well as in the decoder."""

    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = q_dim
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels
        if qk_channels % num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({num_heads}).")
        if v_channels % num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({num_heads}).")

        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.qk_channels_per_head = self.qk_channels // num_heads
        self.v_channels_per_head = self.v_channels // num_heads
        #print("Perceiver Self attention 要求：layernorm1 q_dim:{}".format(q_dim))
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(q_dim)
        #print("Perceiver Self attention 要求：layernorm2 KV_dim:{}".format(kv_dim))
        self.layernorm2 = nn.LayerNorm(kv_dim) if is_cross_attention else nn.Identity()

        # Projection matrices
        self.query = nn.Linear(q_dim, qk_channels) #[q_dim,qk_channels]
        self.key = nn.Linear(kv_dim, qk_channels) #[kv_dim,qk_channels]
        self.value = nn.Linear(kv_dim, v_channels) #[kv_dim,v_channels]

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        #print("进入Perceiver Self Attention 输入形状：{}".format(inputs.shape))
        hidden_states = self.layernorm1(hidden_states)  #hiddent state 就对应于Latent array #[Batch,m,q_dim]
        inputs = self.layernorm2(inputs) #shape [Batch,k,kv_dim]
        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            '''交叉注意力机制'''
            keys = self.key(inputs) #[Batch,k,qk_channels]
            values = self.value(inputs) #[Batch,k, v_channels]
            attention_mask = inputs_mask  #[Batch,max_seqlen]
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head) 
        values = self.transpose_for_scores(values, self.v_channels_per_head) 

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)  #[Batch,q index len, kv index len]

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values) #attention矩阵乘以values [Batch,num_heads,q index len,v_channels] 一般情况下v_channels = q_dim

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #[Batch,q index len,num_heads,v_channels] 一般情况下v_channels = q_dim
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)   #[Batch,q index len,num_heads*v_channels_per_head] 

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

```

模型推理验证:  可以看到 输入X形状是[16,175,768],Latent array形状是[16,256,512],cross attention 和Encoder最终的输出形状都是[16,256,512].注意:实际上$v_{channels}得和{q_{dim}}$保持一致,因为残差链接的原因.所以Encoder的输出形状和Latent array一致.

![image-20240513140931999](C:\Users\13664\AppData\Roaming\Typora\typora-user-images\image-20240513140931999.png)

同样,对于Decoder而言,其输出会和Output Lantent array一致(具体流程和Encoder中一样,故次省略).所以,如果是用于Classification任务,那么Output Latent array的形状就应该是:[Batch,1,hidden dim].这样Decoder的输出就会变成[Batch,1,hidden dim],squeeze掉第1维以后再接上用于分类的现象层得到最终的输出形状就是[Batch,num classes].

![image-20240513141602338](C:\Users\13664\AppData\Roaming\Typora\typora-user-images\image-20240513141602338.png)

