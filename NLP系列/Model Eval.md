# LLM Evaluation

```模型评估```

​        大语言模型的出现为自然语言处理领域解决任务的方法提供了新的思路，从传统的基于静态词向量到预训练模型BERT再到现在的LLM，自然语言处理领域在飞速发展。LLM重构自然语言处理解决下游任务的范式的同时也带来一个至关重要的问题-如何衡量LLM的好坏？

​       与传统的文本分类、命名实体识别任务不同，大语言模型的输出内容千变万化，同样的语义对应的文本也可能会截然不同，其次，像经过RLHF以后的大语言模型能解决各式各样的下游任务，而现有的评估方式没有办法较好地衡量大语言模型生成内容的质量。传统的大语言模型评估数据集如[MMLU]()和[HELM]()没法较好地区分基座模型和对齐后的模型（因为对齐后的模型可以解决各种各样的问题，没对齐的模型相当于具备了庞大的知识储备，在MMLU，HELM这种做知识问答的数据集上二者都可以正确回答大部分问题，二者能力的体现拉不开差距）。因此，这也说明了现有的衡量大语言模型的方法与用户实际感知之间存在着差异[[1]](https://cameronrwolfe.substack.com/p/llm-as-a-judge)。

# 1.LLM as a Judge

​	LLM-as-a-Judge这篇论文则针对现有测评方法的不足展开了更全面的研究，提出了两个测评基准[MT-Bench]()和[Chatbot Arena]()，前者涵盖了一系列开放领域的问答和多轮对话以及指令遵循能力的测试问题，后者则更偏向于是一个众包平台，用户可以同时和两个匿名的机器人聊天并根据个人喜好进行评分。文中指出现有的测评基准可以分为如下三类：

- **Core-knowledge benchmarks**，如[MMLU]()，[ARC]()，[HumanEval]()等，这类型测试通过zero-shot或者few-shot的方式评估预训练大语言模型的核心能力，通常只需要让大语言模型生成一个简短的特定的回复然后进行自动测评。
- **Instruction-following benchmarks**，如[Flan]()，[Self-instruct]()等，这类测试集涵盖了更多的开放领域问答和各式各样的任务，用于衡量指令微调以后的大语言模型的能力。
- **Conversational benchmarks**，如[CoQA](),[OpenAssistant]()等，这类型测试更接近与人类真实的意图，但是在多样性和复杂性上还不足以挑战最新的聊天机器人的能力。（说白了还是太简单）

​	人工评估模型回复的结果非常昂贵和耗时，传统任务如机器翻译的评估方法是基于模型输出和参考答案的相似度进行衡量的，如指标[ROUGE]()，[BLEU]()并不能较好地衡量大语言模型的输出质量，而随着大语言模型能力的提升，其展现出了衡量评估对话助手和人类偏好的潜力。LLM-as-a-judge中提出了三种评估方法，可以单独使用或者互相结合：

- **Pairwise comparison**:**Pairwise**的方式是大语言模型会根据给定的prompt（蕴含两个回复案例）判断哪一个回复更优，而不是直接给单独的回复赋予一个分数。
- **Single answer grading**:这种方式就像**Pointwise**一样，大语言模型基于给定的回复进行打分，分值越高代表质量越好。
- **Reference-guided grading**:通过提供参考案例辅助大语言模型进行打分，相关工作有清华大学的[AlignBench]()等。

## 1.1 Example

### 1.1.1 Pairwise Comparison

![image-20241028150514391](E:\Study\知识构建\NLP系列\src\Model Evaluation\prompt_example.png)

​	一个典型的Pairwise评估方法如上图，将相同的question和两个不同的answer填入模板内让大语言模型进行比较，比较结果为三种：（1）A如果助手A回答更好，（2）B如果助手B回答更好，（3）C如果两者回答效果差不多。而Pairwsie方法虽然能比较不同回答对，但是其执行次数较多，如果有30个回答需要两两比较，那么就需要执行$\choose{}{2}$次。

### 1.1.2 Pointwise scoring(Single Answer Grading)

​	最简单的评估方式就是在prompt中要求大语言模型根据自己的判断对回复进行1~5分打分，也可以利用上更加复杂的提示工程技术辅助模型判断或是增加可解释性，比如给定明确的评分规则，按照小点进行打分，或者让模型在打分时输出其打分的理由帮助操作者判断是否合理。

### 1.1.3 Reference-guided scoring

​	提供参考答案进行评分适用于难度较高的任务上的判断，原文中提到即使使用CoT结合Pointwise scoring技术，模型在判断推理类任务时仍然会犯错。而提供了参考答案则会显著提示模型的判断能力（在数学和逻辑推理任务上的失败率从70%降低到了15%）。原文中提到，以上三种方式都可以结合思维链技术提升模型的判断能力。

### 1.1.4多轮对话评估

​	MT-Bench数据集中有用于评估大语言模型对话能力的两轮对话数据，作者探索了两种测评方式：(1)将两轮拆开，每一轮单独评分。(2)两轮合并在一起视作整体让大语言模型测评。不过作者发现第一种方法经常混淆不同对话的上文，第二种方法整体上能显著减轻这种现象。

## 1.2 Bias

​	虽然实验结果表明LLM-as-a-Judge可以完成各种各样的评估任务取得和人类相当的结果，但是其评估过程也不是完美的，比如会引入偏差。

- **Position bias**: prompt中的案例会收到其位置的影响，比如在pairwise比较中，在前面的例子的评分往往较高。
- **Verbosity bias**: 内容较长的案例往往评分较高。
- **Self-enhancement bias**: 用于测评的模型往往对自己生成的内容打分较高。

​	作者在文章说到用GPT-4测评GPT-3.5和Vicuna-13B的回答，当GPT-3.5回答在前时GPT-4会任务GPT-3.5的回答结果更详细和优越，但是把Vicuna的结果放在前面时GPT-4又任务Vicuna的答案更好，这种现象不仅仅只是在LLM-as-a-Judge的场景下出现，如果读者有丰富的提示工程经历应该能体会到提示工程是比较繁琐的，在进行Few Shot Prompting的实践时会发现提供的案例的顺序会对prompt的效果产生不小的影响，交换prompt中的两个案例顺序很可能就导致不同的结果。

​	第二点，LLM更喜欢较长的回复，即使内容质量并不高。作者设计了一种攻击手段，从MT-Bench中抽取了包含编号的列表并让23个模型生成回复，然后让GPT-4把这些模型的回复变得冗长，没有增加有用信息。然后把这些改写后的测试插入到原有的列表的前面作为新的回复。（如果模型对新的回复评分高于原有的回复，则表明模型倾向于长的回复）实验结果表明LLM确实有Verbosity Bias（但是GPT-4能较好地抵御这种攻击）。

​	第三点，作者注意到某些模型有明显的自我偏好，例如，GPT-4偏爱自己高10%的胜率；Claude-v1偏爱自己高25%的胜率。不过也有喜欢其他模型的，比如GPT-3.5没那么喜欢自己回答的结果。由于数据有限，差异较小，作者表明研究**不能确定**模型是否表现出自我增强的偏差。

### 1.2.1 如何减少Bias？

通过合适的提示工程技术，可以减少Bias的严重程度，比如：

- Few-Shot CoT，在一些复杂的任务上我们可以通过少样本学习结合思维链技术来辅助模型进行判断，从而减少Bias。
- 位置打乱，比如进行多次评价，每一次评价时模型输出在prompt中的位置都不一样，然后把几次评价的结果综合作为最终的结果。
- 多模型投票，比如引入多个不同LLM进行评估并综合所有模型的结果。

# 2.相关工作

## 2.1 ALpacaEval

​	[AlpacEval]()是最有代表性的经过指令微调的大语言模型评估指标之一，通过[AlpacaFarm]()的模型自动化评估方法（win rate），研究人员可以低成本且高效地评估大语言模型的能力。作者从不同数据集种构建了805个指令（Self-Instruct,Open-assistant,Anthropics,etc），然后针对要对比的两个模型收集回复，并用一个LLM 充当评估者$p_{sim}^{eval}$对每一个模型的输出进行评分，从而计算两个模型输出的输赢率(win-rate)对比。同时文中也表明，用于评估的LLM的一致性为65%，而多人投票的一致性为66%，表明$p_{sim}^{eval}$的评估结果和人类偏好高度一致，同时评估的开销也从300$\text{\$}$每1000个样例降低到12$\text{\$}$每一千个样例。由于AlpacEval的高效和低成本，将其用于模型开发（每次训练完后和之前的版本对比）是十分方便的。

​	用于测评AlpacEval的prompt如下图所示。对每一个测试指令来说一对答案（来自两个模型）会被填充到prompt的槽中，由评估模型输出结果——0/1打分或者是模型输出的对数概率$logprobs$，通过计算整个测评数据集上的概率的平均值可以得出模型A回复由于模型B的胜率。（比如$(0.1+0.2+0.3+0.1+0.4+0.7)/6=0.3$，对应的**win-rate**就是$0.3$​。）

![image-20241029160342422](E:\Study\知识构建\NLP系列\assets\image-20241029160342422.png)

​										          	[源自[4]](https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/alpaca_eval_gpt4/alpaca_eval.txt)	

​	**减少Verbosity Bias**：模型评估虽然较于方便，但在评估过程中可能会引入一些偏差，而消除这些偏差或者解释这些偏差可以更好地帮助开发者公正地评估大语言模型。大语言模型评估最主要的偏差就是长度偏差——LLM会倾向于给更长的文本更高的评分，因此原始的AlpacaEval在即使内容差不多甚至长文本的质量要更差的情况下也会更长回复更高的评分。

​	为了消除这种偏差，研究者提出了一种简单的基于回归的偏差消除方法——训练一个包含三个属性作为变量的对数线性模型：(1)模型，（2）LLM的输出长度，（3）指令的困难程度。具体公式如下：

​			
$$
\begin{aligned}\begin{array}{l}
q_{\theta, \phi, \psi}\left(y=m \mid z_{m}, z_{b}, x\right):= \operatorname{logistic}(\underbrace{\theta_{m}-\theta_{b}}_{\text {Model }}+\underbrace{\phi_{m, b} \cdot \tanh \left(\frac{\operatorname{len}\left(z_{m}\right)-\operatorname{len}\left(z_{b}\right)}{\operatorname{std}\left(\operatorname{len}\left(z_{m}\right)-\operatorname{len}\left(z_{b}\right)\right)}\right)}_{\text {Length }}+\underbrace{\left(\psi_{m}-\psi_{b}\right) \gamma_{x}}_{\text {Instruction }})
\end{array} \end{aligned} \tag{1.1}
$$
​	让模型$b$作为基准，如果有$M$个待测评模型和$N$个指令，那么简单线性回归模型的参数量就是$3M+N$，训练样本量就是$MN$，（每一对模型$m和b$需要算出$\theta,\phi_m,\psi_m$，$\phi_b,\psi_b$被吸收了，每一个指令要算出$\gamma_x$，$\gamma_x$是模型间共享的）。作者采用了分步优化的方式学习不同的参数，具体如下：

1. 先学习$\gamma_x$，对于每一个指令$x$，有$M$个用于训练的样本，另$\psi_m-\psi_b=1$然后根据上述公式进行优化可以学习到$\gamma_1,\gamma_2,...\gamma_M$。

2. 再依赖于上一步的$\gamma_x$学习$\theta_m,\phi_{m,b},\psi_m$。每一个模型分开学习，即对于模型$1$和模型$b$训练学习出$\theta_1,\phi_{1,b},\psi_1$，以此类推，即每一个模型不会和其他模型产生关联。

	​	分开学习的好处是模型间不彼此依赖，即有新的模型需要测评时，参数不会依赖于之前已经测评过的模型。当参数学习好以后，我们就可以将指定的变量置为0，判断在剩余的变量的作用下，模型最终的得分是多少。作者将长度这一项变量置为$0$以后计算了在AlpacaEval上的win rate，结果如下图所示[[5]](https://arxiv.org/pdf/2404.04475)：

   ![image-20241030103716966](E:\Study\知识构建\NLP系列\assets\image-20241030103716966.png)

​	作者对比了3种不同的情况：（1）在保留原有信息的同时越精简越好(concise)，(2)普通的情况，(3)回答越详细越好。左边的普通AlpacaEval测试上表明几乎所有的模型在精简和详细的回答模式下的win-rate差异显著，详细的回答得分明显高于精简回答。而在长度控制的AlpacaEval上模型的win-rate则相对稳定，三种不同模式下的标准差从25%降低到了10%。表格中的数据是不同模式下的win-rate，比如最左上角的$22.9$，就是模型$\text{gpt4\_1106\_preview}$在精简的prompt下和标准prompt下的win-rate。旁边的$50.00$就是两个标准prompt下的win-rate，带入公式1有$\operatorname {logistic}(0)=\frac{1}{2}$，故win-rate是50%。$\text{Length-controlled AlpacaEval}$可以更好地评估模型结果，通过prompt让模型输出内容精简或者详细不会显著地改变结果，同时和评估结果也展示出了很人工评估的高度相关性，和Chatbot arena评估结果的斯皮尔曼相关系数从0.94增加到了0.98。

## 2.2 [Align Bench]()

​	AlignBench是一个轻量级的中文模型测评数据集，涵盖了各种领域。为了更客观、公正地评估模型，每一个测试问题都提供了一个人工标注的参考，标注者被要求在网页搜索相关文献然后综合得到最终的参考，因此属于Reference-guided 的评估方法，即通过一个涵盖特定领域评分标准的prompt引导LLM作为公正的评判者对模型的回复进行评分。

![image-20241101104617007](E:\Study\知识构建\NLP系列\assets\image-20241101104617007.png)

​	AlignBench数据分布如下图[源自[6]]()所示：

​					![image-20241030143218585](E:\Study\知识构建\NLP系列\assets\image-20241030143218585.png)	

​	AlginBench用GPT-4进行评估，并设计了一种多维度校准的单点评分方法(multi-dimensional point-wise)，打分为1~10分，其prompt的设计有三点核心：

- **Point-Wise Grading&Chain-of-Thought.**先前的研究表明单点级评分与成对评分与人类评分一致性较高，但是成对评分存在位置偏且时间开销较大。因此，采用了范围从1到10的评分。由于评分的任务涉及复杂的推理任务，因此加入思维链技术提高分数的可靠性和可解释性。
- **Rule-Calibrated Referencing.**提供了一个先由GPT-4生成，再由人工标注者修改的参考答案，确保高质量和正确性。为了评分机制更加可控与可解释，提供了详细的评分规则，阐述了分数区间与参考文献中答案质量之间的关系。并将参考答案设置为8分作为标准，低于8分代表不及参考答案，高于8分代表质量更优。
- **Multi-Dimensional Analysis.**由于涵盖了各种维度的任务且不同维度要求不同，设计了一种多维评分方法来，即根据不同类型的问题设置了不同的评估维度，如写作类任务需要优先考虑创造性，逻辑推理任务主要考虑逻辑连贯性等等。

​	文中的prompt设计如下图[源自[6]]()所示：

![image-20241030160308294](E:\Study\知识构建\NLP系列\assets\image-20241030160308294.png)

​	**一致性评估**：作者采用三种方式用于衡量GPT-4评估和人类评估的一致程度：(1)Sample-level Pearson Correlation.(2)System-level Pearson Correlation.(3)Pairwise Agreement（文中并未提及细节）。同时为了比较该方法的可解释性的质量，作者进行了成对质量比较实验，给定一个问题，模型回答和参考，由GPT-4根据不同prompt给出两个解释(A和B)，有人类标注和对A和B进行评估，主要从如下3个维度评价：(1) 合理性：解释是否正确和公正 (2)可读性：解释的逻辑是否清晰，可理解和详细 (3)一致性：解释是否和最终的评分一致。最终的实验情况如下  [源自[6]]()：

![image-20250413130111830](E:\Study\gitpro\knowledge-planet\NLP系列\assets\image-20250413130111830.png)

​	结果标明规则校准的单点评估方式和人类专家的一致性比其他方式要高。（基于规则的解释质量优于general，证明了评分规则可以提供一个清晰的参考基础标准，从而有助于参考答案和模型答案比较。）开发人员可以从[此处]()提交本地模型的回答结果进行自动化测评，经笔者尝试，发现该测评数据集中的问答难度较高，采用开源百亿级别参数的模型进行测评综合得分只有6分上下，未到达参考评分8分标准。

## 2.3 [Prometheus](https://arxiv.org/pdf/2310.08491)

​	已有许多工作探索了利用闭源专家级大语言模型进行模型评估的可行性，虽然闭源模型能力比开源模型强，但是使用闭源模型也有其局限性，比如：(1)某些领域因商业原因避免敏感数据泄露。(2)想要大规模评估时调用闭源模型api的费用不菲。因此有相关研究者尝试训练开源的LLM用于模型评估，[Prometheus](https://arxiv.org/pdf/2310.08491)就是最具代表性的工作。研究者构建了包含上千种个性化的细粒度评分标准的数据集$\text{FEEDBACK COLLECTION}$，同时者也是第一个利用了不同参考来有效引导大语言模型细粒度评估的工作。(属于直接评估反馈数据集$\text{direct assessment feedback dataset}$，即没有成对的比较或不同反馈的偏好关系)

​	研究者在构建数据集时主要考虑了四点核心：(1)尽可能多的涵盖参考材料（参考回答与评分准则） (2)为了防止长度偏差，每一个评分(1-5)参考长度一致 。(3)为了防止决策偏差，分数的分布一致。(4)限制指令与对应回复的范围。（更贴近用户与LLM交互的真实场景），数据集具体信息如下图[源自[7]](https://arxiv.org/pdf/2310.08491)所示：

![image-20241105140656758](E:\Study\知识构建\NLP系列\assets\image-20241105140656758.png)	最终构建的数据集中每一个实例由$input$和$output$两部分构成，其中输入部分包括四个组件($\text{instruction}$,$\text{response to evaluate}$,$\text{custom rubrics}$,$\text{reference answer}$)，输出部分包括两个组件($\text{feedback}$,$\text{score}$)，如下图[源自[7]](https://arxiv.org/pdf/2310.08491)所示：

​				    ![image-20241107172532433](E:\Study\知识构建\NLP系列\assets\image-20241107172532433.png)

​	**Absolute Grading** 为了衡量经过数据集微调后的LLM是否有良好的评估能力，采用了两种方式进行测试。首先是Absolute Grading，绝对评分的特点在于无需和另一组模型生成的回复进行比较而只依赖于当前评估模型内部的判断。为了衡量LLM是否真的具备良好的评估能力有三点关键因素必须考察：(1)和人类的评估相关系性。(2)通过人工比较反馈的质量如何。(3)和GPT-4的评估相关性。即模型的衡量方式是否和人类一致或是可信赖的闭源专家级模型一致，这也是模型评估类论文中的关键考察因素，如果相关系数低，表明模型评估与人类都不具备一致性，说明其模型评估结果是不值得信赖的。

​	**Ranking Grading** 为了测试只经过绝对评分方式训练的模型是否能担任通用的奖励模型，作者在几个人类偏好基准上用准确率衡量指标做了实验，具体地，作者测试$LLM_{eval}$是否能给予人类偏好的回应更高的分数，由于给定两个差异不那么大的候选答案时模型很可能会给出相同的分数，作者将$LLM_{eval}$的温度系数设置为$1.0$​并迭代评估每一个候选答案直到有一个胜出。

![image-20241106160954802](E:\Study\知识构建\NLP系列\assets\image-20241106160954802.png)

​	$\text{Prometheus}$通过构造直接反馈评估形式（Point-Wise Scoring）的数据集训练开源大语言模型的方式打造了个性化的LLM评估器，而直接反馈评估数据集虽然能帮助大语言模型根据模型生成的回复进行打分和反馈，但可能存在无法较好地适配于成对回复比较优劣的局限性。基于此，作者在原有的$\text{FEEDBACK COLLECTION}$基础上进行了改造，将评分为$1-5$的$5$个样本两两配对形成新的样本，共计$10$个可能性的组合，通过原有的分数可以判断哪一个回答更好（i.e. “回答A比B好”或者“回答B比A好”），并通过提示工程引导$\text{GPT}$识别两个回答间的共性和差异的方式生成新的口头形式的反馈$v_{r_m,r_n}$构成了新的反馈数据集$\text{PREFERENCE FEEDBACK}$​，使得在该数据集上微调后的模型既能以直接评估的方式进行单点打分，也能接受成对的输入进行比较。

> [!IMPORTANT]
>
> Prometheus2采用了Score-rubric-based Point-Wise Scoring形式的训练数据和Score-rubric-based Pairwise Comparison形式的训练数据，使得模型评估时既能进行单点打分也可以做成对比较。单点打分的形式依赖于模型内部的知识和对规则的理解，当不同模型的回答评估分数一致时无法进一步比较优劣。而单纯的无规则约束的成对比较方式模型内部的评判机制难以量化或解释，且存在明显的Position-bias和Verbosity-bias现象，通过规则校准的成对比较方式则能进一步提示模型评估的解释性和稳定性。

​	作者探索了四种不同的$LLM_{eval}$​的方式，分别是(1)$\text{Prompting}$​ 提示工程的方式直接让未经反馈数据集训练过的大语言模型进行判断。(2) $\text{Single-Format Training}$​ 单一形式的训练，即选择一个基础模型$\theta$​，只在直接评估反馈数据集$D_d$​上或者成对排序数据集$D_p$​上训练。(3)$\text{Joint Training}$​ 联合训练，即选择一个基础模型$\theta$​，在直。接评估反馈数据集$D_d$​上和成对排序数据集$D_p$​上训练，这种方式可以赋予模型不同形式的评估能力。(4)$\text{Weight Merging}$​ 即在直接评估反馈数据集$D_d$上训练得到的模型$\theta_d$​和在成对排序数据集$D_p$​上训练得到的模型$\theta_p$进行权重融合。

![image-20241107114658130](E:\Study\知识构建\NLP系列\assets\image-20241107114658130.png)	实验结果[源自[8]](https://arxiv.org/abs/2405.01535)表明通过权重融合的方式得到的与$\text{GPT4}$皮尔逊相关系数最高，取得了最好的效果。$\text{Prometheus}$系列的工作验证了打造本地的大语言模型用于模型评估的可行性，通过构造特定的反馈数据集可以让开源模型具备细粒度的个性化评估能力，通过权重融合的方式可以让大语言模型更好地适应各种类型的评估格式，虽然最终的评估相关系数表明大部分情况下和$\text{GPT4}$不到$80\%$，但也说明利用开源模型打造能和$\text{GPT4}$​​对齐的个性化评估模型仍有较大的进步空间。


# 3. 心理领域模型测评

​	衡量大语言模型垂直领域能力一直是一项挑战，自大语言模型出现后有许多研究者投身于如何让大语言模型解决心理领域的任务如情感识别、心理咨询等。而随着社会资源竞争加剧，网络舆论传播等多因素综合影响，心理问题逐渐得到重视，心理咨询是一个非常有意义和重要的场景，AI技术的发展让LLM结合心理健康交叉领域的应用逐渐成为研究热点，一些研究者尝试利用LLM进行心理咨询和心理健康支持，而如何衡量LLM在垂直领域表现的好坏是至关重要的，现有的大语言模型评估测试集大多由一系列的封闭式的单项或者多项选择题，还有开放式的问答任务构成。当衡量心理咨询领域的大语言模型时，不仅仅应该考虑对话上下文的流畅度，更重要的是大语言模型在对话中是否能灵活运用心理咨询技术建立和患者的关系，给患者提供实质性建议，帮助患者解决问题。
​	$\text{Counseling Bench}$ 提供了一个从七个方面评估咨询有效性的框架:

![CounsellingBench](assets\CounsellingBench.png)

​	为了评估LLM在垂域的表现，作者采用$\text{GPT4}$通过$\text{few-shot in-context learning}$​的方式进行自动化评估，并在提示词中指明了四点需要考虑的因素：(1) 回复是否和建议相符。(2)回复是否使用了指定的策略。(3) 回复的风格是否和人类咨询师相符合。(4)消除长回复带来的偏差。

​	作者在论文中用一幅雷达图表明了$\text{GPT4}$评估各个模型的能力，但是论文中并未详细说明这个评估方式的细节，只是大致提及了一个成对比较的方式用于衡量不同模型回复的好坏，如果雷达图中的结果是成对比较的结果，那么应当有一个基准模型(baseline)，论文中并未详细阐述，因此也有可能是$\text{GPT4}$​​进行单点打分的结果。这篇论文在模型评估方法上的篇幅与其他模型评估工作比之较少，或者说寥寥无几，但笔者认为其为提供了一种多维度指标评估大语言模型能力的方法值得参考，并进一步探索和完善，比如可以结合更丰富、个性化的维度指标评估垂直领域模型能力。

​	Wang等人[x]通过提示工程技术结合GPT4模型构造了一款侧重认知重构的心理领域咨询助手CRBot，通过分析真实用户的聊天数据和结合心理专业健康人员的评审，较为全面地评估了GPT4在心理领域的认知重构能力，实验方以填写心理测评量表（PHQ-9,GAD-7）的方式筛选了一批符合要求的参与者，并通过ZOOM的方式让对话助手和参与者进行多次聊天，同时要求参与者主要注意如下几个关键方面:（1）对话助手CRBot与实际咨询认知的一致性，（2）如何通过认知重构引导参与者，（3）对话中出现的一些道德与安全问题。

​	**Protocol-Adherent,Natural,and In-Depth Conversations.** 考察CRBot是否可以持久地遵循设定协议、进行自然有深度的对话。心理专家发现CRBot可以在没有显示地告知治疗步骤的情况下无缝地融合认知重构，即智能助手不会直接说“现在让我们来重构一下你的认知”类似的表达，这样可以让用户没那么明显地感受到自己在接受治疗，从而促使那些犹豫的人进行参与。

​	**Misuse of Positive Regard.**考察CRBot是否会滥用正向关怀。心理专家们发现CRBot在给予共情和正常化方面做的非常好，并且有多样的表达方式，但是某些情况下CRBot的回答可能显得夸张并与用户实际痛苦脱节，这种现象在心理学文献中通常成为“Toxic Positivity”，如在一段对话中，用户提供了一个极为简单、甚至称不上充分的例子来挑战自己的扭曲想法，而 CRBot 仍然回应道：“这是一个很棒的例子！”，又或者是用户描述了自己的负面经历，而 CRBot 却用过度乐观的语气回应，使用户的痛苦被掩盖。心理专家指出这样的互动可能会在无意间淡化用户的真实痛苦或挫败感，反映了心理治疗中的普遍忧虑：善意的安慰有时可能会使真正的痛苦被轻视或忽略。

​	**Power Dynamics.** CRBot对话时会产生或者强化权力不对等，这种现象主要体现在引导性问题、评判性或“绝对化”的赞美，以及直接给出建议。心理专家指出，虽然某些形式的引导可以帮助来访者识别和重构消极思维，但引导性问题可能在无意中削弱用户的自主性。此外，CRBot在问题结尾使用“对吧？”这种表达，心理专家2则表明其会避免以“对吧”结尾的问题，因为焦虑程度较高的人往往会倾向于迎合对方意见，不管他们是否真正相信这个观点，他们都会顺从或是取悦提问者。

​	**Subjectivity&Context understanding.** CRBot往往会简化或误解用户的经历，在心理咨询中，准确捕捉并反映来访者的情绪状态对于建立信任关系和促进治疗联盟至关重要 。通过镜像来访者的语言，治疗师能够表达理解、验证来访者的观点，并引导他们进一步思考。心理专家发现 CRBot 有时会无意间歪曲用户表达的体验，从而可能破坏这一反思过程。如一位用户描述自己感到“尴尬”，CRBot将这个经历总结为“艰难”，另一位用户表示因工作进度不足感到“压力”但是CRBot直接假设其压力仅源于工作进度的不满意。这类误解揭示了 LLM 在心理治疗中的固有局限性：尽管LLM擅长复述和总结，但缺乏深入探究或澄清歧义的能力，因此难以真正理解用户的个体情境。一次误解就可能无意间改变整个治疗过程的方向。

​	Wang等人的研究并未系统地规划CBT测评维度及制定具体的量化指标，但从该研究中我们可以了解到一些LLM用于CBT的认知重评的局限性，如会产生权力不对等——过度引导、过度积极——滥用正向关怀、简化或误解用户经历——难以真正理解用户面临的情形等，我们可以借鉴研究中专家所列出的典型例子，依次制定评价指标或是注意事项，在模型测评时根据评价指标或事项评分。

​	Chiu[[x]]()等人模型测评框架，xxxx

​	**PsyDT**[[x]]()提出了一个能生成个性化风格的心理咨询多轮对话的框架，并基于传统的模型评估指标（ROUGE,BLEU）和人类专家评分对微调后的模型进行了系统评估。

​	**CPsyCoun**[[x]]()提出了一个构造高质量对话数据集的两阶段方法，并提供了一个自动化评估多轮心理咨询的基准测试。在多轮对话测评的指标上，CPsyCoun提出从Comprehensiveness、Professionalism、Authenticity、Safety四个角度进行自动化评估。对于多轮对话的评估，CPsyCoun将其拆分为多个单轮对话，每次将当前轮次$i$的$query_i$和对应的历史$h_i$作为评估模型的输入，得到当前轮次的回答$\hat r_i$：
$$
\hat{r}_{i}=\left\{\begin{array}{rr}
f_{L L M}\left(q_{i}\right), & i=1 \\
f_{L L M}\left(h_{i}, q_{i}\right), & 1<i \leq m
\end{array}\right.
$$
​	其中$h_i=\set{(q_j,r_j)|j=1,2,...,i-1}$，代表前$i-1$轮的对话历史，接着让GPT-4o作为Judge根据给定的Prompt为每一轮的回答分配一个分数$\hat s_i$。最后将每一轮评估的结果进行平均作为最终的分数：
$$
s_i=\frac{1}{m}\sum_{i=1}^{m}\hat s_i
$$
​	具体的评估指标如图所示：

![image-20250414212630298](assets\image-20250414212630298.png)

​	作者采用了内在评估（Intrinsic Evaluation）和外在评估两种方式（Extrinsic Evaluation）。内在评估方面，作者用提示工程指示GPT-4评估Role-Play prompt和Memo2Demo prompt的提示词方法生成的多轮对话语料，结果标明后者生成的对话质量在理解能力、专业程度和权威度方面有显著提升。而外在评估就是指评估这种生成方式下生成的语料是否能有效提升模型在心理方面的咨询能力，作者基于这些语料微调出了几个模型并采用自动化评估的方式考察。最终结果如下图：

![image-20250428104831437](assets\image-20250428104831437.png)

​	外在评估的实验结果标明，在该数据集上微调后的模型不仅能让模型自然地学习心理专业的咨询技巧还能让模型学会真实场景下的心理咨询的对话风格。

​	**CBT-Bench**[[x]]()是首个测评LLM在认知行为疗法方面能力的心理领域基准测试数据集，其涵盖了三个类型的任务：（1）CBT-QA，用于测评基础CBT知识，由220道多项选择题构成，这些问答对来自于研究生考试，涵盖了广发的CBT知识，包括基础知识，实践知识和案例学习。（2）CBT-CD(由CBT-PC(primary core belief classification)和CBT-FC(fine-grained core belief classification)构成)是评估模型认知理解能力的数据集，由两个不同粒度的分类数据集构成。（3）CBT-DP(Deliberate Practice)则是用于评估模型心理咨询回应能力的数据集，是传统研究生CBT练习时的题目，一共分为了三个难度级别共计156道题。

​	在前两个任务中，使用准确率可以精确衡量模型在CBT知识方面的掌握能力。而在第三类生成类任务上，CNT-Bench采用专家制定的标准作为衡量依据，让人类专家参考标准进行成对比较打分，模型回复和参考回复会随机以A/B的身份出现，标注时具体细节如下：专家在每个指标上和整体情况上都会标注如下五个选择中的一个作为比较结果：（1）A比B好很多（2）A比B好一点（3）A和B一样好（4）B比A好一点（5）B比A好很多。最后再给整体情况一个标注结果。依据标注结果计算各个模型与参考答案的Win Rate。为了保证对比的公平性，通过提示工程让各模型输出内容长度近似。

![image-20250410195234135](assets\image-20250410195234135.png)

![image-20250410195125431](assets\image-20250410195125431.png)

​	在第三类任务上论文中并未采用模型自动化评估的方式，依赖于人工专家虽然能较大程度保证结果可靠，但随着数据规模地增大，采用人类评估将会变得耗时与昂贵，因此，一套行之有效的CBT领域的模型测评方法亟待探索。

# 4.提升评估可靠性&减少Bias

​	

## 4.1 提升评估可靠性



## 4.2 减少Bias

### 4.2.1 减少Position Bias



### 4.2.2 减少Verbosity Bias









# 5.总结与思考





# 6.**参考文献：**

[[1]Using LLMs for Evaluation](https://cameronrwolfe.substack.com/p/llm-as-a-judge)

[[2]Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664)

[[3]Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/pdf/2306.05685)

[[4]AlpacaEval : An Automatic Evaluator for Instruction-following Language Models](https://github.com/tatsu-lab/alpaca_eval/tree/main)

[[5]Length-Controlled AlpacaEval:A Simple Way to Debias Automatic Evaluators]()

[[6]AlignBench: Benchmarking Chinese Alignment of Large Language Models](https://arxiv.org/pdf/2311.18743)

[[7]PROMETHEUS: INDUCING FINE-GRAINED EVALUATION CAPABILITY IN LANGUAGE MODELS](https://arxiv.org/pdf/2310.08491)

[[8]PROMETHEUS2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535)

[[9]ChatCounselor: A Large Language Models for Mental Health Support](https://arxiv.org/abs/2309.15461)

[[10]**Evaluating an LLM-Powered Chatbot for Cognitive Restructuring: Insights from Mental Health Professionals**](https://arxiv.org/abs/2501.15599#:~:text=In%20this%20work%2C%20we%20evaluate%20an%20LLM-powered%20chatbot%2C,to%20deliver%20cognitive%20restructuring%20%28CR%29%2C%20with%2019%20users.)

[[x]**CBT-BENCH: Evaluating Large Language Models on Assisting Cognitive Behavior Therapy**]()

[[x]**CPsyCoun: A Report-based Multi-turn Dialogue Reconstruction and Evaluation Framework for Chinese Psychological Counseling**]()

[[x]**PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling**]()

[[x]A Survey on LLM-as-a-Judge ]()



