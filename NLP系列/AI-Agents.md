

# $\text{LLM-Agents}$

​		随着大语言模型的兴起，AI AgentAI Agent这一词也随之变得火热，实际上AI AgentAI Agent已经经历了几个阶段的演变。我们可以先进行一下简短的回顾：

​		$\text{Symbolic Agents:}$在人工智能研究早期阶段，Symbolic AISymbolic AI占据了主导地位，其特点是依赖于符号性的逻辑。早期的AI AgentsAI Agents主要关注两类问题： 1.transduction problem 2. respresentation/reasoning problem。这些Agents旨在模拟人类思考的方式，并且用于解决问题有明确的推理框架和可解释性。一个经典的例子就是专家系统，但是这类AgentsAgents的缺陷也很明显， 符号主义在处理不确定性和大量现实世界问题时仍有局限性，此外，符号推理赛算法复杂性高，想要平衡其时间效率和性能具有十足挑战性。

  $\text{Reinforcement learning-based agents}$在强化学习（RL）的早期阶段，智能体主要依靠一些基础技术来进行学习，比如策略搜索和价值函数优化。其中，Q-learning和SARSA是比较著名的方法。 但是，随着深度学习技术的快速发展，我们将深度神经网络与强化学习相结合，形成了一种新的学习范式——深度强化学习（DRL）。这种结合让智能体能够从海量的高维数据中学习到复杂的策略，也带来了许多突破性的成果，比如AlphaGo和DQN。深度强化学习的强大之处在于，它让智能体能够在未知的环境中自主地进行学习，而不需要人类的干预或提供明确的指导。这种自主学习的能力，是AI领域的一大飞跃，也是未来智能体能够更好地适应复杂多变环境的关键。

  $\text{LLM-based agents:}$随着大型语言模型（LLM）展现出令人瞩目的涌现能力，研究者们开始用其来打造新一代的人工智能代理。 他们将这些语言模型视为智能体的核心或“大脑”，并通过整合多模态感知和工具使用等策略，大大扩展了智能体的感知和行动能力。基于LLM的代理能够利用“思维链”（Chain of Thought，CoT）和问题分解（task decomposition）等技术，展现出与传统的符号智能体相媲美的推理和规划能力。 此外，它们还能够通过与环境的互动，从反馈中学习并执行新的动作。现如今，基于LLM的智能体已经被用于各种现实世界场景，比如用于软件开发与科学研究。 [[1](https://arxiv.org/abs/2309.07864v3)]

### 智能体系统概述

  在由LLM驱动的智能体系统中，三个关键的不同组件使得LLM能充当智能体的大脑，从而驱动智能体完成不同的目标。

- Reasoning & Planning通过推理，智能体可以将任务细分成更简单、可执行程度更高的子任务。就像人类解决问题一样，基于一些证据和逻辑进行推理，因此，对于智能体来说推理能力 对于解决复杂任务至关重要。规划是人类应对复杂挑战时的核心策略。它帮助人类组织思维、确立目标，并制定实现这些目标的途径。对于智能体而言，规划能力同样关键，而这一能力取决于推理， 通过推理，智能体能够将复杂的任务分解为更易管理的子任务，并为每个子任务制定恰当的计划。随着任务的推进，智能体还能通过自省（反思）来调整其计划，确保它们与真实世界的动态环境保持一致，从而实现自适应性和任务的成功执行。 总的来说，推理和规划可以将复杂任务拆分成更易解决的子任务，同时通过反省之前的推理步骤，从错误中学习，并为未来的行动精炼策略，以提高最终结果的质量。

- Memory.“记忆”存储智能体过去的观察、想法和行动的序列。正如人脑依赖记忆系统来回顾性地利用先前的经验来制定策略和决策一样，智能体需要特定的内存机制来确保它们熟练地处理一系列连续任务。 当人类面对复杂的问题时，记忆机制有助于人们有效地重新审视和应用先前的策略。此外，记忆机制使人类能够通过借鉴过去的经验来适应陌生的环境。此外，记忆可以分为短期记忆和长期记忆，对于基于LLM的智能体而言，In-context learning的内容可以视作LLM的短期记忆， 而长期记忆则是指给LLM提供一个向量知识库，智能体可以通过检索知识库获取其内部信息。

- Tool Use.当人们感知周围环境时，大脑会整合信息，进行推理和决策。人们通过神经系统控制身体，以适应或创造性行动，如聊天、避开障碍物或生火。 如果智能体拥有类似大脑的结构，具备知识、记忆、推理、规划和概括能力，以及多模式感知能力，那么它也被期望能够以各种方式对周围环境做出反应。而基于LLM的智能体的动作模块负责接收来自大脑模块的动作指令，并执行与环境互动的动作。 LLM收到指令再输出文本是其固有的能力，因此我们后继主要讨论其工具使用能力，也就是所谓的Tool Use。

![agent_framwork](src\AI-Agents\agent_framwork.jpg)

## 1.Reasoning & Planning(推理和规划)

### 1.1Reasoning

  Chain of Thought.思维链技术逐渐成为大语言模型解决复杂类任务的标准方法，其通过在提示中加入几个具体的推理步骤提升大语言模型解决问题的性能，此外，有很多CoT的变种，如 Zero Shot CoT，其通过在提示中插入"think step by step"这样的一句话引导模型思考，推理出最终答案。

  Tree of Thoughts.ToT通过在每个步骤探索多种推理可能性来扩展CoT。它首先将问题分解为多个思考步骤，并在每个步骤中生成多个想法，从而创建一个树状结构。 然后基于树状结构进行搜索寻求最优结果，搜索过程可以是BFS（广度优先搜索）或DFS（深度优先搜索），每个状态由分类器（通过提示）或多数投票评估。

### 1.2Planning

  Least-to-Most是提出问题后，先将其分割成若干小问题，然后一一解决这些小问题的一种策略。这种策略受到真实教育场景中用于指导儿童的策略的启发。 与CoT Prompting类似，这个策略首先将需要解决的问题分解成一系列子问题。子问题之间存在逻辑联系和渐进关系。在第二步，逐一解决这些子问题。 与CoT Prompting的最大差异在于，在解决下一个子问题时，会将前面子问题的解决方案作为提示输入。一个具体的字母连接的例子如下：

```markdown
Q: think, machine
A: The last letter of "think" is "k". The last letter of "machine" is "e". 
Concatenating "k" and "e" gives "ke". So "think, machine" output "ke".

Q: think, machine, learning
A: "think, machine" outputs "ke". The last letter of "learning" is "g". 
Concatenating "ke" and "g" gives "keg". So "think, machine, learning" is "keg".

Q: transformer, language
A: The last letter of "transformer" is "r". The last letter of "language" is "e". 
Concatenating "r" and "e" gives "re". So "transformer, language" is "re".

Q: transformer, language, vision
A:
```

![least2most](src\AI-Agents\least2most.png)

​	ReAct通过将动作空间扩展为特定任务的离散动作和语言空间组合，成功将推理和行动集成在LLM中。前者能够使LLM生成自然语言进行推理，分解任务，后者则可以使LLM能够与环境交互（例如使用搜索API）。 ReAct prompt提示模板包含让LLM思考的明确步骤，从Langchain的源码中，我们可以找到如下：

~~~markdown
The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
The only values that should be in the "action" field are: {tool_names}
The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
```
{{{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}}}
```
ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
~~~

![ReAct](src\AI-Agents\ReAct.png)

​	Reflexion是一个为智能体提供动态记忆和自我反思能力从而提升智能体推理能力的框架。如下图所示，Reflexion框架中有三个角色，分别是Actor,Evaluator,Self-Reflection， 其中Actor模型基于观察到的环境和生成文本和动作，动作空间遵循ReAct中的设置，在ReActReAct中，特定任务的动作空间用语言扩展，以实现复杂的推理步骤。然后生成的这些文本和动作轨迹由Evaluator进行评估，比如用0,1来表示好坏，接着Self-Reflection会生成特定的文本反馈，提供更丰富、有效的反思并被存储到记忆中。最后，Actor会根据得到的记忆生成新的文本和动作直到完成任务。

![Reflexion](src\AI-Agents\Reflexion.png)

 	论文中提到，在序列决策任务ALFWorld上，为了实现自动化评估采用了两个技术：1.用LLM进行二分类 2.人工写的启发式规则检测错误。对于后者，简单来说就是 如果智能化执行相同操作并收到相同响应超过3个周期，或者当前环境中的操作次数超过30次(计划不高效)，就进行自我反思。

  Chain of Hindsight(CoH).是一种引入了人类偏好的训练方法，该方法不仅使用正面反馈数据，还利用了负面反馈数据，此外在模型预测时引入了反馈条件 ，使模型可以根据反馈学习并生成更符合人类偏好的内容。但CoT利用了序列形式的反馈，在训练时给模型提供更丰富的信息，具体如下：

```markdown
# How to explain a neural network to a 6-year-old kid? Bad:{a subpar answer} Good:{a good answer}.
```

 	这样的数据拼接格式能组合不同种类的反馈，从而提升模型性能。在推理阶段，我们只需要在prompt中给模型指定好Good就能引导模型生成高质量的结果。 此外，在训练时并不是所有的token都纳入损失函数计算，feedback token即Good or BadGood or Bad只用来提示模型接下来预测时生成质量好的内容还是差的内容，具体损失函数公式如下：

​												          				$\begin{aligned} \log p(\mathbf{x})=\log \prod_{i=1}^{n} \mathbb{1}_{O(x)}\left(x_{i}\right) p\left(x_{i} \mid\left[x_{j}\right]_{j=0}^{i-1}\right) \end{aligned}$

​		其中$\mathbb{1}_{O(x)}$是指示函数，如果$x_i$属于$\text{feedback token(Good or Bad)}$，则取值为$0$，反之为$1$。人类反馈数据，并不止单纯是先前所表述的只有$\text{Good or Bad}$，它可以是更广义的形式，我们可以记作$D_h=\{q,a_i,r_i,z_i\}^{n}_{i=1}$，$q,a_i$分别是问题和答案，$r_i$是答案的评级高低，$z_i$则是人类提供的事后反馈。假设反馈的评级由低到高排的顺序是$r_{n} \geq r_{n-1} \geq \cdots \geq r_{1}$,那么在微调时数据拼接的形式如下：$d_h=(q,z_i,a_i,z_j,a_j,\cdots,z_n,a_n),(i\leq j\leq n)$，训练时模型只会根据给定的前缀预测$a_n$，使得模型能够自我反映以基于反馈序列产生更好的输出。（笔者用小样本``900条数据``在7B级别的LLM上尝试微调时，发现如果反馈评级和回答$z_i,a_i,z_j,a_j,...,z_n,a_n$按照顺序排列，那么模型在推理时没法较好地适配不同反馈前缀，即生成的内容与反馈指定的预期的内容不符。比如$\text{feedback token}$分别是“非提问式共情”和“提问式共情”，数据拼接格式为$q,\underbrace{非提问式共情}_{\text{feedback token}}a_1,\underbrace{提问式共情}_{\text{feedback token}}a_2$。模型在预测时即便给定了不同的$\text{feedback token}$，其对与提问式共情的感知能力仍然不足，一直会进行非提问式共情。）

```markdown
# 我最近心情很糟糕 非提问式共情:{听起来你真的很难受呢} 提问式共情:{我真是替你感到难过，请问你现在赶紧怎样呢？想和我聊一聊吗？}.
```

​		在训练中，由于模型在预测时条件化了另一个模型输出及其反馈，因此可能会简单地“复制”已提供的示例，而不是真正理解任务。 为了防止模型只“复制”示例而不学习任务，作者在训练时随机掩盖了0%到5%的过去标记。这意味着模型无法直接看到这些标记，从而需要真正地理解上文，而不是简单地复制，这种随机掩盖增加了模型的泛化能力。

![CoT](src\AI-Agents\CoT.png)

​		该文对RLHF和CoT方法进行了实验对比，图中的蓝色柱是标准指令上的得分，从整体上看，RLHF在标准指令上最终的效果还是要比CoT要好的，但是当条件词改变，让模型输出更好的质量的内容时，CoT生成的 结果质量要比RLHF要好，而且对于不同条件词的变化更为敏感，说明模型较好地理解了人类偏好。



## 2.Memory(记忆)

​		在神经科学领域，人类的记忆被分为几种不同的类型，这些类型在信息处理和存储方面各有特点，具体如下：

 	1.感觉记忆（Sensory Memory）这是记忆系统的第一阶段，非常短暂，它保存了来自感官的信息，如视觉和听觉，通常只持续几秒钟。例如，当你看到一串数字时，你能在短时间内记住它们，这就像计算机中的缓存，用于临时存储。

 	2.短期记忆（Shot-term Memory）短期记忆是你在短时间内能够主动保持和操纵的信息。它有有限的容量，通常可以持续大约20-30秒，通过重复或其他策略可以转移到长期记忆。这就像计算机的RAM，可以快速读写，但断电后信息会丢失。

​		3.长期记忆（Long-term Memory）长期记忆是信息可以存储很长时间的地方，从几个小时到一生。它分为两种主要的子类型：

  - 显性记忆（Explicit Memory）这是有意识的记忆，可以进一步分为事实和信息（语义记忆）以及个人经历（情景记忆）。就像在硬盘上存储的文件，你可以有意识地检索这些信息。

  - 隐性记忆（Implicit Memory）这是无意识的记忆，包括技能、习惯和条件反射（骑单车、敲键盘）。这些记忆不像显性记忆那样容易被意识到，但它们影响你的行为和反应。这就像计算机的BIOS或操作系统设置，你通常不直接与它们交互，但它们影响计算机如何运行。

  我们可以将智能体和记忆按照如下映射进行理解：

- 感觉记忆就像原始输入对应的语义嵌入，包括文本、图像和其他模态。

- 短期记忆就像上下文学习(In-context learning)，是有限的，受到LLM上下文窗口限制。

- 长期记忆就像外部向量库，智能体可以通过快速检索从向量库中抽取相关的信息进行阅读理解。

  外部记忆可以有效地缓解智能体幻觉问题，并能增强智能体在特定领域的性能。通常将相关非结构化知识由模型转化成语义嵌入并存储进向量数据库中，为了优化 检索速度，一般选择ANN算法返回



## 3.Tool Use(工具使用)
