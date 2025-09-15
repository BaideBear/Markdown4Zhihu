# ai infra基础笔记

## 目录

- [ai infra基础笔记](#ai-infra基础笔记)
  - [目录](#目录)
  - [数据并行](#数据并行)
    - [梯度异步更新](#梯度异步更新)
    - [分布式数据并行（DDP）](#分布式数据并行ddp)
  - [流水线并行](#流水线并行)
  - [张量并行](#张量并行)
    - [切分方式](#切分方式)
    - [MLP切分](#mlp切分)
    - [Muti-Head Attention计算切分](#muti-head-attention计算切分)
    - [Embedding切分](#embedding切分)
    - [Cross-entropy层](#cross-entropy层)
  - [TP+DP](#tpdp)
  - [FlashAttention v1](#flashattention-v1)
    - [计算限制与内存限制](#计算限制与内存限制)
    - [GPU结构，以A100为例](#gpu结构以a100为例)
    - [forward过程](#forward过程)
    - [backward过程](#backward过程)
    - [源码学习（triton实现）](#源码学习triton实现)
  - [FlashAttention v2](#flashattention-v2)
    - [forward过程](#forward过程-1)
    - [backward过程](#backward过程-1)
    - [Thread Blocks优化](#thread-blocks优化)
    - [Wrap层并行](#wrap层并行)
  - [FlashAttention v3](#flashattention-v3)
    - [Hopper Architecture GPU](#hopper-architecture-gpu)
    - [算法1：数据搬运、计算异步（Warp-specialization）](#算法1数据搬运计算异步warp-specialization)
    - [算法2：Pingpong scheduling](#算法2pingpong-scheduling)
    - [算法3：Intra-warpgroup overlapping](#算法3intra-warpgroup-overlapping)
    - [算法4：3-stage pipelining](#算法43-stage-pipelining)
  - [Megatron-LM 代码学习](#megatron-lm-代码学习)
    - [initialize\_megatron](#initialize_megatron)
    - [get\_model, 模型划分](#get_model-模型划分)
    - [CodeGeeXModel，模型类的实现](#codegeexmodel模型类的实现)

## 数据并行

思路：每一个GPU中保存一份参数，在计算完梯度后将所有的梯度汇总，更新参数（all-reduce）

传统思路：参数服务器，分为Worker（计算节点）和Server（负责统计梯度以及下发更新的参数）；

### 梯度异步更新

在Server汇总梯度，搬运数据的时候，Worker节点出于空转状态，可以通过梯度异步更新来解决：

![alt text](https://img.remit.ee/main/image-3.png)

在Worker计算完一轮梯度之后，不会等待聚合梯度返回，而是直接进行写一轮的计算；

这个操作等价于增加了每个batch的大小；

延迟步数：最多"跳过"几次参数更新操作：

![alt text](https://img.remit.ee/main/image-4.png)

- Sequential: 无延迟
- Eventual: 不指定延迟步数，“随便”，什么时候更新了什么时候用新参数
- 1 Bounded delay: 延迟步数为1

### 分布式数据并行（DDP）

使用Ring-ALlReduce来取代Server节点的作用；

在Ring-ALlReduce中，所有GPU的拓扑是一个环形，只能和相邻的GPU进行通信；Ring-AllReduce不会减少通信量，但是能将通信平摊到每个Worker节点上，以提升通信速度；

下面是在4张GPU的情况下看一个例子：

- **Step 1： Reduce-Scatter**

假设为数据大小为N,卡的数量为M,那么经过Reduce-Scatter操作后，每个卡保存N/M大小的Reduce结果：

![alt text](https://img.remit.ee/main/image-5.png)
![alt text](https://img.remit.ee/main/image-6.png)
![alt text](https://img.remit.ee/main/image-7.png)

- **Step 2： All-Gather**

每张卡将自己的结果部分通过环状传递发送给每一张卡：

![alt text](https://img.remit.ee/main/image-8.png)
![alt text](https://img.remit.ee/main/image-9.png)

以此类推；

## 流水线并行

把模型的不同层放在不同的GPU上，一个batch的执行过程如下：

![alt text](https://img.remit.ee/main/image.png)

这种并行方法会带来一些问题：

1. GPU的利用度不够

我们计算一下所有GPU空转的时间：

假设有 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 个GPU，每一块做前向和后向计算的时间是 <img src="https://www.zhihu.com/equation?tex=t_{fb} = t_f + t_b" alt="t_{fb} = t_f + t_b" class="ee_img tr_noresize" eeimg="1"> ,则：

![alt text](https://img.remit.ee/main/image-1.png)

- 整个位置整体的面积是 <img src="https://www.zhihu.com/equation?tex=K * (K*t_{fb})" alt="K * (K*t_{fb})" class="ee_img tr_noresize" eeimg="1"> , 实际工作的面积是 <img src="https://www.zhihu.com/equation?tex=K * t_{fb}" alt="K * t_{fb}" class="ee_img tr_noresize" eeimg="1"> 
- 无用工作的面积是  <img src="https://www.zhihu.com/equation?tex=(K - 1)*t_{fb}" alt="(K - 1)*t_{fb}" class="ee_img tr_noresize" eeimg="1"> , 无用部分占比即为  <img src="https://www.zhihu.com/equation?tex=(K - 1)*K*t_{fb} / K * K*t_{fb} = (K-1)/K" alt="(K - 1)*K*t_{fb} / K * K*t_{fb} = (K-1)/K" class="ee_img tr_noresize" eeimg="1"> 

所以bubble的复杂度为 <img src="https://www.zhihu.com/equation?tex=O(\frac{K-1}{K})" alt="O(\frac{K-1}{K})" class="ee_img tr_noresize" eeimg="1"> , K越大，也就是GPU越多时，空闲的比例就越接近1；

**解决方案：micro-batch**:

流水线并行收到的batch是mini-batch，我们将mini-batch进一步划分为更小的batch(micro-batch)，再送入模型中进行训练：

![alt text](https://img.remit.ee/main/image-2.png)

假设每个mini-batch被划分为 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 个micro-batch，bubble的时间复杂度是 <img src="https://www.zhihu.com/equation?tex=O(\frac{K-1}{K+M-1})" alt="O(\frac{K-1}{K+M-1})" class="ee_img tr_noresize" eeimg="1"> ; 经过实验(Gpipe)， <img src="https://www.zhihu.com/equation?tex=M >= 4K" alt="M >= 4K" class="ee_img tr_noresize" eeimg="1"> 时，bubble产生的影响可忽略不计；

2. 中间结果占用显存

计算的过程中，每层都要保存中间结果，设每层宽度为d，层数为L，GPU数量K，mini-batch大小N

每张GPU额外占据 <img src="https://www.zhihu.com/equation?tex=O(N*L/M*d)" alt="O(N*L/M*d)" class="ee_img tr_noresize" eeimg="1"> 的空间

**解决方案：重计算（re-materialization）**：

空间换时间，在每块GPU上，我们只保存来自上一层（上一张GPU上）的输出，其他的中间结果我们用完直接抛弃，等micro-batch对应的backward到来时，重新计算forward的结果：

设每个mini-batch被划分为 <img src="https://www.zhihu.com/equation?tex=M" alt="M" class="ee_img tr_noresize" eeimg="1"> 个micro-batch，最终的空间占用为 <img src="https://www.zhihu.com/equation?tex=O(N + N/M * L/K * d)" alt="O(N + N/M * L/K * d)" class="ee_img tr_noresize" eeimg="1"> ;

## 张量并行

### 切分方式

1. 行切分

![alt text](https://img.remit.ee/main/image-10.png)

将数据按列切分，参数按行切分，前向需要一次all-reduce操作，反向不需要通信：

![alt text](https://img.remit.ee/main/image-11.png)

2. 列切分

![alt text](https://img.remit.ee/main/image-13.png)

对参数的按列切分，数据不进行切分，前向不需要通信，反向需要一次all-reduce操作：

![alt text](https://img.remit.ee/main/image-12.png)

### MLP切分

![alt text](https://img.remit.ee/main/image-14.png)

对A按列切分，对B按行切分，整个MLP过程中，前向反向各一次all-reduce操作：

![alt text](https://img.remit.ee/main/image-15.png)

forward：

- f: 把输入X发送到两个GPU上，每个GPU独立做forward计算
- g: 每个GPU结束计算后，将输出 <img src="https://www.zhihu.com/equation?tex=Z_i" alt="Z_i" class="ee_img tr_noresize" eeimg="1"> 聚合，GPU之间进行一次All-Reduce,计算出相加的结果 <img src="https://www.zhihu.com/equation?tex=Z" alt="Z" class="ee_img tr_noresize" eeimg="1"> 

backward:

- g: 将 <img src="https://www.zhihu.com/equation?tex=\frac{\partial L}{\partial Z}" alt="\frac{\partial L}{\partial Z}" class="ee_img tr_noresize" eeimg="1"> 发送到两个GPU上，独立做梯度传播
- f: 将两个GPU的梯度聚合，通过All-Reduce获得梯度相加的结果

### Muti-Head Attention计算切分

对于多头注意力，我们只要按照注意力头进行切分即可（列切分），最后将结果concat；对于后面的线性层，进行行切割；整个过程前后向各一个All-Reduce操作：

![alt text](https://img.remit.ee/main/image-16.png)

### Embedding切分

对输入层的word embedding，进行行切分，最终的结果all-reduce；

对输出层的word embedding，进行列切分，没有all-reduce过程；

### Cross-entropy层

![alt text](https://img.remit.ee/main/image-17.png)

可以进行一个All-Gather操作，然后计算softmax，交叉熵即可，但是这会产生额外的一组通信开销 <img src="https://www.zhihu.com/equation?tex=b*s*v" alt="b*s*v" class="ee_img tr_noresize" eeimg="1"> ；

![alt text](https://img.remit.ee/main/image-18.png)

可以在每个GPU上进行求和，再做All-Reduce操作，得到softmax的分母，分别进行softmax、loss计算，最终再进行一次All-Reduce操作聚合loss结果（设通信量为N）;

最终的通信开销变为 <img src="https://www.zhihu.com/equation?tex=b*s + N" alt="b*s + N" class="ee_img tr_noresize" eeimg="1"> ;

## TP+DP

![alt text](https://img.remit.ee/main/image-19.png)

一般将TP的GPU放在同一个机器中，不同的机器上做DP
的操作，原因如下:

1. TP的通讯量远大于DP

对于每一层Transformer：

张量并行一共进行4次All-Reduce操作（attention计算两次，MLP两次），总的通讯量为 <img src="https://www.zhihu.com/equation?tex=8*b*s*h" alt="8*b*s*h" class="ee_img tr_noresize" eeimg="1"> ;

数据并行我们只需要通过All-Reduce来聚合梯度，只需要一次All-Reduce操作，通讯量为 <img src="https://www.zhihu.com/equation?tex=2*h*h" alt="2*h*h" class="ee_img tr_noresize" eeimg="1"> ;

2. backward传递方式不同

对于DP，本层梯度计算完，我们通过All-Reduce得到梯度结果，将结果传递给下一层；

![alt text](https://img.remit.ee/main/image-20.png)

而对DP，我们只需要发送出去梯度结果做All-Reduce过程，不需要等待，继续backward；也就是说，下一层不依赖上一层的结果；

![alt text](https://img.remit.ee/main/image-21.png)

## FlashAttention v1

### 计算限制与内存限制

>  <img src="https://www.zhihu.com/equation?tex=\pi" alt="\pi" class="ee_img tr_noresize" eeimg="1"> : 硬件算力上限，平台倾尽全力每秒的浮点运算数，单位FLOPS or FLOP/s 

>  <img src="https://www.zhihu.com/equation?tex=\beta" alt="\beta" class="ee_img tr_noresize" eeimg="1"> : 硬件带宽极限，平台倾尽全力每秒的内存交换量，单位Bytes/s 

>  <img src="https://www.zhihu.com/equation?tex=\pi_t" alt="\pi_t" class="ee_img tr_noresize" eeimg="1"> : 某个算法所需的运算总量，t代表total，单位FLOPS

>  <img src="https://www.zhihu.com/equation?tex=\beta_t" alt="\beta_t" class="ee_img tr_noresize" eeimg="1"> : 某个算法所需的总数据读取储存量，单位Bytes

>  <img src="https://www.zhihu.com/equation?tex=T_{cal} = \pi_t / \pi, T_{load} = \beta_t / \beta, T = max(T_{cal}, T_{load})" alt="T_{cal} = \pi_t / \pi, T_{load} = \beta_t / \beta, T = max(T_{cal}, T_{load})" class="ee_img tr_noresize" eeimg="1"> 

当 <img src="https://www.zhihu.com/equation?tex=T_{cal} > T_{load}" alt="T_{cal} > T_{load}" class="ee_img tr_noresize" eeimg="1"> 时，计算瓶颈，此时 <img src="https://www.zhihu.com/equation?tex=\pi_t / \beta_t > \pi / \beta" alt="\pi_t / \beta_t > \pi / \beta" class="ee_img tr_noresize" eeimg="1"> ;

当 <img src="https://www.zhihu.com/equation?tex=T_{cal} < T_{load}" alt="T_{cal} < T_{load}" class="ee_img tr_noresize" eeimg="1"> 时，内存瓶颈，此时 <img src="https://www.zhihu.com/equation?tex=\pi_t / \beta_t < \pi / \beta" alt="\pi_t / \beta_t < \pi / \beta" class="ee_img tr_noresize" eeimg="1"> ;

我们定义 <img src="https://www.zhihu.com/equation?tex=\pi / \beta" alt="\pi / \beta" class="ee_img tr_noresize" eeimg="1">  为**计算强度（Operational Intensity）**

以A100 40GB为例， <img src="https://www.zhihu.com/equation?tex=\pi_t / \beta_t = (2N^2d) / (2Nd + 2Nd + 2N^2) = (N^2d) / (2Nd + N^2)" alt="\pi_t / \beta_t = (2N^2d) / (2Nd + 2Nd + 2N^2) = (N^2d) / (2Nd + N^2)" class="ee_img tr_noresize" eeimg="1"> ,  <img src="https://www.zhihu.com/equation?tex=\pi / \beta = 201" alt="\pi / \beta = 201" class="ee_img tr_noresize" eeimg="1"> :

![alt text](https://img.remit.ee/main/image-22.png)

从表中可以看到，memory-bound的情况还是普遍存在的；

> roof-line模型：横坐标为计算强度，纵坐标为硬件性能，刻画了给定一个算法，硬件的理论速度上限

![alt text](https://img.remit.ee/main/image-23.png)

### GPU结构，以A100为例

![alt text](https://img.remit.ee/main/image-24.png)

> SRAM: L1缓存，19TB/s(20 MB)

> GPU HBM: 常说的显存，1.5TB/s(40GB)

> Main Memory: CPU DRAM, 12.8GB/s(>1TB)

> SM: Streaming Multiprocessors，流式多处理器

> SMP: SM Partition, SM由SMP组成

计算流程：将数据从显存（HBM）加载至SRAM中，然后由SM读取并进行计算。计算结果再通过SRAM返回给显存。

### forward过程

符号规定：（Attention计算、safe softmax）
![alt text](https://img.remit.ee/main/image-26.png)
![alt text](https://img.remit.ee/main/image-27.png)

- **Tiling分块流程**

![alt text](https://img.remit.ee/main/image-25.png)

1. 将 <img src="https://www.zhihu.com/equation?tex=Q" alt="Q" class="ee_img tr_noresize" eeimg="1"> 矩阵横切为 <img src="https://www.zhihu.com/equation?tex=T_r" alt="T_r" class="ee_img tr_noresize" eeimg="1"> 块，每块的长度为 <img src="https://www.zhihu.com/equation?tex=B_r" alt="B_r" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=Q_i" alt="Q_i" class="ee_img tr_noresize" eeimg="1"> 表示切完后的第i块矩阵，维度为 <img src="https://www.zhihu.com/equation?tex=(B_r, d)" alt="(B_r, d)" class="ee_img tr_noresize" eeimg="1"> ;
2. 将 <img src="https://www.zhihu.com/equation?tex=K^T" alt="K^T" class="ee_img tr_noresize" eeimg="1"> 矩阵竖切为 <img src="https://www.zhihu.com/equation?tex=T_c" alt="T_c" class="ee_img tr_noresize" eeimg="1"> 块，每块的长度为 <img src="https://www.zhihu.com/equation?tex=B_c" alt="B_c" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=K_j^T" alt="K_j^T" class="ee_img tr_noresize" eeimg="1"> 的维度为 <img src="https://www.zhihu.com/equation?tex=(d, B_c)" alt="(d, B_c)" class="ee_img tr_noresize" eeimg="1"> ;
3. 将 <img src="https://www.zhihu.com/equation?tex=V" alt="V" class="ee_img tr_noresize" eeimg="1"> 矩阵横切为 <img src="https://www.zhihu.com/equation?tex=T_c" alt="T_c" class="ee_img tr_noresize" eeimg="1"> 块，每块的长度为 <img src="https://www.zhihu.com/equation?tex=B_c" alt="B_c" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=V_j" alt="V_j" class="ee_img tr_noresize" eeimg="1"> 的维度为 <img src="https://www.zhihu.com/equation?tex=(B_c, d)" alt="(B_c, d)" class="ee_img tr_noresize" eeimg="1"> ;
4. 初始Attention分数： <img src="https://www.zhihu.com/equation?tex=S_{ij} = Q_i * K_j^T = (B_r, d) * (d, B_c) = (B_r, B_c)" alt="S_{ij} = Q_i * K_j^T = (B_r, d) * (d, B_c) = (B_r, B_c)" class="ee_img tr_noresize" eeimg="1"> ;
5. 对 <img src="https://www.zhihu.com/equation?tex=S_{ij}" alt="S_{ij}" class="ee_img tr_noresize" eeimg="1"> 做softmax、mask、dropout操作，得到 <img src="https://www.zhihu.com/equation?tex=\tilde{P}_{ij}" alt="\tilde{P}_{ij}" class="ee_img tr_noresize" eeimg="1"> （ <img src="https://www.zhihu.com/equation?tex=\tilde{P}" alt="\tilde{P}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=P" alt="P" class="ee_img tr_noresize" eeimg="1"> 分别表示归一化前和归一化后的结果）
6. 计算output： <img src="https://www.zhihu.com/equation?tex=O_{ij} = \tilde{P}_{ij} * V_j = (B_r, B_c) * (B_c, d) = (B_r, d)" alt="O_{ij} = \tilde{P}_{ij} * V_j = (B_r, B_c) * (B_c, d) = (B_r, d)" class="ee_img tr_noresize" eeimg="1"> ，**这里存在一些细节，后面再看**

```python
# ---------------------
# Tc: K和V的分块数
# Tr: Q的分块数量
# ---------------------
for 1 <= j <= Tc:
    for 1 <= i <= Tr:
        do....
```

![alt text](https://img.remit.ee/main/image-28.png)
![alt text](https://img.remit.ee/main/image-29.png)

- **safe softmax**

假设我们算出了上图中的 <img src="https://www.zhihu.com/equation?tex=S_{00}、S_{01}" alt="S_{00}、S_{01}" class="ee_img tr_noresize" eeimg="1"> ，softmax的计算方法如下：

> 定义 <img src="https://www.zhihu.com/equation?tex=S" alt="S" class="ee_img tr_noresize" eeimg="1"> 中的某一行的向量为 <img src="https://www.zhihu.com/equation?tex=x = [x_1, x_2, ..., x_d]" alt="x = [x_1, x_2, ..., x_d]" class="ee_img tr_noresize" eeimg="1"> ，我们将其分为两块： <img src="https://www.zhihu.com/equation?tex=x = [x^{(1)}, x^{(2)}]" alt="x = [x^{(1)}, x^{(2)}]" class="ee_img tr_noresize" eeimg="1"> ;
> 定义： <img src="https://www.zhihu.com/equation?tex=m(x)、m(x^{(1)})、m(x^{(2)})" alt="m(x)、m(x^{(1)})、m(x^{(2)})" class="ee_img tr_noresize" eeimg="1"> 分别为全局的、分块1的、分块2的最大值；
> 我们有： <img src="https://www.zhihu.com/equation?tex=m(x) = m([x^{(1)}, x^{(2)}]) = max(m(x^{(1)}),m(x^{(2)}))" alt="m(x) = m([x^{(1)}, x^{(2)}]) = max(m(x^{(1)}),m(x^{(2)}))" class="ee_img tr_noresize" eeimg="1"> 

> 定义 <img src="https://www.zhihu.com/equation?tex=f(x) = exp(x - m(x)), (x可为任意分块)" alt="f(x) = exp(x - m(x)), (x可为任意分块)" class="ee_img tr_noresize" eeimg="1"> , 那么 <img src="https://www.zhihu.com/equation?tex=f(x) = [e^{m(x^{(1)}) - m(x)}*f(x^{(1)}), e^{m(x^{(2)}) - m(x)}*f(x^{(2)})]" alt="f(x) = [e^{m(x^{(1)}) - m(x)}*f(x^{(1)}), e^{m(x^{(2)}) - m(x)}*f(x^{(2)})]" class="ee_img tr_noresize" eeimg="1"> 
> 定义 <img src="https://www.zhihu.com/equation?tex=l(x) = rowsum(f(x))" alt="l(x) = rowsum(f(x))" class="ee_img tr_noresize" eeimg="1"> ，那么 <img src="https://www.zhihu.com/equation?tex=l(x) = e^{m(x^{(1)}) - m(x)}*l(x^{(1)}) + e^{m(x^{(2)}) - m(x)}*l(x^{(2)})" alt="l(x) = e^{m(x^{(1)}) - m(x)}*l(x^{(1)}) + e^{m(x^{(2)}) - m(x)}*l(x^{(2)})" class="ee_img tr_noresize" eeimg="1"> 

>最终的softmax结果如下：

<img src="https://www.zhihu.com/equation?tex=softmax(x) = \frac{f(x)}{l(x)} = \frac{[e^{m(x^{(1)}) - m(x)}*f(x^{(1)}), e^{m(x^{(2)}) - m(x)}*f(x^{(2)})]}{e^{m(x^{(1)}) - m(x)}*l(x^{(1)}) + e^{m(x^{(2)}) - m(x)}*l(x^{(2)})}" alt="softmax(x) = \frac{f(x)}{l(x)} = \frac{[e^{m(x^{(1)}) - m(x)}*f(x^{(1)}), e^{m(x^{(2)}) - m(x)}*f(x^{(2)})]}{e^{m(x^{(1)}) - m(x)}*l(x^{(1)}) + e^{m(x^{(2)}) - m(x)}*l(x^{(2)})}" class="ee_img tr_noresize" eeimg="1">

整体的伪代码如下：

![alt text](https://img.remit.ee/main/image-30.png)

一些符号的补充定义：
 <img src="https://www.zhihu.com/equation?tex=S_{ij}" alt="S_{ij}" class="ee_img tr_noresize" eeimg="1"> : QK小块相乘的结果；
 <img src="https://www.zhihu.com/equation?tex=\tilde{m}_{ij}" alt="\tilde{m}_{ij}" class="ee_img tr_noresize" eeimg="1"> : 对当前块 <img src="https://www.zhihu.com/equation?tex=S_{ij}" alt="S_{ij}" class="ee_img tr_noresize" eeimg="1"> 来说，是每行的局部最大值；
 <img src="https://www.zhihu.com/equation?tex=\tilde{P}_{ij}" alt="\tilde{P}_{ij}" class="ee_img tr_noresize" eeimg="1"> : <img src="https://www.zhihu.com/equation?tex=P" alt="P" class="ee_img tr_noresize" eeimg="1"> 矩阵归一化前的结果，相当于 <img src="https://www.zhihu.com/equation?tex=f(x^{(1)})" alt="f(x^{(1)})" class="ee_img tr_noresize" eeimg="1"> ;
 <img src="https://www.zhihu.com/equation?tex=\tilde{l}_{ij}" alt="\tilde{l}_{ij}" class="ee_img tr_noresize" eeimg="1"> :分块rowsum的结果，相当于 <img src="https://www.zhihu.com/equation?tex=l(x^{(1)})" alt="l(x^{(1)})" class="ee_img tr_noresize" eeimg="1"> ;
 <img src="https://www.zhihu.com/equation?tex=m" alt="m" class="ee_img tr_noresize" eeimg="1"> :每行的全局最大值，相当于 <img src="https://www.zhihu.com/equation?tex=m(x)" alt="m(x)" class="ee_img tr_noresize" eeimg="1"> ；
 <img src="https://www.zhihu.com/equation?tex=l" alt="l" class="ee_img tr_noresize" eeimg="1"> :每行的全局rowsum，相当与 <img src="https://www.zhihu.com/equation?tex=l(x)" alt="l(x)" class="ee_img tr_noresize" eeimg="1"> ；
 <img src="https://www.zhihu.com/equation?tex=m_i" alt="m_i" class="ee_img tr_noresize" eeimg="1"> : 表示 <img src="https://www.zhihu.com/equation?tex=max(\tilde{m}_{i0}, \tilde{m}_{i1}, ... ,\tilde{m}_{i(j-1)})" alt="max(\tilde{m}_{i0}, \tilde{m}_{i1}, ... ,\tilde{m}_{i(j-1)})" class="ee_img tr_noresize" eeimg="1"> ,如果当前分块是 <img src="https://www.zhihu.com/equation?tex=S_{ij}" alt="S_{ij}" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=m_i" alt="m_i" class="ee_img tr_noresize" eeimg="1"> 是前 <img src="https://www.zhihu.com/equation?tex=j-1" alt="j-1" class="ee_img tr_noresize" eeimg="1"> 个分块的局部最大值；
 <img src="https://www.zhihu.com/equation?tex=m_i^{new}" alt="m_i^{new}" class="ee_img tr_noresize" eeimg="1"> :相对于 <img src="https://www.zhihu.com/equation?tex=m_i" alt="m_i" class="ee_img tr_noresize" eeimg="1"> ，加入了当前块 <img src="https://www.zhihu.com/equation?tex=S_{ij}" alt="S_{ij}" class="ee_img tr_noresize" eeimg="1"> 的更新；
 <img src="https://www.zhihu.com/equation?tex=l_i" alt="l_i" class="ee_img tr_noresize" eeimg="1"> ：与 <img src="https://www.zhihu.com/equation?tex=m_i" alt="m_i" class="ee_img tr_noresize" eeimg="1"> 同理；
 <img src="https://www.zhihu.com/equation?tex=l_i^{new}" alt="l_i^{new}" class="ee_img tr_noresize" eeimg="1"> : 与 <img src="https://www.zhihu.com/equation?tex=m_i^{new}" alt="m_i^{new}" class="ee_img tr_noresize" eeimg="1"> 同理；

![alt text](https://img.remit.ee/main/image-31.png)

**O的更新**

被圈出的部分的S、P部分的乘积即为最终的 <img src="https://www.zhihu.com/equation?tex=O_i" alt="O_i" class="ee_img tr_noresize" eeimg="1"> :

![alt text](https://img.remit.ee/main/image-32.png)

我们的思路是： <img src="https://www.zhihu.com/equation?tex=O_i = O_i + 最新的结果" alt="O_i = O_i + 最新的结果" class="ee_img tr_noresize" eeimg="1"> ，通过迭代的方式计算 <img src="https://www.zhihu.com/equation?tex=O" alt="O" class="ee_img tr_noresize" eeimg="1"> 的结果：


<img src="https://www.zhihu.com/equation?tex=O_i^{j+1} = P_{i,:j+1} * V_{:j+1} \\
= softmax(S_{i,:j+1}) * V_{:j+1} \\
= diag(l^{(j+1)})^{-1} [exp([S_{i,:j},S_{i(j+1)}] - m^{(j+1)})] \begin{pmatrix}V_{:j} \\V_{j+1} \end{pmatrix} \\
= diag(l^{(j+1)})^{-1} [exp(S_{i:j} - m^{(j+1)})V_{:j} + exp(S_{i(j+1)}- m^{(j+1)})V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [e^{-m^{(j+1)}} exp(S_{i,:j}) V_{:j} + e^{-m^{(j+1)}} exp(S_{i(j+1)}) V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [diag(l^{(j)}) e^{m^{(j)}-m^{(j+1)}} diag(l^{(j)})^{-1} exp(S_{i,:j}-m^{(j)}) V_{:j} + e^{-m^{(j+1)}} exp(S_{i(j+1)}) V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [diag(l^{(j)}) e^{m^{(j)}-m^{(j+1)}} O_i^{(j)} + e^{\tilde{m}-m^{(j+1)}} exp(S_{i(j+1)}-\tilde{m}) V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [diag(l^{(j)}) e^{m^{(j)}-m^{(j+1)}} O_i^{(j)} + e^{\tilde{m}-m^{(j+1)}} \tilde{P}_{i(j+1)} V_{j+1}]
" alt="O_i^{j+1} = P_{i,:j+1} * V_{:j+1} \\
= softmax(S_{i,:j+1}) * V_{:j+1} \\
= diag(l^{(j+1)})^{-1} [exp([S_{i,:j},S_{i(j+1)}] - m^{(j+1)})] \begin{pmatrix}V_{:j} \\V_{j+1} \end{pmatrix} \\
= diag(l^{(j+1)})^{-1} [exp(S_{i:j} - m^{(j+1)})V_{:j} + exp(S_{i(j+1)}- m^{(j+1)})V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [e^{-m^{(j+1)}} exp(S_{i,:j}) V_{:j} + e^{-m^{(j+1)}} exp(S_{i(j+1)}) V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [diag(l^{(j)}) e^{m^{(j)}-m^{(j+1)}} diag(l^{(j)})^{-1} exp(S_{i,:j}-m^{(j)}) V_{:j} + e^{-m^{(j+1)}} exp(S_{i(j+1)}) V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [diag(l^{(j)}) e^{m^{(j)}-m^{(j+1)}} O_i^{(j)} + e^{\tilde{m}-m^{(j+1)}} exp(S_{i(j+1)}-\tilde{m}) V_{j+1}] \\
= diag(l^{(j+1)})^{-1} [diag(l^{(j)}) e^{m^{(j)}-m^{(j+1)}} O_i^{(j)} + e^{\tilde{m}-m^{(j+1)}} \tilde{P}_{i(j+1)} V_{j+1}]
" class="ee_img tr_noresize" eeimg="1">

### backward过程

- softmax求导

对于 <img src="https://www.zhihu.com/equation?tex=y = softmax(z), L = f(y)" alt="y = softmax(z), L = f(y)" class="ee_img tr_noresize" eeimg="1"> :


<img src="https://www.zhihu.com/equation?tex=\frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z_j}" alt="\frac{\partial L}{\partial z_j} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z_j}" class="ee_img tr_noresize" eeimg="1">

我们分析其中的 <img src="https://www.zhihu.com/equation?tex=\frac{\partial y}{\partial z_j}" alt="\frac{\partial y}{\partial z_j}" class="ee_img tr_noresize" eeimg="1"> :


<img src="https://www.zhihu.com/equation?tex=\left\{
\begin{aligned}
\frac{\partial y_i}{\partial x_j} = y_i(1-y_i), 当i = j\\
\frac{\partial y_i}{\partial x_j} = -y_iy_j, 当i \neq j
\end{aligned}
\right.
\\
" alt="\left\{
\begin{aligned}
\frac{\partial y_i}{\partial x_j} = y_i(1-y_i), 当i = j\\
\frac{\partial y_i}{\partial x_j} = -y_iy_j, 当i \neq j
\end{aligned}
\right.
\\
" class="ee_img tr_noresize" eeimg="1">

最终我们有：


<img src="https://www.zhihu.com/equation?tex=\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z} = \frac{\partial L}{\partial y}(diag(y) - y^Ty)\\
\frac{\partial L}{\partial z_i} = y_i(\frac{\partial L}{\partial y_i} - \sum_{j=1}^n \frac{\partial L}{\partial y_j} y_j)
" alt="\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z} = \frac{\partial L}{\partial y}(diag(y) - y^Ty)\\
\frac{\partial L}{\partial z_i} = y_i(\frac{\partial L}{\partial y_i} - \sum_{j=1}^n \frac{\partial L}{\partial y_j} y_j)
" class="ee_img tr_noresize" eeimg="1">

- 标准backward计算

![alt text](https://img.remit.ee/main/image-33.png)

- 分块backward计算

![alt text](https://img.remit.ee/main/image-34.png)

11-15行：重计算过程

16行：

<img src="https://www.zhihu.com/equation?tex=dV_j = \sum_i (P_{ij})^T dO_i
" alt="dV_j = \sum_i (P_{ij})^T dO_i
" class="ee_img tr_noresize" eeimg="1">

17行：

<img src="https://www.zhihu.com/equation?tex=dP_{ij} = dO_i V_j^T
" alt="dP_{ij} = dO_i V_j^T
" class="ee_img tr_noresize" eeimg="1">

19-20行：

结论如下：

<img src="https://www.zhihu.com/equation?tex=\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z} = \frac{\partial L}{\partial y}(diag(y) - y^Ty)\\
" alt="\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z} = \frac{\partial L}{\partial y}(diag(y) - y^Ty)\\
" class="ee_img tr_noresize" eeimg="1">

以下是推导：


<img src="https://www.zhihu.com/equation?tex=ds_i = dp_i(diag(p_i) - p_i^Tp_i) \\
= dp_i diag(p_i) - dp_i  p_i^Tp_i \\
= dp_i diag(p_i) - do_i V^T p_i^T p_i \\
= dp_i diag(p_i) - do_i o_i^T p_i \\
= p_i \circ [dp_i - rowsum(do_i \circ o_i)] \\
dS_{ij} = P_{ij} \circ [dP_{ij} - rowsum(do_i \circ o_i)]
" alt="ds_i = dp_i(diag(p_i) - p_i^Tp_i) \\
= dp_i diag(p_i) - dp_i  p_i^Tp_i \\
= dp_i diag(p_i) - do_i V^T p_i^T p_i \\
= dp_i diag(p_i) - do_i o_i^T p_i \\
= p_i \circ [dp_i - rowsum(do_i \circ o_i)] \\
dS_{ij} = P_{ij} \circ [dP_{ij} - rowsum(do_i \circ o_i)]
" class="ee_img tr_noresize" eeimg="1">

21行：


<img src="https://www.zhihu.com/equation?tex=dQ_i = \sum_j dS_{ij} K_j
" alt="dQ_i = \sum_j dS_{ij} K_j
" class="ee_img tr_noresize" eeimg="1">

22行：


<img src="https://www.zhihu.com/equation?tex=dK_{ij} = \sum_i dS_{ij}^T Q_i
" alt="dK_{ij} = \sum_i dS_{ij}^T Q_i
" class="ee_img tr_noresize" eeimg="1">

### 源码学习（triton实现）

传送门：(https://github.com/DL-Attention/flash-attention-1/blob/main/flash_attn/flash_attn_triton.py)

我主要看了_fwd_kernel以及_bwd_kernel这两个核心函数：

**_fwd_kernel**：

索引初始化：

```python
    start_m = tl.program_id(0)      # 当前block处理Q的第几个块（行）
    off_hb = tl.program_id(1)       # batch*head的全局索引
    off_b = off_hb // nheads        # batch索引
    off_h = off_hb % nheads         # head索引
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # Q的行
    offs_n = tl.arange(0, BLOCK_N) # K/V的列
    offs_d = tl.arange(0, BLOCK_HEADDIM) #head)dim维度下标
```

初始化Q/K/V/bias的指针：

```python
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
        if BIAS_TYPE == 'vector':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])
```

遍历K/V块：

```python
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
```

QK相乘计算，mask，softmax缩放，中间变量：

```python
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True) #Q * K^T计算

# ... mask、softmax缩放

    acc_o_scale = tl.exp(m_i - m_ij)
    tl.store(t_ptrs, acc_o_scale)
    acc_o_scale = tl.load(t_ptrs)
    acc_o = acc_o * acc_o_scale[:, None] #对应于伪代码12行后一项的迭代更新

# ... load V
    p = p.to(v.dtype)
    acc_o += tl.dot(p, v) # 前一项更新
    m_i = m_ij
    l_i_new = tl.exp(lse_i - m_ij) + l_ij
    lse_i = m_ij + tl.log(l_i_new) #更新中间变量
```

结束循环后，最终归一化：

```python
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
```

**疑问：这个代码中，Q是在KV循环之外被载入的，与伪代码有一定的不同**；

> 2025-6-25: 破案了，看了v2的实现，应该是优化内存读取的操作

**_bwd_kernel**:

_bwd_kernel对K维度进行分块执行，调用_bwd_kernel_one_col_block实现，整个实现的顺序逻辑和伪代码相同；

## FlashAttention v2

### forward过程

![alt text](https://img.remit.ee/main/image-35.png)

我们可以看一下和v1版本在forward上的主要变化：

1. Q的搬运被放到了K块循环的外边，不用重复地从HBM中拿出O的迭代结果了
2. 我们发现O更新时少了一项 <img src="https://www.zhihu.com/equation?tex=diag(l_i^{(j)})^{-1}" alt="diag(l_i^{(j)})^{-1}" class="ee_img tr_noresize" eeimg="1"> , 这个操作被拿到循环外侧了，降低了非矩阵计算的次数
3. 13行多了一个 <img src="https://www.zhihu.com/equation?tex=L_i" alt="L_i" class="ee_img tr_noresize" eeimg="1"> 的计算： <img src="https://www.zhihu.com/equation?tex=L_i = m_i^{(T_c)} + log(l_i^{(T_c)})" alt="L_i = m_i^{(T_c)} + log(l_i^{(T_c)})" class="ee_img tr_noresize" eeimg="1"> , 在backward阶段起作用（这里的m，l是全局的rowmax、rowsum）

### backward过程

![alt text](https://img.remit.ee/main/image-36.png)

在backward中，循环的顺序又和v1相同了，原因如下：

> 对 <img src="https://www.zhihu.com/equation?tex=dV_j" alt="dV_j" class="ee_img tr_noresize" eeimg="1"> , 沿着i方向all-reduce,按行汇总， <img src="https://www.zhihu.com/equation?tex=dK_j" alt="dK_j" class="ee_img tr_noresize" eeimg="1"> 也是一样； <img src="https://www.zhihu.com/equation?tex=dQ_i" alt="dQ_i" class="ee_img tr_noresize" eeimg="1"> 沿着j方向做all-reduce，按列汇总； <img src="https://www.zhihu.com/equation?tex=dS_{ij},dP_{ij}" alt="dS_{ij},dP_{ij}" class="ee_img tr_noresize" eeimg="1"> 只和当前的ij块有关；综上来看，我们把KV放在外循环更合适一些。

上面提到的 <img src="https://www.zhihu.com/equation?tex=L_i" alt="L_i" class="ee_img tr_noresize" eeimg="1"> 在backward的重计算阶段起到了作用：


<img src="https://www.zhihu.com/equation?tex=P_i^{(j)} = diag(l_i^{(j)})^{-1} exp(S_i^{(j)} - m_i^{(j)}) = exp(S_i^{(j)} - L_i)
" alt="P_i^{(j)} = diag(l_i^{(j)})^{-1} exp(S_i^{(j)} - m_i^{(j)}) = exp(S_i^{(j)} - L_i)
" class="ee_img tr_noresize" eeimg="1">

通过这种方法，减少了原本l, m的读写量

这张图展示了循环层次不同，worker的分割方式是不同的：
![alt text](https://img.remit.ee/main/image-39.png)

### Thread Blocks优化

v2 新增了seq_len维度的分块，将seq_len划分为num_m_block份，每份长度kBlockM，block的数量由batch_size * num_heads变为了batch_size * num_heads * num_m_block；**这一操作的目的是尽量将SM吃满**。

![alt text](https://img.remit.ee/main/image-37.png)
![alt text](https://img.remit.ee/main/image-38.png)

### Wrap层并行

![alt text](https://img.remit.ee/main/image-40.png)

对于v1，每个wrap在列方向上的对应结果，需要不同wrap之间通信，对结果聚合才能得到最终的 <img src="https://www.zhihu.com/equation?tex=O" alt="O" class="ee_img tr_noresize" eeimg="1"> ,而v2中，不同wrap中的计算是独立的，不需要进行wrap之间的通讯；

## FlashAttention v3

v3基于Hopper架构进行优化，遂简单了解一番Hopper架构。

### Hopper Architecture GPU

1. 更强的SM，相比于A100，SM更多（132个），运算速度更强
2. FP8 Tensor core: 对FP8运算更加兼容
3. Thread Block Cluster：在Thread Block上新增一个 Thread Block Cluster 层，能更灵活地调度空间：
![alt text](https://img.remit.ee/main/image-41.png)
![alt text](https://img.remit.ee/main/image-42.png)
Thread Block Cluster 在硬件上对应Graph Processing Cluster（GPC），GPC提供了SM-to-SM Network，加速不同SM之间的数据传输，资料存放的位置为distributed shared memory (DSMEM)：
![alt text](https://img.remit.ee/main/image-43.png)
每个Thread Block也成为cooperative thread arrays (CTA，后面有用)，对应SM；
4. Tensor Memory Accelerator（TMA）:新功能，能够使SM的计算任务与数据传输任务重叠：
![alt text](https://img.remit.ee/main/image-44.png)
5. Register Dynamic Reallocation：Wrap Group中的register可以做reallocate，能够让我们有更多的RMEM能够使用；

### 算法1：数据搬运、计算异步（Warp-specialization）

Warp-specialization是指将Thread Block中的wraps分成Producer Warp Group和Consumer Warp Group；

使用生产者-消费者算法：

> Producer：对应上面提到的TMA，做数据搬运工作（从HBM拉取到SMEM）
> Consumer：对应Tensor Core，负责计算任务

![alt text](https://img.remit.ee/main/image-45.png)

Producer: 现将 <img src="https://www.zhihu.com/equation?tex=Q_i" alt="Q_i" class="ee_img tr_noresize" eeimg="1"> 先搬运到SMEM中，在根据Consumer的需求将 <img src="https://www.zhihu.com/equation?tex=K_i, V_i" alt="K_i, V_i" class="ee_img tr_noresize" eeimg="1"> 搬运到SMEM中；
Consumer: 和v2的算法是相近的，只是在使用QKV之前多了等待加载的动作；

> ques: 两次矩阵乘法一次是SS-GEMM, 一次是RS-GEMM，猜测是由于第二次乘法的操作数 <img src="https://www.zhihu.com/equation?tex=O_i" alt="O_i" class="ee_img tr_noresize" eeimg="1"> 在register中

### 算法2：Pingpong scheduling

>背景：在Hopper架构中，softmax和GEMM在不同的模块（multi-function unit 和 Tensor Core）上运算，在这一条件下，可以让softmax和GEMM做异步计算

我们现在有两个warp group，我们可以强制warp group 2在GEMM0做完后，warp group 1才能做GEMM1，这样就完成了计算上的重叠：

![alt text](https://img.remit.ee/main/image-46.png)

### 算法3：Intra-warpgroup overlapping

对于一个wrap group我们也能做相同的事情，只要让GEMM1延迟到下一轮计算中softmax的计算开始时即可：

![alt text](https://img.remit.ee/main/image-47.png)

伪代码如下：

![alt text](https://img.remit.ee/main/image-48.png)

### 算法4：3-stage pipelining

和上一个算法是类似的，核心思路是，在softmax的运算时间比GEMM长，我们可以在第i轮的softmax期间，完成i+1轮的GEMM0和第i-1轮的GEMM1：

![alt text](https://img.remit.ee/main/image-49.png)

伪代码如下：

![alt text](https://img.remit.ee/main/image-50.png)

**FP8优化、backward**：

TODO

## Megatron-LM 代码学习

选择了codegeex的训练代码来看，我们从入口pretrain_codegeex开始看：

```python
if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
```

我们继续追溯pretrain函数，这里我给出函数的主要功能部分：

```python
def pretrain(
    train_valid_test_dataset_provider,
    model_provider,
    forward_step_func,
    valid_forward_step_func=None,
    extra_args_provider=None,
    args_defaults={},
):
    # 对Megatron进行初始化，准备分布式训练环境
    initialize_megatron(
        extra_args_provider=extra_args_provider, args_defaults=args_defaults
    )
    # ......
    # 同步训练开始时间，取最小的
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    # ......
    # 获得Model、Optimizer、lr-scheduler
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    # ......
    # 加载数据集
    # 训练启动
    if args.do_train and args.train_iters > 0:
        iteration = train(
            forward_step_func,
            valid_forward_step_func,
            model,
            optimizer,
            lr_scheduler,
            train_data_iterator,
            valid_data_iterator,
        )
    print_datetime("after training is done")
    # evaluate、test过程
```

### initialize_megatron

接下来我们看initialize_megatron过程：

```python
def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
):
    # ......
    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed)
    if args.lazy_mpu_init:
        # ......
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()
        # ......
```

我们可以看到，initialize_megatron中的主要步骤为_initialize_distributed函数，我们继续看这个函数：

```python 
def _initialize_distributed():
    if torch.distributed.is_initialized():
        #初始化成功
    else:
        #初始化流程
        if args.rank == 0:
        print("> initializing torch distributed ...", flush=True)
        # 将local-rank初始化成rank % device-count.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            if args.force_device is not None:
                print(
                    f"  > forcefully set the device to {args.force_device}, originally {device}"
                )
                device = args.force_device
            torch.cuda.set_device(device)
        # 将进程大组初始化
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost") # rank 0 的进程ip
        master_port = os.getenv("MASTER_PORT", "6000") # rank 0 的进程端口
        init_method += master_ip + ":" + master_port
        # ......
        torch.distributed.init_process_group( #生成大的进程组
            backend=args.distributed_backend, # nccl，gloo等，后端
            world_size=args.world_size, # 全局进程数
            rank=args.rank, # 当前进程的rank
            init_method=init_method, # 用来“链接彼此”的通讯地址
            timeout=timeout #进程等待时间
        )

    # 对DP/TP/PP设置进程子组
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
            )
    # deepspeed ZERO优化的checkpoint
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        setup_deepspeed_random_and_activation_checkpointing(args)
```

对于initialize_model_parallel，我们看一下他的划分方式：

![alt text](https://img.remit.ee/main/image-51.png)

所有的节点构成一个进程大组；
一份模型参数应该在一个MP(Model Parallism)组上，对图中：[[g0, g1, g4, g5, g8, g9, g12, g13], [g2, g3, g6, g7, g10, g11, g14, g15]]
张量并行间需要通信的节点放在一个TP组上：[[g0, g1], [g4, g5],[g8, g9], [g12, g13], [g2, g3], [g6, g7], [g10, g11], [g14, g15]]
流水线并行对应PP组：[[g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]]
数据并行需要通信的节点放在同一个DP组上：[[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]]

> 分组时，原则是根据通讯量划分组别，一般TP、DP不跨机，DP跨机实现

```python
def initialize_model_parallel(
    tensor_model_parallel_size_=1, #每个TP组的线程数量
    pipeline_model_parallel_size_=1, #每个PP组的线程数量
    virtual_pipeline_model_parallel_size_=None,
):
    # 确认可分性
    # ......
    data_parallel_size = world_size // (  #根据给出的两个参数，可以确认DP组的大小
        tensor_model_parallel_size * pipeline_model_parallel_size
    )
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size
    # ......
    # DP初始化
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
    # MP、PP、TP的初始化，方法类似
```

### get_model, 模型划分

我们继续看pretrain函数中的setup_model_and_optimizer，我们主要看加载、分割模型的部分，也就是get_model函数：

```python
def get_model(model_provider_func):
    args = get_args()

    # 返回cpu上的模型
    if ( #virtual pipeline（优化技巧）
        mpu.get_pipeline_model_parallel_world_size() > 1
        and args.virtual_pipeline_model_parallel_size is not None
    ):
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process, post_process=post_process
            )
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage() # 判断是否是PP组的第一个进程
        post_process = mpu.is_pipeline_last_stage() # 判断是否是PP组的最后一个进程
        model = model_provider_func(pre_process=pre_process, post_process=post_process) #返回模型

    #......

    #如果用了deepspeed，搬运工作直接交给deepspeed
    if args.deepspeed:
        return model

    # 将cpu上的模型搬运到gpu上
    print(f" > moving model to GPU ...", flush=True)
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())
    print(f" > moving to GPU done", flush=True)

    # Fp16转换
    if args.fp16 or args.bf16:
        print(f" > converting model to fp16 ...", flush=True)
        model = [Float16Module(model_module, args) for model_module in model]
        print(f" > converting to fp16 done", flush=True)

    #使用pytorch定义的DistributedDataParallel管理数据并行
    if args.DDP_impl == "torch":
        i = torch.cuda.current_device()
        model = [
            torchDDP(
                model_module,
                device_ids=[i],
                output_device=i,
                process_group=mpu.get_data_parallel_group(),
            )
            for model_module in model
        ]
        return model

    #自定义的DistributedDataParallel管理数据并行
    if args.DDP_impl == "local":
        print(f" > creating DDP model ...", flush=True)
        model = [
            LocalDDP(
                model_module,
                args.accumulate_allreduce_grads_in_fp32,
                args.use_contiguous_buffers_in_ddp,
            )
            for model_module in model
        ]
        print(f" > creating DDP model done", flush=True)
        return model

    raise NotImplementedError(
        "Unknown DDP implementation specified: {}. " "Exiting.".format(args.DDP_impl)
    )

```

下面我们来看get_model传入的参数model_provider，它被定义在pretrain_codegeex.py中，也就是我们的训练入口：

```python
def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    # ......
    with deepspeed.zero.Init(
        data_parallel_group=mpu.get_data_parallel_group(),
        remote_device=None if args.remote_device == "none" else args.remote_device,
        config_dict_or_path=args.deepspeed_config,
        enabled=args.zero_stage == 3,
        mpu=mpu,
    ):
        if args.deepspeed and not args.no_pipeline_parallel:
            #使用deepspeed载入模型
        else:
            model = CodeGeeXModel( #定义的并行模型文件
                num_tokentypes=0,
                parallel_output=True,
            )
            
            if args.load_state is not None:
                timers = get_timers()
                print_rank_0("Loading warmstarting model states ...")
                timers("load-model-states").start()
                mp_rank = mpu.get_tensor_model_parallel_rank()
                if os.path.isdir(args.load_state):
                    model_path = os.path.join(
                        args.load_state, "mp_rank_{:02d}_model_states.pt".format(mp_rank)
                    )
                else:
                    model_path = args.load_state
                print_rank_0(f"Loading model from {model_path} ...")
                state_dict = torch.load(model_path, map_location="cpu") #将模型加载到cpu中
                if "module" in state_dict:
                    state_dict = state_dict["module"]  # strip other client states
                model.load_state_dict(state_dict)
                timers("load-model-states").stop()
                timers.log(["load-model-states"])
    # ......
    return model
```

### CodeGeeXModel，模型类的实现

我们在看CodeGeeXModel的实现之前，先看一下它的父类MegatronModule：

```python
class MegatronModule(torch.nn.Module): #继承自nn.Module

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings  #input 与output 是否共享Word Embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        return self.state_dict(destination, prefix, keep_vars) #保存checkpoint

    def word_embeddings_weight(self): #获取Word Embeddings
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            return self.language_model.embedding.word_embeddings.weight
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception( #强制使用同一个Word Embeddings
                    "word_embeddings_weight() called for last "
                    "stage, but share_word_embeddings is false"
                )
            return self.word_embeddings.weight
        raise Exception( #如果是中间层，那么没有Word Embeddings
            "word_embeddings_weight() should be " "called for first and last stage only"
        )

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings: #强制share embeddingg
            raise Exception(
                "initialize_word_embeddings() was called but "
                "share_word_embeddings is false"
            )

        # 并行度为1时，输入输出层在同一张卡上，不需要强制共享
        if args.pipeline_model_parallel_size == 1:
            return

        if mpu.is_pipeline_last_stage(): #PP组最后一个进程
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = "word_embeddings_for_head"
            
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size,
                args.hidden_size,
                init_method=init_method_normal(args.init_method_std),
            )
            self.word_embeddings.weight.data.fill_(0) #用0填充word_embeddings
            self.word_embeddings.weight.shared = True

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                torch.distributed.all_reduce( #输入输出层all-reduce，确保we是相同的
                    self.word_embeddings_weight().data, group=mpu.get_embedding_group()
                )
        else:
            # 初始化出错
```

下面我们来看CodeGeeX中各个类的实现：

![alt text](https://img.remit.ee/main/image-52.png)

首先是Embedding类，类中实现了word embeddings以及position embeddings（nn.Embedding），我们主要看一下word embeddings的并行实现VocabParallelEmbedding：

```python
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()

        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim # hidden_size

        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size() #当前进程所在TP组的进程数
        # 在vocab_size维度对word-embedding进行切割（横切）
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size( #计算当前词汇表范围
            self.num_embeddings, #vocab_size
            get_tensor_model_parallel_rank(), #当前rank
            self.tensor_model_parallel_size, #组内的进程总数
        )
        self.num_embeddings_per_partition = ( #每个partition的宽度
            self.vocab_end_index - self.vocab_start_index
        )

        #载入参数
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter( #载入完整的we
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=args.params_dtype,
                    # dtype=torch.float32,
                )
            )
            _initialize_affine_weight_cpu( #对we进行切割
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
            )
        else: #gpu载入，同理
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype,
                    # dtype=torch.float32,
                )
            )
            _initialize_affine_weight_gpu( #与cpu不同，不同进程的分块要使用不同的seed
                self.weight, init_method, partition_dim=0, stride=1
            )

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # 对input进行mask，只处理范围内的词
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight, #切割好的权重
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # 不同组的结果做all reduce
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output
```

对于ParallelSelfAttention，我们对QKV矩阵列切分（也就是按注意力头切分），对线性层行切分；这样，我们在前向的最后进行一次all-reduce，反向的最后也进行一次all-reduce，降低通信成本。我们主要看一下矩阵乘的行分割以及列分割的实现。

**列切分：ColumnParallelLinear**:

> f算子（_CopyToModelParallelRegion）:在forward中copy输入X，backward中对梯度做All-Reduce；
> g算子（_GatherFromModelParallelRegion）:在forward中对输出all-gather,backward中将梯度split

```python

class _CopyToModelParallelRegion(torch.autograd.Function): #f算子
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output) #将梯度加起来

class _GatherFromModelParallelRegion(torch.autograd.Function): #g算子
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_) #聚合输出

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)

class ColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size, #参数的第一个维度
        output_size,#参数的第二个维度
        bias=True,
        gather_output=True, #是否对输出做all-gather
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        skip_init=False,
        device=None,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size() #当前TP组中的进程总数
        self.output_size_per_partition = divide(output_size, world_size) #分割后的列维度大小
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.device = device
        
        args = get_args()
        if not skip_init:
            if args.use_cpu_initialization: #CPU上初始化参数
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
                self.master_weight = _initialize_affine_weight_cpu( #分割参数
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                )
            else: #gpu上的操作与cpu上是类似的
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=self.device if self.device is not None else torch.cuda.current_device(),
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=0, stride=stride
                )
        else:
            self.register_parameter("weight", None)

        if bias and not skip_init:
            # ......, 对bias进行分割，分割方式和weight是相同的
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_tensor_model_parallel_region(input_) #新建一个_CopyToModelParallelRegion实例

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel) #新建一个_GatherFromModelParallelRegion实例
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

```

**行分割：RowParallelLinear**:

> f算子（_ScatterToModelParallelRegion）：forward过程中按列将输入分割，backward过程all-gather梯度；
> g算子（_ReduceFromModelParallelRegion）：forward过程All-Reduce输出，backward不进行操作，直接传递梯度即可；

```python
class _ScatterToModelParallelRegion(torch.autograd.Function): #f算子
    """Split the input and keep only the corresponding chuck to the rank."""
    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_) #按列分割输入

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output) #all-gather梯度
 
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_) #对结果做All-Reduce

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output #直接返回梯度

class RowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        skip_init=False,
        device=None,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.device = device
        # ......, 参数的载入与分割，和列分割车不多

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_) #实例化f算子
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel) #实例化g算子
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

```
