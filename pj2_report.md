## Project 2  实验报告

>   22302010017 包旭



### 一、文件结构

-   `HMM`：实现了手写 HMM 进行 NER 任务，主文件为 `HMM.py`
-   `CRF`：实现了 CRF 进行 NER 任务
    -   `CRF.py`：进行训练和预测
    -   `generate.py`：根据训练和预测的一定格式的 `.txt` 文件，结合 `NER` 目录下的 `templates_for_crf.utf8` 模板文件，生成一个新的 `.txt` 文件，主文件可以根据新的文本文件生成对应的特征函数，从而利用 CRF 执行 NER 任务
-   `BiLSTM+CRF`：实现了 BiLSTM+CRF 进行 NER 任务， 主文件为 `BiLSTM_CRF.py`





### 二、实验思路和过程

-   关于HMM

​	HMM 任务主要是维护三个矩阵，即初始概率矩阵，转移概率矩阵和发射概率矩阵。得到这三个矩阵并不难，我们可以通过 Python 中提供的字典类型和字典内嵌字典来完成，通过遍历训练集再计算得到相应的概率值。

​	需要注意的是，验证集上可能出现训练集中没出现过的观测值、转移模式等。这个时候如果不做特殊处理，在viterbi 解码的时候就会出现相乘之后整个序列的概率为 0 的情况。由于整个序列的概率是连乘，所以只有有一个地方为 0 就会出问题，综合下来概率为 0 的情况不在少数，已经不是可以忽略不计的情况了。为了解决这个问题，我将三个矩阵的每个元素都做了平滑处理，使其维持在很小的一个值而不为 0 。同样，我在维特比解码的时候也加了检测机制，如果检测到哪个矩阵的元素为 0 就把它改成一个很小的值。同时为了避免发生数据下溢，经过调试，我将该值设为了 1e-10。有了这个机制之后模型的准确率明显上升。

-   关于 CRF 

​	我的 CRF 调用了 `sklearn_crfsuite` 这个库，自己实现的部分就是利用 templates 去套每一个 word ，生成相应的模版，然后生成特征函数，交给模型去计算 loss ,反向传播。因为是直接调库的所以这一部分不是很难，也没遇到什么难题。

-   关于 BiLSTM+CRF

​	CRF 在 BiLSTM 这个任务中的作用其实是优化 LSTM 的学习结果，避免一些显然不可能出现的情况，比如没有 B 这种标签就直接进了 M 。所以为了优化这种情况，CRF 层应该有一个 transition 矩阵，作用类似于 HMM 的转移矩阵。

​	对于输入的序列，先利用 `prepare_sequence` 函数将其转换为张量，然后输入到 LSTM 模型中，经过 embedding层、双向 LSTM 层、输出层得到一个对应各个标签的 emission 得分。接下来的工作就是手写 CRF 层的任务。对于一个长度为 s 的序列，计算出 32^s 种可能中的每一种情况的概率（以中文训练过程为例），即每种可能的得分，再计算出所有情况概率的总和，即所有情况的总分，然后相除取对数得到损失函数，利用 Pytorch 的 backward 的方法去自动优化这个损失函数。然后根据每个序列的得分进行 viterbi 解码，得到一个由数字表示的最佳序列，最后由这个数字序列根据一开始就确定的字典找出索引，得到最佳标签序列。



### 三、实验结果

以下是各个模型在中文和英文验证集上 `check.py` 的运行结果。

-   HMM

    -   中文

        ![](https://pic.imgdb.cn/item/6669a370d9c307b7e94058ae.png)

    -   英文

        ![](https://pic.imgdb.cn/item/6669a398d9c307b7e940e27a.png)

-   CRF

    -   中文

        ![](https://pic.imgdb.cn/item/6669a428d9c307b7e942f028.png)

    -   英文

        ![](https://pic.imgdb.cn/item/6669a3edd9c307b7e9421005.png)

-   BiLSTM + CRF

    -   中文

        ![](https://pic.imgdb.cn/item/6669ba4bd9c307b7e9880dfa.png)

    -   英文

        ![](https://pic.imgdb.cn/item/6669a8f1d9c307b7e9529fdd.png)







 