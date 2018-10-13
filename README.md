gcForest算法解读
===============

# 一. 背景

> gcForest（多粒度级联森林）是南大周志华教授提出的新的决策树集成方法——一种深度森林结构，决策树集成的集成；通过随机森林的级联结构学习。gcForest的性能较之深度神经网络有很强的竞争力，深度神经网络需要花大力气调参，相比之下 gcForest 要容易训练得多。实际上，在几乎完全一样的超参数设置下，gcForest 在处理不同领域（domain）的不同数据时，也能达到极佳的性能。gcForest 的训练过程效率高且可扩展。

> 深度神经网络可以简单的理解为多层非线性函数的堆叠，当人工很难或者不想去寻找两个目标之间的非线性映射关系，就多堆叠几层，让机器自己去学习它们之间的关系，这就是深度学习最初的想法。既然神经网络可以堆叠为深度神经网络，那是不是可以考虑将其他的学习模型堆叠起来，以获取更好的表示性能，gcForest就是基于这种想法提出来的一种深度结构。gcForest通过级联的方式堆叠多层随机森林，以获得更好的特征表示和学习性能。

> 深度神经网络虽然取得很好的性能，但是也存在一些问题。第一、要求大量的训练数据。深度神经网络的模型容量很大，为了获得比较好的泛化性能，需要大量的训练数据，尤其是带标签的数据。获取大规模数据需要耗费很大的人工成本；第二、深度神经网络的计算复杂度很高，需要大量的参数，尤其是有很多超参数（hyper-parameters）需要优化。比如网络层数、层节点数等。所以神经网络的训练需要很多trick；第三、深度神经网络目前最大的问题就是缺少理论解释。

> gcForest使用级联的森林结构来进行表征学习，需要很少的训练数据，就能获得很好的性能，而且基本不怎么需要调节超参数的设置。gcForest不是要推翻深度神经网络，也不是以高性能为目的的研究，只是在深度结构研究方面给我们提供了一些思路，而且确实在一些应用领域获得了很好的结果，是一项很有意义的研究工作。

# 二. 算法介绍

> gcForest有两种应用方法，分别为：Cascade Forest——级联森林 和 Multi-Grained Scanning——多粒度扫描 。具体如下：

## (一) Cascade Forest——级联森林

> 将输入特征经过级联森林，输出新的类别概率向量，将新的类别概率向量链接输入特征作为下一层输入，经过多个级联森林（k-fold交叉验证用于避免过拟合，并决定是否添加级），输出最终类别概率向量。Cascade Forest结构如图所示：

![image](https://github.com/ShaoQiBNU/SQgcForest/blob/master/images/1.png)

> Cascade结构的每一级接收前一级处理的特征信息并将所得结果传输给下一级。每一级由两种类型的森林组成，其中蓝色表示random forests，黑色表示complete-random tree forests。

> 每个complete-random tree forests中包含500个complete-random trees，每个树随机选取数据特征作为划分节点的父节点，当每个叶节点中类型相同或者不超过10个样例时，停止划分，即不再生长。

> 每个random forest包含500个树，与完全随机树森林不同的是树划分父节点的选取是通过随机选取<a href="https://www.codecogs.com/eqnedit.php?latex=\sqrt{d}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\sqrt{d}" title="\sqrt{d}" /></a>（d是输入特征数量）个特征，选最佳的gini指数特征作为父节点划分节点。

> 其中单个森林产生类别预测分布的过程如下：论文中是3分类，则Forest中的每个tree对x分类都会得到一个三维的类别概率向量，将所有结果取平均就得到最终的class vector，如图所示：

![image](https://github.com/ShaoQiBNU/SQgcForest/blob/master/images/2.png)

> 将同一级Forest产生的class vector连接就得到新的x（增强特征augmented feature），连接输入特征作为下一级Forest的输入向量。在Cascade的最后，将众多Forest产生的向量进行取平均，就产生了一个总的三维向量，哪个概率大，则x就属于哪一类。

**k折交叉验证机制：**为了降低过拟合风险，每个森林产生的类向量由k折交叉验证（k-fold cross validation）产生。具体来说，每个实例都将被用作 k -1 次训练数据，产生 k -1 个类向量，然后对其取平均值以产生作为级联中下一级的增强特征的最终类向量。

**early stop机制：**需要注意的是，在扩展一个新的级后，整个级联的性能将在验证集上进行估计，如果没有显着的性能增益，训练过程将终止；因此，级联中级的数量是自动确定的。与模型的复杂性固定的大多数深度神经网络相反，gcForest 能够适当地通过终止训练来决定其模型的复杂度（early stop）。这使得 gcForest 能够适用于不同规模的训练数据，而不局限于大规模训练数据。

## (二) Multi-Grained Scanning——多粒度扫描

> 深度神经网络在处理特征关系方面是强大的，例如，卷积神经网络对图像数据有效，其中原始像素之间的空间关系是关键的。递归神经网络对序列数据有效，其中顺序关系是关键的。受这种认识的启发，论文采用多粒度扫描流程来增强级联森林。通过使用多个尺寸的滑动窗口，最终的变换特征矢量将包括更多的特征。多粒度扫描就是将原始数据转化为输入向量格式。主要有两种类型：序列数据和图片数据，作用跟DNN类似。如图所示：

![image](https://github.com/ShaoQiBNU/SQgcForest/blob/master/images/3.png)

> 假设序列数据特征维度是400，特征窗口大小为100，滑动步长为1，则通过滑动特征窗口将会得到301个100维的向量。通过两个森林训练得到301个3维的向量，并且将得到的概率向量连接为转换后的输入向量。维度变化过程：

```
400维 -> 301个100维 -> 2棵森林，301个3维 -> 1806维(2x301x3) 
```

> 假设图片特征是20 x 20，特征窗口大小选择为10 x 10，滑动步长为1，则通过滑动特征窗口得到121个10 x 10的矩阵，通过两个森林训练得到121个3维的向量，并且将得到的概率向量连接为转换后的输入向量。维度变化过程：

```
20 x 20 -> 121个10 x 10 -> 2棵森林，121个3维 -> 726维(2x121x3)
```

## (三) 整体结构

> gcForest的整体结构如图所示：第一阶段使用多个尺寸的滑动窗口得到不同尺度的的输出，级联在第二阶段的不同level层 。

![image](https://github.com/ShaoQiBNU/SQgcForest/blob/master/images/4.png)

> 也有另外一种级联方式，如图所示：第一阶段使用多个尺寸的滑动窗口得到不同尺度的的输出，然后连接到一起，整体级联在第二阶段的不同level层 。

![image](https://github.com/ShaoQiBNU/SQgcForest/blob/master/images/5.png)

# 三. 应用

> gcForest代码在可以https://github.com/kingfengji/gcForest和https://github.com/pylablanche/gcForest找到。前者是python2.7的版本，后者是3.0+版本，后者参数说明如下：

```
shape_1X: int or tuple list or np.array (default=None)
    训练量样本的大小，格式为[n_lines, n_cols]，调用mg_scanning时需要！对于序列数据，可以给出单个int。

n_mgsRFtree: int (default=30)
    多粒度扫描时构建随机森林使用的决策树数量.

window: int (default=None)
    多粒度扫描期间使用的窗口大小列表。如果“无”，则不进行切片，window的选择决定了不同的粒度，如5，则只用5的窗口去滑动，而[4,5]则是用4和5分别滑动，即多粒度扫描。

stride: int (default=1)
    切片数据时使用的滑动间隔，类似于CNN中的stride。

cascade_test_size: float or int (default=0.2)
    级联训练集分裂的分数或绝对数，即训练时的测试集大小.

n_cascadeRF: int (default=2)
    级联层中随机森林的数量，对于每个伪随机森林，创建完整的随机森林，因此一层中随机森林的总数将为2 * n_cascadeRF。一个随机森林和一个完全随机树森林

n_cascadeRFtree: int (default=101)
    每个级联层的随即森林中包含的决策树的数量.

min_samples_mgs: float or int (default=0.1)
    多粒度扫描期间，要执行拆分行为时节点中最小样本数. 如果int ，number_of_samples = int。 如果float，min_samples 表示要考虑的初始n_samples的分数。

min_samples_cascade: float or int (default=0.1)
    训练级联层时，要执行拆分行为时节点中最小样本数. 如果int number_of_samples = int。 如果float，min_samples表示要考虑的初始n_samples的分数。

cascade_layer: int (default=np.inf)
    级联层层数的最大值，用来限制级联的结构。一般模型可以自己根据交叉验证结果选取出合适的级联。

tolerance: float (default=0.0)
    判断级联层是否增长的准确度公差。如果准确性的提高不如tolerance，那么层数将
    停止增长。

n_jobs: int (default=1)
    随机森林并行运行的工作数量。如果为-1，则设置为cpu核心数.
```

> 具体应用见上面两个链接，应用gcForest实现MNIST判别，代码如下：

```python
'''
Apply gcForest on MNIST
'''

######################### load packages #######################
import numpy as np
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from GCForest import gcForest

######################### load datasets #######################
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, y_train = X_train[:2000], y_train[:2000]

######################### reshape #######################
X_train = X_train.reshape((2000, 784))
X_test = X_test.reshape((10000, 784))

######################### build model and train #######################
gcf = gcForest(shape_1X=[28,28], window=[7, 10, 14], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7)
gcf.fit(X_train, y_train)

######################### predict #######################
y_pred = gcf.predict(X_test)

######################### evaluating accuracy #######################
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print('gcForest accuracy : {}'.format(accuracy))
```

