import torch

torch.utils.data.DataLoader()

# 但我们常用的主要有5个：

# dataset: Dataset类，决定数据从哪读取以及如何读取
# bathsize: 批大小
# num_works: 是否多进程读取机制
# shuffle: 每个epoch是否乱序
# drop_last: 当样本数不能被batchsize整除时， 是否舍弃最后一批数据

# Epoch： 所有训练样本都已输入到模型中，称为一个Epoch
# Iteration： 一批样本输入到模型中，称为一个Iteration
# Batchsize： 一批样本的大小， 决定一个Epoch有多少个Iteration

# 假设样本总数是80，Batchsize是8，那么1Epoch=10 Iteration。
# 假设样本总数是87，Batchsize是8，如果drop_last=True, 那么1Epoch=10 Iteration。
# 假设样本总数是87，Batchsize是8，如果drop_last=False，那么1Epoch=11 Iteration, 最后1个Iteration有7个样本。


torch.utils.data.Dataset()
# Dataset抽象类，所有自定义的Dataset都需要继承它，并且必须复写__getitem__()这个类方法。
# __getitem__方法的是Dataset的核心，作用是接收一个索引，返回一个样本。
# 参数里面接收index，然后我们需要编写究竟如何根据这个索引去读取我们的数据部分。

# 读哪些数据？ 我们每一次迭代要去读取一个batch_size大小的样本，那么读哪些样本呢？
# 从哪读数据？ 也就是在硬盘当中该怎么去找数据，在哪设置这个参数。
# 怎么读数据？
