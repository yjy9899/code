from __future__ import division
from __future__ import print_function  # 输出函数



import time
import argparse

import torch.optim as optim
import numpy as np
import scipy.sparse as sp

import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import csv
import matplotlib.pyplot as plt
import seaborn as sns


# 导入 NVIDIA System Management Interface 库，用于监控 GPU
# import pynvml
#
# # 初始化 NVIDIA 库
# pynvml.nvmlInit()
# # 获取可用的 GPU 设备数量
# device_count = pynvml.nvmlDeviceGetCount()


# def load_data(path="../apt+14/100%数据/lei_data雷学姐/", dataset="11"):
def load_data(path="data/cora6/", dataset="cora6"):

    # def load_data(path="../apt+14/100%数据/", dataset="cve_3关系"):

    """Load citation network dataset (cora only for now)"""
    # 打印Loading dataset
    print('Loading {} dataset...'.format(dataset))
    # 导入content文件
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 取特征，第一列到倒数第二列
    labels = encode_onehot(idx_features_labels[:, -1])  # 取出所属类别

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 取出节点索引  idx
    idx_map = {j: i for i, j in enumerate(idx)}  # 构造节点的索引字典，将索引转换成从0-整个索引结长度的字典编号 idx_map
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),  # 导入edge数据  表示两个编号节点之间有一条边
                                    dtype=np.int32)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)  # 将之前的edges_unordered编号转换成idx_map字典编号后的边
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # 构建边的邻接矩阵：有边为1，没边为0
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix  计算转置矩阵  将有向图转成无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)  # 对特征做了归一化的操作  normalize归一化函数:按行做了归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))  # 对A+I做归一化的操作  adj + sp.eye(adj.shape[0]):邻接矩阵加上单位阵的操作

    # 训练，验证，测试的样本 100%

    # #100
    # idx_train = range(645) # 训练集
    # idx_val = range(645, 859)  # 验证集
    # idx_test = range(859, 1073)  # 测试集

    # 100  wode
    idx_train = range(1997)  # 训练集
    idx_val = range(1997, 2282)  # 验证集
    idx_test = range(2282, 2853)  # 测试集

    # # 100  lei
    # idx_train = range(1055)  # 训练集
    # idx_val = range(1055, 1186)  # 验证集
    # idx_test = range(1186, 1319)  # 测试集

    # 将numpy的数据转换成torch格式
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# encode_onehot用于将标签进行独热编码
def encode_onehot(labels):
    classes = set(labels)  #
    # 构建类别到独热编码的映射字典，其中np.identity(len(classes))生成一个对角矩阵，enumerate(classes)提供类别的索引。
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):  # 对mx节点做归一化
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 对结点（矩阵）按行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.  # 如果是无穷大（0的倒数），转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 获取模型预测的类别
    correct = preds.eq(labels).double()  # 判断预测正确的标签
    correct = correct.sum()  # 统计正确预测的数量
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# 定义训练模型
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()  # 梯度清零
    output = model(features, adj)  # 运行模型，输入参数（features,ad）
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # 损失函数
    acc_train = accuracy(output[idx_train], labels[idx_train])  # 计算准确率
    loss_train.backward()  # 反向传播
    optimizer.step()  # 更新梯度

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# 在测试阶段计算混淆矩阵
def calculate_confusion_matrix(output, labels):
    # 将概率值转换为预测结果
    predictions = output.argmax(dim=1)
    # 获取类别数量
    num_classes = output.shape[1]
    # 初始化混淆矩阵为全零矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes)
    # 计算混淆矩阵中的交叉频数
    for i in range(len(labels)):
        confusion_matrix[labels[i], predictions[i]] += 1
    return confusion_matrix


def find_misclassified_samples(output, labels):
    # 将概率值转换为预测结果
    predictions = output.argmax(dim=1)

    # 找到被误判的数据的索引
    misclassified_indices = (predictions != labels).nonzero().squeeze()

    # 找到被误判成的类别
    misclassified_classes = predictions[misclassified_indices]

    return misclassified_indices, misclassified_classes


def find_tp_fp_tn_fn_indices(output, labels, class_index):
    # 将概率值转换为预测结果
    predictions = output.argmax(dim=1)

    # 找到真阳性实例的索引
    tp_indices = (predictions == class_index) & (labels == class_index)

    # 找到假阳性实例的索引
    fp_indices = (predictions == class_index) & (labels != class_index)

    # 找到真阴性实例的索引
    tn_indices = (predictions != class_index) & (labels != class_index)

    # 找到假阴性实例的索引
    fn_indices = (predictions != class_index) & (labels == class_index)

    return tp_indices.nonzero().squeeze(), fp_indices.nonzero().squeeze(), tn_indices.nonzero().squeeze(), fn_indices.nonzero().squeeze()


def get_positive_negative_predictions(output, threshold=0.5):
    # 将概率值转换为二分类结果
    predictions = output.argmax(dim=1)
    # 根据阈值判断正负例子
    positive_predictions = (output[:, 1] >= threshold).nonzero().squeeze()
    negative_predictions = (output[:, 0] >= threshold).nonzero().squeeze()
    return positive_predictions, negative_predictions

# # Visualize the confusion matrix
# def plot_confusion_matrix(confusion_matrix):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(confusion_matrix, annot=True, fmt="f", cmap="Blues", cbar=False)
#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")
#     plt.title("Confusion Matrix")
#     plt.savefig("confusion_matrix.jpg")
#     plt.show()

# 定义测试模型
def test():
    # for i in range(device_count):
    #     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    #     util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    #     print(f"GPU {i} 使用率: {util.gpu}%")
    #
    #
    # pynvml.nvmlShutdown()

    model.eval()
    output = model(features, adj)
    positive_predictions, negative_predictions = get_positive_negative_predictions(output, threshold=0.5)

    # 输出正负例子结果
    print("Positive Predictions:", positive_predictions)
    print("Negative Predictions:", negative_predictions)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # 获取混淆矩阵
    confusion_matrix = calculate_confusion_matrix(output, labels)
    # 打印混淆矩阵
    print("Confusion Matrix全部:")
    print(confusion_matrix)

    # Plot and save the confusion matrix as a JPG image
    # plot_confusion_matrix(confusion_matrix)

    misclassified_indices, misclassified_classes = find_misclassified_samples(output, labels)

    # 计算每个类别的准确率、精确率、召回率和F1-Score
    num_classes = output.shape[1]
    for class_index in range(num_classes):
        tp_indices, fp_indices, tn_indices, fn_indices = find_tp_fp_tn_fn_indices(output, labels, class_index)

        # 计算准确率
        tp = torch.sum(tp_indices).item()
        fp = torch.sum(fp_indices).item()
        tn = torch.sum(tn_indices).item()
        fn = torch.sum(fn_indices).item()
        accuracy_class = (tp + tn) / (tp + fp + tn + fn)

        # 计算精确率
        if tp + fp == 0:
            precision_class = 0
        else:
            precision_class = tp / (tp + fp)

        # 计算召回率
        if tp + fn == 0:
            recall_class = 0
        else:
            recall_class = tp / (tp + fn)

        # 计算F1-Score
        if precision_class + recall_class == 0:
            f1_score_class = 0
        else:
            f1_score_class = 2 * (precision_class * recall_class) / (precision_class + recall_class)

        # 输出每个类别的性能指标
        print("Class {}: Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}".format(class_index,
                                                                                                       accuracy_class,
                                                                                                       precision_class,
                                                                                                       recall_class,
                                                                                                       f1_score_class))

    for i in range(len(misclassified_indices)):
        index = misclassified_indices[i].item()
        true_class = labels[index].item()
        misclassified_class = misclassified_classes[i].item()
        print(f"Node {index} - True Class: {true_class}, Misclassified as Class {misclassified_class} ")
        # print(f"Node {index} - True Class: {true_class}, Misclassified as Class: {misclassified_class} (Index in All Data: {index})")

    # 输出被误判的数据的索引
    print("Misclassified Samples Index: ", misclassified_indices)


    # print(len(preds))
    # print(len(labels[idx_test]))
    # print(np.array(preds))
    # print(np.array(labels[idx_test]))
    
    # preds = output[idx_test].max(1)[1].type_as(labels)
    # y_pred = np.array(preds)
    # y_test = np.array(labels[idx_test])

    preds = output[idx_test].max(1)[1].type_as(labels)
    y_pred = preds.cpu().numpy()  # 确保在CPU上
    y_test = labels[idx_test].cpu().numpy()  # 确保在CPU上

    # f1_score = calculate_f1_score(y_test, y_pred)
    # print("F1 Score:", f1_score)
    # sklearn.metrics.precision_score(y_test, y_pred, labels=None, pos_label=1,
    #                                 average='binary', sample_weight=None)

    # 准确率
    accuracy_result = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    print("accuracy:%s" % accuracy_result)
    print("测试准确率: {:.3f}%".format(accuracy_result * 100))

    # 精确率
    percision_result = precision_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    print("precision:%s" % percision_result)
    print("测试精确率: {:.3f}%".format(percision_result * 100))

    # 召回率
    recall_result = recall_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    print("recall:%s" % recall_result)
    print("测试召回率: {:.3f}%".format(recall_result * 100))

    # F1-score
    f1_score_result = f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)
    print("f1_score:%s" % f1_score_result)
    print("测试F1-score值: {:.3f}%".format(f1_score_result * 100))
    #
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    # # 将结果写入CSV
    # writer.writerow({
    #     'lr': lr,
    #     'acc': acc_test.item(),
    #     'loss': loss_test.item()
    # })
    #
    # print(f"完成 lr={lr} 的迭代")





import time
import argparse
import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义超参数空间
param_space = {
    'lr': (0.0001, 0.01),
    'hidden': (64, 256),
    'dropout': (0.1, 0.5),
    'epoch': (0, 500)
}

# GCN模型定义
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, epoch):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.epoch = epoch

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# 适应度函数和优化架构
def evaluate_fitness(params, adj, features, labels, idx_train, idx_val):
    model = GCN(nfeat=features.shape[1], nhid=params['hidden'], nclass=labels.max().item() + 1, dropout=params['dropout'], epoch=params['epoch'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=5e-4)
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
    # optimizer.zero_grad()
    # output = model(features, adj)
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # loss_train.backward()
    # optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        acc_val = accuracy(output[idx_val], labels[idx_val])
    return acc_val

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def initialize_population(size):
    population = []
    for _ in range(size):
        particle = {
            'params': {
                'lr': random.uniform(*param_space['lr']),
                'hidden': random.randint(*param_space['hidden']),
                'dropout': random.uniform(*param_space['dropout']),
                'epoch': random.randint(*param_space['epoch'])
            },
            'velocity': {
                'lr': 0,
                'hidden': 0,
                'dropout': 0,
                'epoch': 0
            }
        }
        population.append(particle)
    return population

def update_velocity_and_position(particle, global_best, w=0.5, c1=1.0, c2=1.0):
    r1, r2 = random.random(), random.random()
    for param in particle['params']:
        v = (w * particle['velocity'][param] +
             c1 * r1 * (particle['best'][param] - particle['params'][param]) +
             c2 * r2 * (global_best['params'][param] - particle['params'][param]))
        particle['velocity'][param] = v
        if param == 'hidden' or param == 'epoch':
            particle['params'][param] = int(max(param_space[param][0], min(param_space[param][1], particle['params'][param] + v)))
        else:
            particle['params'][param] = max(param_space[param][0], min(param_space[param][1], particle['params'][param] + v))


def mutate(particle, mutation_rate=0.1):
    if random.random() < mutation_rate:
        param_to_mutate = random.choice(list(particle['params'].keys()))
        if param_to_mutate == 'hidden' or param_to_mutate == 'epoch':
            particle['params'][param_to_mutate] = random.randint(*param_space[param_to_mutate])
        else:
            particle['params'][param_to_mutate] = random.uniform(*param_space[param_to_mutate])


def gps_algorithm(population_size, generations, adj, features, labels, idx_train, idx_val):
    population = initialize_population(population_size)
    global_best = None
    best_fitness = float('-inf')
    fitness_history = []

    for generation in range(generations):
        for particle in population:
            fitness = evaluate_fitness(particle['params'], adj, features, labels, idx_train, idx_val)
            if fitness > best_fitness:
                best_fitness = fitness
                global_best = particle
            if 'best_fitness' not in particle or fitness > particle.get('best_fitness', float('-inf')):
                particle['best_fitness'] = fitness
                particle['best'] = particle['params'].copy()

        for particle in population:
            update_velocity_and_position(particle, global_best)
            mutate(particle) 
        fitness_history.append(best_fitness)

        print(f"Generation {generation + 1}/{generations}: Best Fitness = {best_fitness:.4f}")
        print(f"  Best Params: lr = {global_best['params']['lr']:.4f}, hidden = {global_best['params']['hidden']}, dropout = {global_best['params']['dropout']:.2f}, epoch = {global_best['params']['epoch']}")

    return global_best['params'], fitness_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.03, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=350, help='Number of epochs to train.')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

    # args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    best_params, fitness_history = gps_algorithm(population_size=10, generations=20, adj=adj, features=features, labels=labels, idx_train=idx_train, idx_val=idx_val)
    print("Best hyperparameters found:", best_params)
    
    print(fitness_history)  

    # 绘制适应度历史图
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_history, label='Max Fitness (Accuracy)')
    plt.xlabel('Generation')
    plt.ylabel('Max Fitness (Accuracy)')
    plt.title('GPSO Fitness over Generations')
    plt.legend()
    plt.show()

    # 使用最佳超参数运行模型
    model = GCN(nfeat=features.shape[1], nhid=best_params['hidden'], nclass=labels.max().item() + 1, dropout=best_params['dropout'], epoch=best_params['epoch'])
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=5e-4)

    # 训练时间
    t_total = time.time()

    for epoch in range(best_params['epoch']):
        train(epoch)
    # train(best_params['epoch'])


    print("Optimization Finished!")

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    test()     

    # model.eval()
    # output = model(features, adj)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # acc_test = accuracy(output[idx_test], labels[idx_test])

    # # 打印混淆矩阵
    # confusion_matrix = calculate_confusion_matrix(output, labels)
    # print("Confusion Matrix:")
    # print(confusion_matrix)

    # # 输出评价指标
    # for class_index in range(output.shape[1]):
    #     tp_indices, fp_indices, tn_indices, fn_indices = find_tp_fp_tn_fn_indices(output, labels, class_index)
    #     tp = torch.sum(tp_indices).item()
    #     fp = torch.sum(fp_indices).item()
    #     tn = torch.sum(tn_indices).item()
    #     fn = torch.sum(fn_indices).item()

    #     precision_class = tp / (tp + fp) if tp + fp > 0 else 0
    #     recall_class = tp / (tp + fn) if tp + fn > 0 else 0
    #     f1_score_class = 2 * precision_class * recall_class / (precision_class + recall_class) if precision_class + recall_class > 0 else 0

    #     print(f"Class {class_index}: Precision: {precision_class:.3f}, Recall: {recall_class:.3f}, F1-Score: {f1_score_class:.3f}")

    # print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))


