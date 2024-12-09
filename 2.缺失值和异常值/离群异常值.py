import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

def abnorm_detect(x1, label_index, detect_label):
    x = zscore(x1)  # 对数据进行标准化

    if detect_label == 1:  # 使用3σ法则检测异常值
        fig, ax = plt.subplots()
        index_ab_all = []

        for i in range(x.shape[1]):
            x_mean = np.mean(x[:, i])
            x_std = np.std(x[:, i])
            index_ab = np.where(np.abs(x[:, i] - x_mean) > 3 * x_std)[0]  # 异常值的索引
            index_ab_all.extend(index_ab)
            index_nor = np.setdiff1d(np.arange(x.shape[0]), index_ab)
            ax.scatter(i * np.ones(len(index_nor)), x[index_nor, i], c='b')
            ax.scatter(i * np.ones(len(index_ab)), x[index_ab, i], c='r')

        index_ab_unique = np.unique(index_ab_all)
        print("检测到的异常值索引为：")
        print(index_ab_unique)

        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(label_index)

        y = np.delete(x1, index_ab_unique, axis=0)

    elif detect_label == 2:  # 使用箱线图检测异常值
        fig, ax = plt.subplots()
        ax.boxplot(x)
        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(label_index)

        index_ab_all = []
        for i in range(x.shape[1]):
            Q = np.percentile(x[:, i], [25, 75])
            q1 = Q[0] - 1.5 * (Q[1] - Q[0])
            q2 = Q[1] + 1.5 * (Q[1] - Q[0])
            index_ab = np.where((x[:, i] < q1) | (x[:, i] > q2))[0]
            index_ab_all.extend(index_ab)

        index_ab_unique = np.unique(index_ab_all)
        print("检测到的异常值索引为：")
        print(index_ab_unique)

        y = np.delete(x1, index_ab_unique, axis=0)

    elif detect_label == 3:  # 使用z分数检测异常值
        fig, ax = plt.subplots()
        index_ab_all = []

        for i in range(x.shape[1]):
            index_ab = np.where(np.abs(x[:, i]) > 3)[0]  # 异常值的索引
            index_ab_all.extend(index_ab)
            index_nor = np.setdiff1d(np.arange(x.shape[0]), index_ab)
            ax.scatter(i * np.ones(len(index_nor)), x[index_nor, i], c='b')
            ax.scatter(i * np.ones(len(index_ab)), x[index_ab, i], c='r')

        index_ab_unique = np.unique(index_ab_all)
        print("检测到的异常值索引为：")
        print(index_ab_unique)

        ax.set_xticks(np.arange(x.shape[1]))
        ax.set_xticklabels(label_index)

        y = np.delete(x1, index_ab_unique, axis=0)

    elif detect_label == 4:  # 使用孤立森林检测异常值
        fig, ax = plt.subplots()
        index_ab_unique = []

        clf = IsolationForest(contamination=0.05)
        clf.fit(x)
        scores = clf.score_samples(x)
        threshold = np.percentile(scores, 5)
        index_ab_unique = np.where(scores < threshold)[0]

        print("检测到的异常值索引为：")
        print(index_ab_unique)

        ax.hist(scores, bins=50)
        ax.axvline(threshold, color='r', linestyle='-', label='Threshold')
        ax.legend()

        y = np.delete(x1, index_ab_unique, axis=0)

    return y, index_ab_unique

df = pd.read_excel('Wimbledon_featured_matches.xlsx')

columns_of_interest = ['set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'server', 'serve_no',
                     'point_victor', 'p1_points_won', 'p2_points_won', 'game_victor', 'set_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
                     'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won',
                     'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed',
                     'rally_count', ]

# 从 DataFrame 中提取感兴趣的列数据
x1 = df[columns_of_interest].values
detect_label = 4  # 选择异常检测方法，可以根据需要修改为 2, 3 或 4
# 调用异常检测函数
y, index_ab_unique = abnorm_detect(x1, columns_of_interest, detect_label)

# 打印处理后的数据和异常值索引
# print("处理后的数据：")
# print(y)

print("\n检测到的异常值索引：")
print(index_ab_unique)

