import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve
df = pd.read_excel('1701.xlsx')

group = df.copy()

# 存储处理后的数据框
processed_dataframes = []
# 获取每组数据的最后一个值
last_values = group.groupby(['set_no', 'game_no']).tail(1)
# 在 list 的最前面插入0
list_last_p1 = np.insert(last_values['p1_points_won'].values, 0, 0)
list_last_p2 = np.insert(last_values['p2_points_won'].values, 0, 0)

# 计算每组内的相对得分
df_grouped_pro = group.groupby(['set_no', 'game_no'])
i = 0
for (set_no, game_no), group_pro in df_grouped_pro:
    # 使用 transform 函数进行相对得分计算
    group_pro['p1_points'] = group_pro['p1_points_won'] - list_last_p1[i]
    group_pro['p2_points'] = group_pro['p2_points_won'] - list_last_p2[i]
    i += 1
    # 将处理后的数据添加到列表中
    processed_dataframes.append(group_pro)

# 将处理后的数据合并到原始数据框
df = pd.concat(processed_dataframes)

# point_victor 列作为目标变量
df['point_victor'] = df['point_victor'].replace(2, 0)
target_variable = df['point_victor']

# 删除 match_id、player1、player2、p1_score 和 p2_score 列
columns_to_drop = ['match_id', 'player1', 'player2', 'p1_score', 'p2_score']
df.drop(columns=columns_to_drop, inplace=True)

# # 1. elapsed_time 统一转化为按分钟计数
# df['elapsed_time'] = df['elapsed_time'] / 60
def convert_time_to_minutes(time_str):
    time_object = datetime.strptime(time_str, '%H:%M:%S')
    minutes = (time_object.hour * 60) + time_object.minute + (time_object.second / 60)
    return minutes

# 使用 apply 函数将转换函数应用于整个列
df['elapsed_time'] = df['elapsed_time'].apply(convert_time_to_minutes)

# 2. winner_shot_type F：正手1；B：反手2
df['winner_shot_type'] = df['winner_shot_type'].map({'F': 1, 'B': 2})

# 3. serve_depth CTL：1；NCTL：2
df['serve_depth'] = df['serve_depth'].map({'CTL': 1, 'NCTL': 2})

# 4. return_depth D：1；ND：2
df['return_depth'] = df['return_depth'].map({'D': 1, 'ND': 2})

# 5. serve_width 表示发球的方向编码
serve_width_mapping = {'B': 1, 'BC': 2, 'BW': 3, 'C': 4, 'W': 5}
df['serve_width'] = df['serve_width'].map(serve_width_mapping)

# 替换缺失值
df.fillna(0, inplace=True)

columns_to_add = ['rally_count', 'speed_mph', 'serve_width', 'serve_depth', 'return_depth',
                  'winner_shot_type', 'game_victor', 'set_victor', 'server', 'serve_no',
                  'elapsed_time', 'set_no', 'game_no', 'point_no']
# columns_to_add = ['winner_shot_type', 'game_victor', 'set_victor', 'server', 'serve_no',
#                   'elapsed_time', 'set_no', 'game_no', 'point_no']
from sklearn.preprocessing import MinMaxScaler


# columns_to_normalize 包含需要进行归一化的列
columns_to_normalize = ['elapsed_time', 'rally_count', 'speed_mph', 'p1_distance_run', 'p2_distance_run', 'point_no']

# 创建 MinMaxScaler 实例
scaler = MinMaxScaler()

# 对指定列进行归一化
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# 选择包含 'p1' 的列
p1_columns = df.columns[df.columns.str.contains('p1')].tolist()
p1_columns = p1_columns + columns_to_add

# 选择包含 'p2' 的列
p2_columns = df.columns[df.columns.str.contains('p2')].tolist()
p2_columns = p2_columns + columns_to_add
print(p1_columns)
print(p2_columns)
p1_data = df[p1_columns]
p2_data = df[p2_columns]

X_train, X_test, y_train, y_test = train_test_split(p1_data, target_variable, test_size=0.2, random_state=42)

# 对于分类问题
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 使用模型对测试数据集进行预测
y_pred = clf.predict(X_test)
y_scores = clf.predict_proba(X_test)[:, 1]  # 获取概率分数

# 计算各种评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
#
# print(y_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_scores)
# roc_auc = roc_auc_score(y_test, y_scores)
#
# # 创建一个图表
# plt.figure(figsize=(8, 6), dpi=300)
#
# # 绘制 ROC 曲线
# sns.lineplot(x=fpr, y=tpr, color='#D54A3B', lw=2, label=f'ROC curve (AUC={roc_auc:.2f})')
#
# # 对角线上的虚线
# color_rgb = [31/255, 119/255, 180/255]
# plt.plot([0, 1], [0, 1], color=color_rgb, lw=2, linestyle='--', label='Random Guessing')
#
# plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# plt.grid(True)
# plt.xlabel('False Positive Rate', fontsize=14)
# plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
# plt.legend(loc="lower right", fontsize=12)
# plt.show()


# precision, recall, _ = precision_recall_curve(y_test, y_scores)
#
# plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
# plt.show()



# conf_matrix = confusion_matrix(y_test, y_pred)
# # 创建热图，并设置dpi参数
# plt.figure(figsize=(8, 6), dpi=300)
# sns.heatmap(conf_matrix,
#             annot=True,
#             fmt='d',
#             linewidths=0.2,
#             cmap='Blues',
#             cbar=True,
#             square=True,
#             annot_kws={"size": 16}
#             )
# plt.xlabel('Predicted Label', fontsize = 14)
# plt.ylabel('True Label', fontsize = 14)
# plt.title('Confusion Matrix', fontsize = 18)
# plt.show()



# 获取特征重要性
feature_importance = clf.feature_importances_

# 创建一个 DataFrame 以便进行可视化
importance_df = pd.DataFrame({'Feature': p1_data.columns, 'Importance': feature_importance})
# 将 Feature 列中的 '_' 替换为空格
importance_df['Feature'] = importance_df['Feature'].replace('_', ' ', regex=True).str.title()
# 排序并绘制特征重要性条形图
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 只选择前8个特征
top8_features = importance_df.head(8)
top8_features = top8_features[::-1]

# 排序并绘制前8个特征的重要性条形图

plt.figure(figsize=(10, 4),dpi = 300)
import colorsys

original_color = "#b53021"
original_rgb = tuple(int(original_color[i:i+2], 16) for i in (1, 3, 5))
h, l, s = colorsys.rgb_to_hls(original_rgb[0] / 255.0, original_rgb[1] / 255.0, original_rgb[2] / 255.0)
num_colors = 8
color_sequence = [original_color]
for i in range(num_colors):
    current_color = colorsys.hls_to_rgb(h+i*0.002, min(1.0, l + i * 0.06), min(1.0,s+i*0.015))
    current_color_hex = "#{:02X}{:02X}{:02X}".format(int(current_color[0] * 255), int(current_color[1] * 255), int(current_color[2] * 255))
    color_sequence.append(current_color_hex)
color_sequence = color_sequence[::-1]

ax = sns.barplot(x='Importance', y='Feature', data=top8_features, orient='h', palette=color_sequence, linewidth=0.4)
plt.ylabel('Feature', fontsize = 12)
plt.xlabel('Importance', fontsize = 12)
plt.title('Feature Importance - Top 8 ', fontsize = 16)
ax.set_xlim(0, 0.17)

for index, value in enumerate(top8_features['Importance']):
    ax.text(value, index, '   '+top8_features['Feature'].iloc[index]+f' - ({value:.3f})', ha='left', va='center', color='black', fontsize=11, weight='normal')

for index, value in enumerate(top8_features['Importance']):
    ax.axhline(index + 0.5, color='grey', linestyle='--', linewidth=0.5)
    if index % 2 == 0:
        rect = plt.Rectangle((0, index-0.5), 0.17, 1, fc='#F0F0F0', ec='none', zorder=-1)
        ax.add_patch(rect)

ax.set_yticklabels([])
plt.show()



#
# # # 对于回归问题
# # reg = RandomForestRegressor(random_state=42)
# # reg.fit(X_train, y_train)
#
# # # 对于回归问题
# # mse = mean_squared_error(y_test, reg.predict(X_test))
# # print(f'Mean Squared Error: {mse}')
#
# # from sklearn.model_selection import GridSearchCV
# #
# # param_grid = {
# #     'n_estimators': [50, 100, 200],
# #     'max_depth': [None, 10, 20],
# #     # 添加其他超参数...
# # }
# #
# # grid_search = GridSearchCV(clf, param_grid, cv=5)
# # grid_search.fit(X_train, y_train)
# #
# # best_params = grid_search.best_params_
# # best_accuracy = grid_search.best_score_
# #
# # print(f'Best Parameters: {best_params}')
# # print(f'Best Accuracy: {best_accuracy}')
