import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_excel('2.xlsx')
# print(df.columns)
# 选择特征列
# features = df.drop(['p1_games', 'p2_games','elapsed_time', 'S1_gain', 'S1_loss', 'S1', 'S1_sum_new', 'S2_gain', 'S2_loss', 'S2',
#        'p1_points', 'p2_points','set_no','p1_sets', 'p2_sets','p1_points_won', 'p2_points_won', 'point_no','S2_sum_new','S1_sum', 'S2_sum','Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'sum_diff', 'Mg_1', 'Mg_2'], axis=1)

features = df.drop(['set_no', 'p1_sets', 'p2_sets', 'Mg_2','elapsed_time', 'p1_points_won', 'p2_points_won', 'point_no', 'S1_gain', 'S1_loss', 'S1', 'S2_gain', 'S2_loss', 'S2','S1_sum_new', 'S2_sum_new', 'S1_sum', 'S2_sum','Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'sum_diff'], axis=1)
# features = df.drop(['set_no',  'p1_sets', 'p2_sets','elapsed_time', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'sum_diff', 'S1_gain', 'S1_loss', 'S1', 'S2_gain', 'S2_loss', 'S2','S1_sum_new', 'S2_sum_new', 'S1_sum', 'S2_sum'],axis=1)

# 选择目标变量
target = df['sum_diff']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 定义超参数搜索范围
param_grid = {
    'n_estimators': [300],
    'max_depth': [None],
    'min_samples_leaf': [1]
}
# 创建MinMaxScaler对象
scaler = MinMaxScaler()

# 对训练集进行归一化
X_train = scaler.fit_transform(X_train)

# 对测试集进行归一化
X_test= scaler.transform(X_test)

# 初始化随机森林回归模型
rf_model = RandomForestRegressor(random_state=42)
# 使用GridSearchCV进行超参数搜索
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数
# print("Best Parameters:", grid_search.best_params_)

# 预测测试集
y_pred = grid_search.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

# 提取特征重要性
feature_importances = grid_search.best_estimator_.feature_importances_

# 将特征重要性排序
sorted_idx = feature_importances.argsort()

# 取特征
features = features.columns[sorted_idx]
importances = feature_importances[sorted_idx]

# print(features, importances)

# 创建DataFrame
df_top18 = pd.DataFrame({'Feature': features, 'Importance': importances})

# 按重要性降序排列
df_top18 = df_top18.sort_values(by='Importance', ascending=False)

df_top18.to_excel('top-44.xlsx')

# # 设置图表大小
# plt.figure(figsize=(12, 10))
#
# # 绘制水平条形图
# bars = plt.barh(range(len(top18_importances)), top18_importances)
#
# # 添加数据标签
# for bar, label in zip(bars, top18_importances):
#     plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' {label:.3f}',
#              va='center', ha='left', fontsize=10, color='black')
#
# # 设置y轴刻度标签
# plt.yticks(range(len(top18_importances)), top18_features)
# plt.xlabel('Feature Importance')
# plt.title('Top 18 Random Forest Feature Importance')
# # plt.grid(axis='x', linestyle='--', alpha=0.6)
# plt.tick_params(axis='y', which='both', left=False)
# plt.legend(['Feature Importance'], loc='lower right',fontsize=16)
# plt.xlim(0,0.317)
# plt.ylim(-0.55,17.55)
# plt.tight_layout()
# # 绘制水平虚线
# for i in range(len(top18_importances) - 1):
#     plt.hlines(i + 0.5, 0, 0.32, colors='gray', linestyles='dashed', alpha=0.4)
#
# plt.show()
