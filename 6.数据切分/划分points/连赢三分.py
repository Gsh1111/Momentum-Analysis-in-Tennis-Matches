import pandas as pd

# 读取已经处理后的 Excel 文件
file_path_updated = 'processed_data_updated.xlsx'
df = pd.read_excel(file_path_updated)

# 添加新的列来标记连续得分的位置
df['p1_TPS'] = 0
df['p2_TPS'] = 0

# 找到每局内连续得分的位置并标记
for _, group in df.groupby(['match_id', 'set_no', 'game_no']):
    # 遍历每局的数据
    for idx in range(2, len(group)):
        if (group.at[group.index[idx], 'point_victor'] == 1 and
            group.at[group.index[idx - 1], 'point_victor'] == 1 and
            group.at[group.index[idx - 2], 'point_victor'] == 1):
            df.at[group.index[idx], 'p1_TPS'] = 1

        if (group.at[group.index[idx], 'point_victor'] == 2 and
            group.at[group.index[idx - 1], 'point_victor'] == 2 and
            group.at[group.index[idx - 2], 'point_victor'] == 2):
            df.at[group.index[idx], 'p2_TPS'] = 1

# 保存更新后的数据集
# df.to_excel('TPSs_per_game.xlsx', index=False)

from scipy.stats import chi2_contingency


# 创建一个交叉表用于卡方检验
cross_table = pd.crosstab(df['game_victor'], df['p1_TPS'])

# 执行卡方检验
chi2, p, _, _ = chi2_contingency(cross_table)

# 输出卡方值和p值
print(f"Chi-squared value: {chi2}")
print(f"P-value: {p}")

# 判断显著性水平
alpha = 0.05
if p < alpha:
    print("在显著性水平0.05下，拒绝原假设，即两个变量之间存在关联性。")
else:
    print("在显著性水平0.05下，接受原假设，即两个变量之间不存在关联性。")


