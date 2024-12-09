import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置风格
sns.set(style='white')

# 读取 Excel 文件
file_path = 'Wimbledon_featured_matches.xlsx'
df = pd.read_excel(file_path)


# 三连胜
df['p1_TPS'] = 0
df['p2_TPS'] = 0
df.at[0, 'Three Points'] = 0
df.at[1, 'Three Points'] = 0
for _, group in df.groupby(['match_id', 'set_no', 'game_no']):
    for idx in range(2, len(group)):
        if (group.at[group.index[idx], 'point_victor'] == 1 and
            group.at[group.index[idx - 1], 'point_victor'] == 1 and
            group.at[group.index[idx - 2], 'point_victor'] == 1):
            df.at[group.index[idx], 'Three Points'] = 1
        else:
            df.at[group.index[idx], 'Three Points'] = 0

        if (group.at[group.index[idx], 'point_victor'] == 2 and
            group.at[group.index[idx - 1], 'point_victor'] == 2 and
            group.at[group.index[idx - 2], 'point_victor'] == 2):
            df.at[group.index[idx], 'p2_TPS'] = 1

# 假设 df 是包含 'game_flag', 'set_flag', 'p1_ace', 'p2_ace', ... 的数据框
heatmap_data = df[['Ace', 'Winner', 'Unf Err','Three Points' ,'Double Fault']]
# print(df['Three Points'])
sorted_columns = sorted(df.columns, key=len, reverse=False)
df = df[sorted_columns]
# 计算相关性矩阵
df_corr = heatmap_data.corr()

# 绘制热力图
fig, ax = plt.subplots(figsize=(10, 8))
# mask
mask = np.triu(np.ones_like(df_corr, dtype=bool))
mask = mask[1:, :-1]
corr = df_corr.iloc[1:, :-1].copy()
# 颜色映射
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# 生成半边热图
plt.figure(figsize=(10, 6), dpi = 300)
# 绘制热力图
sns.heatmap(corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            linewidths=10,
            vmin=-0.6,
            vmax=0.6,
            cmap=cmap,
            annot_kws={'size':14},
            cbar_kws={"shrink": .8},
            square=True
            )
# 调整标签字体大小
# 刻度
yticks = [i.upper() for i in corr.index]
xticks = [i.upper() for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0,fontweight='bold')
plt.xticks(plt.xticks()[0], labels=xticks,fontweight='bold')
plt.tick_params(axis='both', which='both', labelsize=10)

plt.show()

