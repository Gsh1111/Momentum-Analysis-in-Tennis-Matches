import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置风格
sns.set(style='white')

# 读取 Excel 文件
file_path = '相关性分析.xlsx'
df = pd.read_excel(file_path)
# 假设 df 是包含 'game_flag', 'set_flag', 'p1_ace', 'p2_ace', ... 的数据框
heatmap_data = df[['p1_ACE', 'p1_W', 'p1_DF', 'p1_UE','p1_TPS']]

# 计算相关性矩阵
df_corr = heatmap_data.corr()

# # 设置一个“上三角行”蒙版
# mask = np.zeros_like(correlation_matrix)
# mask[np.triu_indices_from(mask)] = True

# 绘制热力图
fig, ax = plt.subplots(figsize=(10, 8))
# mask
mask = np.triu(np.ones_like(df_corr, dtype=bool))
mask = mask[1:, :-1]
corr = df_corr.iloc[1:, :-1].copy()
# 颜色映射
cmap = sns.diverging_palette(220, 10, 75, 50, as_cmap=True, )
# 生成半边热图
plt.figure(figsize=(10, 8), dpi = 300)
# 绘制热力图
sns.heatmap(corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            linewidths=5,  vmin=-0.8, vmax=0.8,
            cmap=cmap,
            annot_kws={'size':12},
            cbar_kws={"shrink": .8}, square=True)
# 调整标签字体大小
# 刻度
yticks = [i.upper() for i in corr.index]
xticks = [i.upper() for i in corr.columns]
plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
plt.xticks(plt.xticks()[0], labels=xticks)
plt.tick_params(axis='both', which='both', labelsize=10)
plt.title('Correlation Heatmap', loc='center', fontsize = 18)
plt.show()

