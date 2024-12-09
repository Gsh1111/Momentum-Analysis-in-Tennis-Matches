import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据
data = {
    'ACE': [0.74, 0.26],
    'WINNER': [0.78, 0.22],
    'CONSECUTIVE WINS': [0.95, 0.05],
    'DOUBLE FAULT': [0.31, 0.69],
    'UNF ERR': [0.42, 0.58],
    'CONSECUTIVE LOSES': [0.05, 0.95]
}
l = [0.48, 0.56, 0.90, -0.38, -0.38, -0.90]
df = pd.DataFrame(data, index=['Win', 'Lose'])

# 将索引转换为列
df.reset_index(inplace=True)

# 使用 seaborn 画柱状图
plt.figure(figsize=(12, 3.5),dpi = 300)

sns.barplot(data=df.melt(id_vars='index'), x='variable', y='value', hue='index', palette=["skyblue", '#E6756E'], edgecolor='black')

# 在每个组之间画一条竖直虚线
for i in range(1, len(df.columns) - 1):
    if i == 3:
        plt.axvline(x=i - 0.5, linestyle='-', color='gray', linewidth=2.8)
    else:
        plt.axvline(x=i - 0.5, linestyle='--', color='gray', linewidth=0.8)

plt.xlabel('Indicators',fontsize=14)
plt.ylabel('Proportion',fontsize=14)
# plt.legend(title='Outcome', loc='upper left', labels=['Win', 'Lose'])
# 在左边三个柱子背后添加灰色填充区域

plt.xlim(-0.5, len(df.columns) - 1.5)
plt.axvspan(-0.5, len(df.columns) - 1.5, facecolor='lightgray', alpha=0.05)
# 显示图例
# 获取图例的元素
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Win', markerfacecolor='skyblue', markersize=10),
                   plt.Line2D([0], [0], marker='o', color='w', label='Lose', markerfacecolor='#E6756E', markersize=10)]

plt.legend( loc='upper left', handles=legend_elements, fontsize=10, ncol=2)
plt.tight_layout()  # 自动调整布局
plt.show()

