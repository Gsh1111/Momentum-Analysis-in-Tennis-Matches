import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 提供的数据
data = np.array([
    [0.857142857, 0.142857143],
    [0.178752108, 0.821247892]
])

# 定义标签和位置
categories = ['Win', 'Lose']
labels = ['Server is p1', 'Server is p2']

# 将数据转换为DataFrame格式
df = pd.DataFrame(data, columns=categories, index=labels)
bar_width = 0.35

# 创建堆叠柱状图

fig, ax = plt.subplots(figsize=(6, 2.5),dpi=300)
color_rgb = [31/255, 119/255, 180/255]
width = 0.25

bar1 = sns.barplot(data=df, width=bar_width, x=df.index, y='Win', color=color_rgb,
                   label='Win', alpha=0.7, dodge=width,
                   edgecolor='black', ls='-', linewidth=1)
bar2 = sns.barplot(data=df, width=bar_width, x=df.index, y='Lose', color='#D54A3B',
                   label='Lose', alpha=0.7, bottom=df['Win'], dodge=width,
                   edgecolor='black', ls='-', linewidth=1)


plt.ylim(0,1.0)
plt.xlim(-0.4,1.4)

# 添加虚线
for i, label in enumerate(labels):
    ax.vlines(i - 0.3, 0, plt.ylim()[1], colors='gray', linestyles='dashed', linewidth=0.8)
    ax.vlines(i + 0.3, 0, plt.ylim()[1], colors='gray', linestyles='dashed', linewidth=0.8)

plt.axvspan(-0.4, -0.3, facecolor='#F0F0F0', alpha=0.3)
plt.axvspan(0.3, 0.7, facecolor='#F0F0F0', alpha=0.3)
plt.axvspan(1.3, 1.4, facecolor='#F0F0F0', alpha=0.3)

h1 = bar1.patches[0].get_height()
h2 = bar1.patches[1].get_height()
# 在"Win"柱的顶部添加水平虚线

ax.axhline(y=h1, color='gray', linestyle='--', linewidth=0.8)
ax.axhline(y=h2, color='gray', linestyle='--', linewidth=0.8)


# 添加标签和标题
ax.set_xlabel('Server', fontsize=14)
ax.set_ylabel('Values', fontsize=14)
ax.set_title('Win rate of player 1', fontsize=16)

# 放置图例在中间下侧
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='center', fontsize=13)
plt.savefig('your_figure.png', dpi=300)
# 显示柱状图
plt.show()
