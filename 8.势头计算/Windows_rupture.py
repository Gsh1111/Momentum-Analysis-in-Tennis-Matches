import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
import seaborn as sns
# 读取Excel表格
df = pd.read_excel('2.xlsx')
time = df[['elapsed_time']].values
signal_Mg_1 = df[['Mg_1']].values
signal_Mg_2 = df[['Mg_2']].values

df['set_change'] = df['set_no'].ne(df['set_no'].shift())
df['game_change'] = df['game_no'].ne(df['game_no'].shift())

# 找出 'p1_TPS' 为 1 的所有横坐标,找出 'p2_TPS' 为 1 的所有横坐标
p1_tps_time = df.loc[df['p1_TPS'] == 1, 'elapsed_time'].tolist()
p2_tps_time = df.loc[df['p2_TPS'] == 1, 'elapsed_time'].tolist()

# 创建子图
fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=300)  # Use 2 rows for both signals
sns.scatterplot(x=p1_tps_time, y=df.loc[df['p1_TPS'] == 1, 'Mg_1'], color='#579AC3', s=120, marker='o', label='p1_TPS', ax=axes[0])

# Analyze Mg_1
result_Mg_1 = rpt.Window(width=18, model="l2").fit(signal_Mg_1).predict(pen=5)
axes[0].plot(time.flatten(), signal_Mg_1.flatten(), lw=2, label='Momentum_1', color='#E6756E')

# 绘制检测到的变化点位置
for bkp in result_Mg_1:
    if bkp < len(time.flatten()):  # 确保 bkp 在索引范围内
        axes[0].axvline(x=time.flatten()[bkp], color="#1F79BC", linestyle='-.', label='change points')

# 在 set 变化的位置画一条虚线和更改背景色
set_change_times_Mg_1 = df[df['set_change']]['elapsed_time']

for j in range(0, len(set_change_times_Mg_1), 2):
    start_time = set_change_times_Mg_1.iloc[j]
    end_time = set_change_times_Mg_1.iloc[j + 1] if j + 1 < len(set_change_times_Mg_1) else df['elapsed_time'].max()

    axes[0].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
    axes[0].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

    color = 'white' if (j + 1) % 2 == 0 else '#F0F0F0'  # 每两个区域之间交替使用浅灰色和白色
    axes[0].axvspan(start_time, end_time, facecolor=color, alpha=0.3)

axes[0].set_title("Momentum of Player 1")

# Analyze Mg_2
result_Mg_2 = rpt.Window(width=12, model="l2").fit(signal_Mg_2).predict(pen=1)
axes[1].plot(time.flatten(), signal_Mg_2.flatten(), lw=2, label='Momentum_2', color="#1F79BC")
sns.scatterplot(x=p2_tps_time, y=df.loc[df['p2_TPS'] == 1, 'Mg_2'], color='#E6756E', s=120, marker='o', label='p2_TPS', ax=axes[1])

# 绘制检测到的变化点位置
for bkp in result_Mg_2:
    if bkp < len(time.flatten()):  # 确保 bkp 在索引范围内
        axes[1].axvline(x=time.flatten()[bkp], color='#E6756E', linestyle='-.', label='change points')

# 在 set 变化的位置画一条虚线和更改背景色
set_change_times_Mg_2 = df[df['set_change']]['elapsed_time']

for j in range(0, len(set_change_times_Mg_2), 2):
    start_time = set_change_times_Mg_2.iloc[j]
    end_time = set_change_times_Mg_2.iloc[j + 1] if j + 1 < len(set_change_times_Mg_2) else df['elapsed_time'].max()

    axes[1].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
    axes[1].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

    color = 'white' if (j + 1) % 2 == 0 else '#F0F0F0'  # 每两个区域之间交替使用浅灰色和白色
    axes[1].axvspan(start_time, end_time, facecolor=color, alpha=0.3)

axes[1].set_title("Momentum of Player 2")
axes[1].set_ylim(-3,4.5)


# 创建一个单独的图例
# 创建两个图例元素，一个用于折线，一个用于标记
line_legend = plt.Line2D([0], [0], color='#E6756E', lw=2, label='Momentum 1')
marker_legend = plt.Line2D([0], [0], marker='o', color='w', label='Three Points P1', markerfacecolor="#1F79BC", markersize=10)
axes[0].legend(handles=[line_legend, marker_legend], loc='upper left')

line_legend = plt.Line2D([0], [0], color='#E6756E', lw=2, label='Momentum 2')
marker_legend = plt.Line2D([0], [0], marker='o', color='w', label='Three Points P2', markerfacecolor='#E6756E', markersize=10)
axes[1].legend(handles=[line_legend, marker_legend], loc='upper left')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
