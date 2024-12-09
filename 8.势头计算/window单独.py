import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt

# 读取Excel表格
df = pd.read_excel('2.xlsx')
time = df[['elapsed_time']].values
signal = df[['sum_diff']].values

df['set_change'] = df['set_no'].ne(df['set_no'].shift())
df['game_change'] = df['game_no'].ne(df['game_no'].shift())

# 算法列表
# algorithms = ["Pelt", "Window", "Binseg", "BottomUp"]
algorithms = ["Window"]
# 创建子图
fig, ax = plt.subplots(figsize=(12, 3 * len(algorithms)), dpi=300)

for i, algorithm in enumerate(algorithms):
    # 运行算法
    if algorithm == "Pelt":
        result = rpt.Pelt(model="rbf").fit(signal).predict(pen=2)
    elif algorithm == "Window":
        result = rpt.Window(width=10, model="l2").fit(signal).predict(pen=2)
    elif algorithm == "Binseg":
        result = rpt.Binseg(model="l2").fit(signal).predict(pen=10)
    elif algorithm == "BottomUp":
        result = rpt.BottomUp(model="l2").fit(signal).predict(pen=10)
    elif algorithm == "Dynp":
        result = rpt.Dynp(model="l2").fit_predict(signal, n_bkps=10)

    # 在子图中绘制时间序列数据
    ax.plot(time.flatten(), signal.flatten(), lw=2, label='sum_diff', color='#E6756E')

    # 绘制检测到的变化点位置
    for bkp in result:
        if bkp < len(time.flatten()):  # 确保 bkp 在索引范围内
            ax.axvline(x=time.flatten()[bkp], color="#1F79BC", linestyle='-.', lw=2.8, label='change points')

    # 在 set 变化的位置画一条虚线和更改背景色
    set_change_times = df[df['set_change']]['elapsed_time']

    for j in range(0, len(set_change_times), 2):
        start_time = set_change_times.iloc[j]
        end_time = set_change_times.iloc[j + 1] if j + 1 < len(set_change_times) else df['elapsed_time'].max()

        ax.axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
        ax.axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

        color = 'white' if (j + 1) % 2 == 0 else '#F0F0F0'  # 每两个区域之间交替使用浅灰色和白色
        ax.axvspan(start_time, end_time, facecolor=color, alpha=0.3)

    ax.set_title(r'Detection of change points by $\mathbf{Window}$',loc='center',fontsize=14)

# 创建一个单独的图例
# 创建两个图例元素，一个用于折线，一个用于标记
line_legend = plt.Line2D([0], [0], color='#E6756E', lw=1.5, label='Performance_diff', markersize=6)
marker_legend = plt.Line2D([0], [0], color="#1F79BC", label='Change Points', markerfacecolor="#1F79BC", markersize=6)
fig.legend(handles=[line_legend, marker_legend], loc='upper left', bbox_to_anchor=(0.03, 1.005), ncol=2)
# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
