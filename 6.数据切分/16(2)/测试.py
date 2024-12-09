import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = '1601.xlsx'
df = pd.read_excel(path)

# 筛选需要的列
selected_columns = ['elapsed_time', 'set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_points_won', 'p2_points_won']

# 转换elapsed_time为timedelta类型
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])

# 将 timedelta 转换为分钟
df['elapsed_time'] = df['elapsed_time'].dt.total_seconds() / 60
print(df['elapsed_time'])

group = df[selected_columns].copy()

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
df_grouped = pd.concat(processed_dataframes)

# 设置图形大小
plt.figure(figsize=(14, 8), dpi=500)


# 绘制Player 1 Sets和Player 2 Sets曲线
plt.subplot(3, 1, 1)
sns.lineplot(x='elapsed_time', y='p1_sets', data=df_grouped, label='sets won by player 1')
sns.lineplot(x='elapsed_time', y='p2_sets', data=df_grouped, label='sets won by player 2')
plt.title('time series of sets')
plt.xlabel('Elapsed Time (min)')
plt.ylabel('Count')
plt.ylim(-0.2, 2.3)
plt.legend(loc="upper left")

# 绘制Player 1 Games和Player 2 Games曲线
plt.subplot(3, 1, 2)
sns.lineplot(x='elapsed_time', y='p1_games', data=df_grouped, label='games won by player 1')
sns.lineplot(x='elapsed_time', y='p2_games', data=df_grouped, label='games won by player 2')
plt.title('time series of games')
plt.xlabel('Elapsed Time (min)')
plt.ylabel('Count')
plt.ylim(-0.5, 8)
plt.legend(loc="upper left")

# 绘制Player 1 Points Won和Player 2 Points Won曲线
plt.subplot(3, 1, 3)
sns.lineplot(x='elapsed_time', y='p1_points', data=df_grouped, label='winning points of player 1')
sns.lineplot(x='elapsed_time', y='p2_points', data=df_grouped, label='winning points of player 2')
plt.title('time series of points')
plt.xlabel('Elapsed Time (min)')
plt.ylabel('Count')
plt.ylim(-1, 18)
plt.legend(loc="upper left")

# 调整子图之间的间隔
plt.tight_layout()

# 显示图形
plt.show()


# 读取Excel表格
df = pd.read_excel(path)  # 替换为你的Excel文件路径

# 转换elapsed_time为timedelta类型
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])

# 将 timedelta 转换为分钟
df['elapsed_time'] = df['elapsed_time'].dt.total_seconds() / 60


# 设置权重
wu = 0.8
wd = 0.2
wa = 0.2
ww = 0.8
ws_1_win = 1
ws_1_lose = 1.16
ws_2_win = 1
ws_2_lose = 1.16

# 设置权重
weight_net_pt = 0.5
weight_net_pt_won = 0.25

# weight_p1_break_pt = 0.8
# weight_p2_break_pt = 0.8
weight_break_pt_won = 1.5
weight_break_pt_missed = 0.6


# 计算 S_loss 和 S_gain 以及 S1
df['S1_gain'] = (wa * df['p1_ace'] + ww * df['p1_winner']) * ws_1_win + weight_break_pt_won * df['p1_break_pt_won'] + weight_net_pt_won * df['p1_net_pt_won']
df['S1_loss'] = (wu * df['p1_unf_err'] + wd * df['p1_double_fault']) * ws_1_lose + weight_break_pt_missed * df['p1_break_pt_missed']
df['S1'] = df['S1_gain'] - df['S1_loss']
df['S1_sum'] = df['S1'].cumsum()

# 计算 S_loss 和 S_gain 以及 S2
df['S2_gain'] = (wa * df['p2_ace'] + ww * df['p2_winner']) * ws_2_win + weight_break_pt_won * df['p2_break_pt_won'] + weight_net_pt_won * df['p2_net_pt_won']
df['S2_loss'] = (wu * df['p2_unf_err'] + wd * df['p2_double_fault']) * ws_2_lose + weight_break_pt_missed * df['p2_break_pt_missed']
df['S2'] = df['S2_gain'] - df['S2_loss']
df['S2_sum'] = df['S2'].cumsum()


# 添加标记列，检查 'game_no' 是否发生变化

df['set_change'] = df['set_no'].ne(df['set_no'].shift())
df['game_change'] = df['game_no'].ne(df['game_no'].shift())

# 在 set 变化的位置画一条虚线和更改背景色
set_change_times = df[df['set_change']]['elapsed_time']

# 使用 seaborn 画图
plt.figure(figsize=(10, 6),dpi=300)
sns.lineplot(x='elapsed_time', y='S1_sum', label='S1_Sum', data=df)
sns.lineplot(x='elapsed_time', y='S2_sum', color='#D54A3B', label='S2_Sum', data=df)

# 在 game_change 为 True 的点上绘制特殊标记
plt.scatter(df[df['game_change']]['elapsed_time'], df[df['game_change']]['S1_sum'], marker='o', color='#E6756E', label='Game Change (S1)')
plt.scatter(df[df['game_change']]['elapsed_time'], df[df['game_change']]['S2_sum'], marker='o', color='#579AC3', label='Game Change (S2)')


# 设置图表标题和标签
plt.title('Line Plot of S1 and S2 Cumulative Sum')
plt.xlabel('Elapsed Time (min)')
plt.ylabel('Cumulative Sum')

# 在 set 变化的位置画一条虚线
for i in range(0, len(set_change_times), 2):
    start_time = set_change_times.iloc[i]
    end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df['elapsed_time'].max()

    plt.axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
    plt.axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

    color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
    plt.axvspan(start_time, end_time, facecolor=color, alpha=0.3)

# 显示图表
plt.legend()
plt.show()