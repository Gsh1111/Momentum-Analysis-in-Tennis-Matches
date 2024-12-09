import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# # 读取Excel表格
# df = pd.read_excel('1701.xlsx')
# # 提取指定的列
# selected_columns = [
#     'match_id', 'player1', 'player2', 'elapsed_time', 'set_no', 'game_no',
#     'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_points_won', 'p2_points_won'
# ]
# data = df[selected_columns]
# data.to_excel('可视化.xlsx')



# 读取Excel表格
df = pd.read_excel('可视化.xlsx')  # 替换为你的Excel文件路径

# 筛选需要的列
selected_columns = ['elapsed_time', 'set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'p1_points_won', 'p2_points_won']

# # 转换elapsed_time为timedelta类型
# df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])
#
# # 将 timedelta 转换为分钟
# df['elapsed_time'] = df['elapsed_time'].dt.total_seconds() / 60
# # print(df['elapsed_time'])
# df.to_excel('可视化.xlsx')
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



# # 创建子图
# fig, axes = plt.subplots(3, 1, figsize=(10,6), dpi=300)
#
# # 绘制Player 1 Sets和Player 2 Sets曲线
# sns.lineplot(x='elapsed_time', y='p1_sets', data=df_grouped, label='sets won by player 1', ax=axes[0], linewidth=2.5)
# sns.lineplot(x='elapsed_time', y='p2_sets', data=df_grouped, color ='#d56e6d', label='sets won by player 2', ax=axes[0], linewidth=2.5)
# axes[0].set_title('Time Series of $\mathbf{Sets}$')
# axes[0].set_xlabel('Elapsed Time (min)')
# axes[0].set_ylabel('Count')
# axes[0].set_xlim(-15, 310)
# axes[0].set_ylim(-0.2, 3.35)
# axes[0].legend(loc="upper left", ncol=1, fontsize=9)
#
# # 绘制Player 1 Games和Player 2 Games曲线
# sns.lineplot(x='elapsed_time', y='p1_games', data=df_grouped, label='games won by player 1', ax=axes[1], linewidth=2)
# sns.lineplot(x='elapsed_time', y='p2_games', data=df_grouped, color ='#d56e6d', label='games won by player 2', ax=axes[1], linewidth=2)
# axes[1].set_title('Time Series of $\mathbf{Games}$')
# axes[1].set_xlabel('Elapsed Time (min)')
# axes[1].set_ylabel('Count')
# axes[1].set_xlim(-15, 310)
# axes[1].set_ylim(-0.5, 10)
# axes[1].legend(loc="upper left", ncol=1, fontsize=9)
#
# # 绘制Player 1 Points Won和Player 2 Points Won曲线
# sns.lineplot(x='elapsed_time', y='p1_points', data=df_grouped, label='points won by player 1', ax=axes[2], linewidth=1.5)
# sns.lineplot(x='elapsed_time', y='p2_points', color ='#d56e6d', data=df_grouped, label='points won by player 2', ax=axes[2], linewidth=1.5)
# axes[2].set_title('Time Series of $\mathbf{Points}$')
# axes[2].set_xlabel('Elapsed Time (min)')
# axes[2].set_ylabel('Count')
# axes[2].set_xlim(-15, 310)
# axes[2].set_ylim(-1, 18)
# axes[2].legend(loc="upper left", ncol=1, fontsize=9)
#
#
#
# # 在每个子图中加上标记列
# df_grouped['set_change'] = df_grouped['set_no'].ne(df_grouped['set_no'].shift())
# set_change_times = df_grouped[df_grouped['set_change']]['elapsed_time']
#
# for i in range(0, len(set_change_times), 2):
#     start_time = set_change_times.iloc[i]
#     end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df_grouped['elapsed_time'].max()
#
#     axes[0].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[0].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)
#
#     color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
#     axes[0].axvspan(start_time, end_time, facecolor=color, alpha=0.3)
#
#     axes[1].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[1].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[1].axvspan(start_time, end_time, facecolor=color, alpha=0.3)
#
#     axes[2].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[2].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[2].axvspan(start_time, end_time, facecolor=color, alpha=0.3)
#
#
#
# plt.tight_layout()
# plt.show()


