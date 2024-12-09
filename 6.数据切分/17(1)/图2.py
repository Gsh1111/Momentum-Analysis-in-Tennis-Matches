import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取Excel表格
df = pd.read_excel('1701-pro.xlsx')  # 替换为你的Excel文件路径

# 转换elapsed_time为timedelta类型
# df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])

# 将 timedelta 转换为分钟
# df['elapsed_time'] = df['elapsed_time'].dt.total_seconds() / 60


# 设置权重
wu = 0.8
wd = 0.2
wa = 0.2
ww = 0.8
ws_1_win = 1
ws_1_lose = 1.16
ws_2_win = 1
ws_2_lose = 1.16
weight_net_pt = 0.5
weight_net_pt_won = 0
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

# # 计算两个数据列的差值
# df['diff'] = df['S1_sum'] - df['S2_sum']
# print(df['diff'])
# # 找到差值为0的点
# intersection_points = df[(df['diff'] <= 0.5) & (df['diff'] >= -0.5)]
#
#
# # 打印交点的横坐标
# print("交点的横坐标：", intersection_points['elapsed_time'])



# 使用 seaborn 画图
plt.figure(figsize=(10, 6),dpi=300)
sns.lineplot(x='elapsed_time', y='S1_sum', label='Performance of Player 1', data=df, errorbar=None)
sns.lineplot(x='elapsed_time', y='S2_sum', color='#D54A3B', label='Performance of Player 2', data=df, errorbar=None)

# 在 game_change 为 True 的点上绘制特殊标记
plt.scatter(df[df['game_change']]['elapsed_time'], df[df['game_change']]['S1_sum'], marker='o', color='#E6756E', label='Game Change of Player 1')
plt.scatter(df[df['game_change']]['elapsed_time'], df[df['game_change']]['S2_sum'], marker='o', color='#579AC3', label='Game Change of Player 2')


# 设置图表标题和标签
plt.title('Performance of Player 1 and Player 2 by Time',fontsize= 14)
plt.xlabel('Elapsed Time (min)',fontsize= 14)
plt.ylabel('Cumulative Sum',fontsize= 14)

# 添加标记列，检查 'game_no' 是否发生变化
df['set_change'] = df['set_no'].ne(df['set_no'].shift())
df['game_change'] = df['game_no'].ne(df['game_no'].shift())
set_change_times = df[df['set_change']]['elapsed_time']
# 在 set 变化的位置画一条虚线
for i in range(0, len(set_change_times), 2):
    start_time = set_change_times.iloc[i]
    end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df['elapsed_time'].max()

    plt.axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
    plt.axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

    color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
    plt.axvspan(start_time, end_time, facecolor=color, alpha=0.3)

# 显示图表
plt.legend(ncol=2,fontsize=11)
plt.show()


#
# w_ACE = 0.3
# w_w = 0.32
# w_TPS_1 = 0.38
# w_TPS_2 = 0.42
# w_DF = 0.55
# w_UE = 0.45
# # ws_1_win = 1.2
# # ws_1_lose = 1
# # ws_2_win = 1
# # ws_2_lose = 1.2
# # weight_p1_break_pt = 0.8
# # weight_p2_break_pt = 0.8
# weight_break_pt_won = 1.5
# weight_break_pt_missed = 0.6
# lambda_value = 0.2
# # 1.时间间隔
# df['time_interval'] = df['elapsed_time'].diff().fillna(0)
#
# # 2.三连胜
# df['p1_TPS'] = 0
# df['p2_TPS'] = 0
# for _, group in df.groupby(['match_id', 'set_no', 'game_no']):
#     for idx in range(2, len(group)):
#         if (group.at[group.index[idx], 'point_victor'] == 1 and
#             group.at[group.index[idx - 1], 'point_victor'] == 1 and
#             group.at[group.index[idx - 2], 'point_victor'] == 1):
#             df.at[group.index[idx], 'p1_TPS'] = 1
#
#         if (group.at[group.index[idx], 'point_victor'] == 2 and
#             group.at[group.index[idx - 1], 'point_victor'] == 2 and
#             group.at[group.index[idx - 2], 'point_victor'] == 2):
#             df.at[group.index[idx], 'p2_TPS'] = 1
#
# df['server'] = df['server'].replace(2,0)
#
# df['S1_gain'] =  (1+0.2 * df['server']) * (w_ACE * df['p1_ace'] + w_w * df['p1_winner']
#                  + weight_break_pt_won * df['p1_break_pt_won'] + df['p1_TPS'] * w_TPS_1)
# df['S1_loss'] =  (1-0.2 * df['server']) * (w_UE * df['p1_unf_err'] + w_DF * df['p1_double_fault']
#                  + weight_break_pt_missed * df['p1_break_pt_missed'] + df['p2_TPS'] * w_TPS_2)
#
# df['server'] = df['server'].replace(0,1)
# df['server'] = df['server'].replace(1,0)
# df['S2_gain'] =  (1+0.2 * df['server']) * (w_ACE * df['p2_ace'] + w_w * df['p2_winner']
#                  + weight_break_pt_won * df['p2_break_pt_won'] + df['p2_TPS'] * w_TPS_2)
# df['S2_loss'] =  (1-0.2 * df['server']) * (w_UE * df['p2_unf_err'] + w_DF * df['p2_double_fault']
#                  + weight_break_pt_missed * df['p2_break_pt_missed'] + df['p2_TPS'] * w_TPS_2)
#
# # 初始化 Mg 列表
# Mg_1 = [0]
# Mg_2 = [0]
# for i in range(1, len(df)):
#     delta_t = df['time_interval'].iloc[i]
#     f = 0.6 + 0.4 * np.exp(-lambda_value * delta_t)
#
#     delta_Mg_1 = 1.2 * df['S1_gain'].iloc[i] - 1 * df['S1_loss'].iloc[i]
#     delta_Mg_2 = 1.2 * df['S2_gain'].iloc[i] - 1 * df['S2_loss'].iloc[i]
#     Mg_1_i = f * Mg_1[i - 1] + delta_Mg_1
#     Mg_2_i = f * Mg_2[i - 1] + delta_Mg_2
#     Mg_1.append(Mg_1_i)
#     Mg_2.append(Mg_2_i)
#
# df['Mg_1'] = Mg_1
# df['Mg_2'] = Mg_2
#
#
# # 创建一个带有两个子图的图表
# fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharex=True, dpi=300)
#
# # 第一个子图：p1 和 p2 的累积得分
# sns.lineplot(x='elapsed_time', y='S1_sum', label='Performance of P1', data=df, ax=axes[0])
# sns.lineplot(x='elapsed_time', y='S2_sum', color='#D54A3B', label='Performance of P2', data=df, ax=axes[0])
#
# axes[0].set_title('Performance of Player 1 and Player 2 by Time [Match_id: 1305]')
# axes[0].set_xlabel('Elapsed Time (min)')
# axes[0].set_ylabel('Performance')
# axes[0].set_ylim(-10.8,5.5)
# axes[0].legend(loc='upper center')
#
# # 第二个子图：Mg_1 和 Mg_2 的累积得分
# sns.lineplot(x='elapsed_time', y='Mg_1', label='Momentum of P1', data=df, ax=axes[1])
# sns.lineplot(x='elapsed_time', y='Mg_2', label='Momentum of P2', color='#E6756E', data=df, ax=axes[1])
#
# # 添加 p1_TPS 和 p2_TPS 的散点图
# p1_tps_time = df.loc[df['p1_TPS'] == 1, 'elapsed_time'].tolist()
# p2_tps_time = df.loc[df['p2_TPS'] == 1, 'elapsed_time'].tolist()
# sns.scatterplot(x=p1_tps_time, y=df.loc[df['p1_TPS'] == 1, 'Mg_2'], color='#579AC3', marker='o', label='Three Points of P1', ax=axes[1])
# sns.scatterplot(x=p2_tps_time, y=df.loc[df['p2_TPS'] == 1, 'Mg_1'], color='#E6756E', marker='o', label='Three Points of P2', ax=axes[1])
#
# axes[1].set_title('Momentum of Player 1 and Player 2 by Time [Match_id: 1305]')
# axes[1].set_xlabel('Elapsed Time (min)')
# axes[1].set_ylabel('Momentum')
# axes[1].set_ylim(-3.4,6.3)
# axes[1].legend(loc='upper center',ncol=2)
#
# # 在 set 变化的位置画一条虚线
# for i in range(0, len(set_change_times), 2):
#     start_time = set_change_times.iloc[i]
#     end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df['elapsed_time'].max()
#
#     axes[0].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[0].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[1].axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
#     axes[1].axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)
#
#     color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
#     axes[0].axvspan(start_time, end_time, facecolor=color, alpha=0.3)
#     axes[1].axvspan(start_time, end_time, facecolor=color, alpha=0.3)
#
# plt.tight_layout()
# plt.show()
