import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = [ '1301.xlsx', '1302.xlsx', '1304.xlsx', '1305.xlsx', '1306.xlsx', '1307.xlsx', '1308.xlsx',
         '1309.xlsx', '1310.xlsx', '1311.xlsx', '1312.xlsx', '1313.xlsx', '1314.xlsx', '1315.xlsx', '1316.xlsx',
         '1401.xlsx', '1402.xlsx', '1403.xlsx', '1404.xlsx', '1405.xlsx', '1406.xlsx', '1407.xlsx', '1408.xlsx',
         '1501.xlsx', '1502.xlsx', '1503.xlsx', '1504.xlsx', '1601.xlsx', '1602.xlsx', '1701.xlsx']

# path = ['1303.xlsx']
# 设置权重
w_ACE = 0.3
w_w = 0.32
w_TPS_1 = 0.38
w_TPS_2 = 0.42
w_DF = 0.55
w_UE = 0.45
# ws_1_win = 1.2
# ws_1_lose = 1
# ws_2_win = 1
# ws_2_lose = 1.2
# weight_p1_break_pt = 0.8
# weight_p2_break_pt = 0.8
weight_break_pt_won = 1.5
weight_break_pt_missed = 0.6
lambda_value = 0.2


def Momentum(path):
    df = pd.read_excel(path)  # 替换为你的Excel文件路径
    # df['elapsed_time'] = pd.to_datetime(df['elapsed_time'], errors='coerce').dt.strftime('%H:%M:%S')
    # 转换elapsed_time为timedelta类型
    df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])

    # 将 timedelta 转换为分钟
    df['elapsed_time'] = df['elapsed_time'].dt.total_seconds() / 60

    # 1.时间间隔
    df['time_interval'] = df['elapsed_time'].diff().fillna(0)

    # 2.三连胜
    df['p1_TPS'] = 0
    df['p2_TPS'] = 0
    for _, group in df.groupby(['match_id', 'set_no', 'game_no']):
        for idx in range(2, len(group)):
            if (group.at[group.index[idx], 'point_victor'] == 1 and
                group.at[group.index[idx - 1], 'point_victor'] == 1 and
                group.at[group.index[idx - 2], 'point_victor'] == 1):
                df.at[group.index[idx], 'p1_TPS'] = 1

            if (group.at[group.index[idx], 'point_victor'] == 2 and
                group.at[group.index[idx - 1], 'point_victor'] == 2 and
                group.at[group.index[idx - 2], 'point_victor'] == 2):
                df.at[group.index[idx], 'p2_TPS'] = 1

    df['server'] = df['server'].replace(2,0)

    df['S1_gain'] =  (1+0.2 * df['server']) * (w_ACE * df['p1_ace'] + w_w * df['p1_winner']
                     + weight_break_pt_won * df['p1_break_pt_won'] + df['p1_TPS'] * w_TPS_1)
    df['S1_loss'] =  (1-0.2 * df['server']) * (w_UE * df['p1_unf_err'] + w_DF * df['p1_double_fault']
                     + weight_break_pt_missed * df['p1_break_pt_missed'] + df['p2_TPS'] * w_TPS_2)

    df['server'] = df['server'].replace(0,1)
    df['server'] = df['server'].replace(1,0)
    df['S2_gain'] =  (1+0.2 * df['server']) * (w_ACE * df['p2_ace'] + w_w * df['p2_winner']
                     + weight_break_pt_won * df['p2_break_pt_won'] + df['p2_TPS'] * w_TPS_2)
    df['S2_loss'] =  (1-0.2 * df['server']) * (w_UE * df['p2_unf_err'] + w_DF * df['p2_double_fault']
                     + weight_break_pt_missed * df['p2_break_pt_missed'] + df['p2_TPS'] * w_TPS_2)

    # 初始化 Mg 列表
    Mg_1 = [0]
    Mg_2 = [0]
    for i in range(1, len(df)):
        delta_t = df['time_interval'].iloc[i]
        f = 0.6 + 0.4 * np.exp(-lambda_value * delta_t)

        delta_Mg_1 = 1.2 * df['S1_gain'].iloc[i] - 1 * df['S1_loss'].iloc[i]
        delta_Mg_2 = 1.2 * df['S2_gain'].iloc[i] - 1 * df['S2_loss'].iloc[i]
        Mg_1_i = f * Mg_1[i - 1] + delta_Mg_1
        Mg_2_i = f * Mg_2[i - 1] + delta_Mg_2
        Mg_1.append(Mg_1_i)
        Mg_2.append(Mg_2_i)

    df['Mg_1'] = Mg_1
    df['Mg_2'] = Mg_2

    # 在 set 变化的位置画一条虚线和更改背景色
    df['set_change'] = df['set_no'].ne(df['set_no'].shift())
    df['game_change'] = df['game_no'].ne(df['game_no'].shift())
    set_change_times = df[df['set_change']]['elapsed_time']

    # 使用 seaborn 设置为默认样式
    sns.set(style='ticks')  # 或者 style='whitegrid'
    # 画出 Mg_1
    plt.figure(figsize=(10, 6),dpi=300)
    sns.lineplot(x=df['elapsed_time'].values, y=Mg_1, label='Mg_1', errorbar=None)
    sns.lineplot(x=df['elapsed_time'].values, y=Mg_2, label='Mg_2', color='#D54A3B', errorbar=None)

    p1_tps_time = df.loc[df['p1_TPS'] == 1, 'elapsed_time'].tolist()
    p2_tps_time = df.loc[df['p2_TPS'] == 1, 'elapsed_time'].tolist()

    # 找出 'p1_TPS' 和 'p2_TPS' 为 1 时对应的 Mg_1
    p1_tps_Mg_1 = [Mg_1[i] for i in range(len(Mg_1)) if df['p1_TPS'].iloc[i] == 1]
    p2_tps_Mg_1 = [Mg_2[i] for i in range(len(Mg_2)) if df['p2_TPS'].iloc[i] == 1]

    sns.scatterplot(x=p1_tps_time, y=p1_tps_Mg_1, color='#E6756E', marker='o', label='p1_TPS')
    sns.scatterplot(x=p2_tps_time, y=p2_tps_Mg_1, color='#579AC3', marker='o', label='p2_TPS')

    # 在 set 变化的位置画一条虚线
    for i in range(0, len(set_change_times), 2):
        start_time = set_change_times.iloc[i]
        end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df['elapsed_time'].max()

        plt.axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
        plt.axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

        color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
        plt.axvspan(start_time, end_time, facecolor=color, alpha=0.3)


    plt.xlabel('elapsed_time(min)')
    plt.ylabel(f"{df['player1'][1]} vs {df['player2'][1]}")
    plt.title(f"Mg_1 and Mg_2 by Time [{path[0:4]}] [{df['player1'][1]} vs {df['player2'][1]}] ")
    plt.legend()
    plt.savefig(f'Momentum_{path[0:4]}.png')
    plt.close()

for i in path:
    print(i)
    Momentum(i)