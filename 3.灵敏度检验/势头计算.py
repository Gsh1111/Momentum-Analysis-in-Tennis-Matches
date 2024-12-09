import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_file_name = '.csv'
path_list = []

# 获取当前文件夹中所有匹配的文件
matching_files = [file for file in os.listdir() if csv_file_name in file]

# 定义倍率列表
sensitivity_factors = [0.5, 0.6, 1, 1.4, 1.5]

# 存储结果的列表
result_list = []

for file in matching_files:
    print(file)
    # 创建以文件名为名的文件夹
    folder_name = file.split('.')[0]
    os.makedirs(folder_name, exist_ok=True)

    df = pd.read_csv(file)
    path = []

    for match_id, group in df.groupby('match_id'):
        filename = f"{folder_name}/{match_id[-4:]}.xlsx"  # 文件名使用match_id
        group.to_excel(filename, index=False)
        path.append(filename)

    # 设置权重
    w_ACE = 0.3
    w_w = 0.32
    w_TPS_1 = 0.38
    w_TPS_2 = 0.42
    w_DF = 0.55
    w_UE = 0.45
    weight_break_pt_won = 1.5
    weight_break_pt_missed = 0.6
    lambda_value = 0.2

    for factor in sensitivity_factors:
        # 在这里修改参数值
        w_ACE *= factor
        w_w *= factor
        w_TPS_1 *= factor
        w_TPS_2 *= factor
        w_DF *= factor
        w_UE *= factor
        weight_break_pt_won *= factor
        weight_break_pt_missed *= factor
        lambda_value *= factor
        correct_num = 0
        num = 0

        def Momentum(path):
            if os.path.isfile(path):
                df = pd.read_excel(path)  # 替换为你的Excel文件路径
                df['ElapsedTime'] = pd.to_timedelta(df['ElapsedTime'])
                df['ElapsedTime'] = df['ElapsedTime'].dt.total_seconds() / 60
                df['time_interval'] = df['ElapsedTime'].diff().fillna(0)
                df['p1_TPS'] = 0
                df['p2_TPS'] = 0
                for _, group in df.groupby(['match_id', 'SetNo', 'GameNo']):
                    for idx in range(2, len(group)):
                        if (group.at[group.index[idx], 'PointWinner'] == 1 and
                            group.at[group.index[idx - 1], 'PointWinner'] == 1 and
                            group.at[group.index[idx - 2], 'PointWinner'] == 1):
                            df.at[group.index[idx], 'p1_TPS'] = 1
                        if (group.at[group.index[idx], 'PointWinner'] == 2 and
                            group.at[group.index[idx - 1], 'PointWinner'] == 2 and
                            group.at[group.index[idx - 2], 'PointWinner'] == 2):
                            df.at[group.index[idx], 'p2_TPS'] = 1

                # df['PointServer'] = df['PointServer'].replace(2,1)
                # df['PointServer'] = df['PointServer'].replace(3,0)
                # df['PointServer'] = df['PointServer'].replace(4,0)
                df['PointServer'] = df['PointServer'].replace(2,0)

                df['S1_gain'] =  (1+0.2 * df['PointServer']) * (w_ACE * df['P1Ace'] + w_w * df['P1Winner'] + weight_break_pt_won * df['P1BreakPointWon'] + df['p1_TPS'] * w_TPS_1)
                df['S1_loss'] =  (1-0.2 * df['PointServer']) * (w_UE * df['P1UnfErr'] + w_DF * df['P1DoubleFault'] + weight_break_pt_missed * df['P1BreakPointMissed'] + df['p2_TPS'] * w_TPS_2)
                df['PointServer'] = df['PointServer'].replace(0,1)
                df['PointServer'] = df['PointServer'].replace(1,0)
                df['S2_gain'] =  (1+0.2 * df['PointServer']) * (w_ACE * df['P2Ace'] + w_w * df['P2Winner'] + weight_break_pt_won * df['P2BreakPointWon'] + df['p2_TPS'] * w_TPS_2)
                df['S2_loss'] =  (1-0.2 * df['PointServer']) * (w_UE * df['P2UnfErr'] + w_DF * df['P2DoubleFault'] + weight_break_pt_missed * df['P2BreakPointMissed'] + df['p2_TPS'] * w_TPS_2)
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
                df['set_change'] = df['SetNo'].ne(df['SetNo'].shift())
                df['game_change'] = df['GameNo'].ne(df['GameNo'].shift())
                last_Mg_1 = Mg_1[-1]
                last_Mg_2 = Mg_2[-1]
                if last_Mg_1 > last_Mg_2:
                    pred_winner = 1
                else:
                    pred_winner = 2

                last_set_winner = df['SetWinner'].iloc[-1]
                if last_set_winner == pred_winner:
                    return (1,1)
                else:
                    return (0,1)
            else:
                return (0,0)
        for i in path:

            a,b = Momentum(i)
            correct_num += a
            num += b
            if b==1:
                print(i)
            else:
                print(f'{i}:don\'t exit!')

        # 还原参数值，以便下一次循环
        w_ACE /= factor
        w_w /= factor
        w_TPS_1 /= factor
        w_TPS_2 /= factor
        w_DF /= factor
        w_UE /= factor
        weight_break_pt_won /= factor
        weight_break_pt_missed /= factor
        lambda_value /= factor

        result_str = (factor, file[:-4], correct_num, num, correct_num / num)
        result_list.append(result_str)


# 将结果保存为DataFrame
result_df = pd.DataFrame(result_list, columns=['Factor', 'Year-Match', 'Correct_Num', 'Total_Num', 'Success_Rate'])
result_df.to_excel(f'{csv_file_name[:-4]}_sensitivity_analysis.xlsx', index=False)