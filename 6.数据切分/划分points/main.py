import pandas as pd
import numpy as np

# 读取 Excel 文件
file_path = 'Wimbledon_featured_matches.xlsx'
df = pd.read_excel(file_path)

# 根据 match_id 分组
df_grouped = df.groupby('match_id')

# 存储处理后的数据框
processed_dataframes = []

# 遍历每个 match_id 的分组
for match_id, group in df_grouped:
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
df_processed = pd.concat(processed_dataframes)

df_processed.to_excel('processed_data.xlsx')