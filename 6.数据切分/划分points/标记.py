import pandas as pd
import copy

# 读取Excel文件
df = pd.read_excel('processed_data_updated.xlsx')

df_1 = df[['match_id', 'set_no', 'game_no', 'game_victor']].copy()
# 存储处理后的数据框
processed_dataframes = []
# 根据match_id、set_no和game_no进行划分
grouped_1 = df_1.groupby(['match_id', 'set_no', 'game_no'])

for (match_id, set_no, game_no), group in grouped_1:
    current_value = 0
    # 创建group的副本
    group_copy = group.copy()
    for i in range(len(group_copy)-1, -1, -1):
        if group_copy['game_victor'].iloc[i] == 1 or group_copy['game_victor'].iloc[i] == 2:
            current_value = group_copy['game_victor'].iloc[i]

        group_copy.loc[group_copy.index[i], 'game_flag'] = current_value
    # 将处理后的数据添加到列表中
    processed_dataframes.append(copy.deepcopy(group_copy))

df_1 = pd.concat(processed_dataframes)
game_flag_column = df_1[['game_flag']].reset_index(drop=True)  # 提取出df_1中的game_flag
df = pd.concat([df, game_flag_column], axis=1)  # 将game_flag列拼接到df中


df_2 = df[['match_id', 'set_no', 'set_victor']].copy()

# 存储处理后的数据框
processed_dataframes = []
# 根据match_id、set_no进行划分
grouped_2 = df_2.groupby(['match_id', 'set_no'])

for (match_id, set_no), group in grouped_2:
    current_value = 0
    # 创建group的副本
    group_copy = group.copy()
    for i in range(len(group_copy) - 1, -1, -1):
        if group_copy['set_victor'].iloc[i] == 1 or group_copy['set_victor'].iloc[i] == 2:
            current_value = group_copy['set_victor'].iloc[i]

        group_copy.loc[group_copy.index[i], 'set_flag'] = current_value
    # 将处理后的数据添加到列表中
    processed_dataframes.append(copy.deepcopy(group_copy))

df_2 = pd.concat(processed_dataframes)
set_flag_column = df_2[['set_flag']].reset_index(drop=True)  # 提取出df_2中的set_flag
df = pd.concat([df, set_flag_column], axis=1)  # 将set_flag列拼接到df中

# 保存处理后的数据到新的Excel文件
df.to_excel('flag.xlsx')
