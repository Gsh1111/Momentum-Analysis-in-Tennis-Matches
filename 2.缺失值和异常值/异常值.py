import pandas as pd

df = pd.read_excel('Wimbledon_featured_matches.xlsx')
def check_value_range(column, valid_values):
    invalid_values = [value for value in column.unique() if value not in valid_values]
    return invalid_values

MAX = 1000
# 定义每个变量的取值范围
valid_ranges = {
    'set_no': {1, 2, 3, 4, 5},
    'game_no': set(range(1, 14)),
    'point_no': set(range(1, MAX)),
    'p1_sets': {0, 1, 2},
    'p2_sets': {0, 1, 2},
    'p1_games': set(range(7)),
    'p2_games': set(range(7)),
    'p1_score': {0, 15, 30, 40, 'AD'},
    'p2_score': {0, 15, 30, 40, 'AD'},
    'server': {1, 2},
    'serve_no': {1, 2},
    'point_victor': {1, 2},
    'p1_points_won': set(range(0, MAX)),
    'p2_points_won': set(range(0, MAX)),
    'game_victor': {0, 1, 2},
    'set_victor': {0, 1, 2},
    'p1_ace': {0, 1},
    'p2_ace': {0, 1},
    'p1_winner': {0, 1},
    'p2_winner': {0, 1},
    'winner_shot_type': {'F', 'B'},
    'p1_double_fault': {0, 1},
    'p2_double_fault': {0, 1},
    'p1_unf_err': {0, 1},
    'p2_unf_err': {0, 1},
    'p1_net_pt': {0, 1},
    'p2_net_pt': {0, 1},
    'p1_net_pt_won': {0, 1},
    'p2_net_pt_won': {0, 1},
    'p1_break_pt': {0, 1},
    'p2_break_pt': {0, 1},
    'p1_break_pt_won': {0, 1},
    'p2_break_pt_won': {0, 1},
    'p1_break_pt_missed': {0, 1},
    'p2_break_pt_missed': {0, 1},
    # 'p1_distance_run': (0, float('inf')),
    # 'p2_distance_run': (0, float('inf')),
    'rally_count': set(range(1, MAX)),
    # 'speed_mph': (1, float('inf')),
    'serve_width': {'B', 'BC', 'BW', 'C', 'W'},
    'serve_depth': {'CTL', 'NCTL'},
    'return_depth': {'D', 'ND'},
}

valid_ranges_continuous = {
    'p1_distance_run': (0, float('inf')),
    'p2_distance_run': (0, float('inf')),
    'speed_mph': (0, float('inf')),
}

# 检查每个离散型变量的取值范围是否合法
for column, valid_values in valid_ranges.items():
    invalid_values = check_value_range(df[column], valid_values)
    if invalid_values:
        print(f"Invalid values in column '{column}': {invalid_values}")

# 检查每个连续型变量的取值范围是否合法
for column, (min_value, max_value) in valid_ranges_continuous.items():
    invalid_values = df[(df[column] < min_value) | (df[column] > max_value)][column]
    if not invalid_values.empty:
        print(f"Invalid values in column '{column}': {invalid_values}")

# 选择多列
selected_columns = ['p1_score', 'p2_score', 'winner_shot_type', 'rally_count']

# 遍历每一列
for column in selected_columns:
    # 检查当前列中的异常值
    invalid_values = check_value_range(df[column], valid_ranges[column])

    # 提取包含异常值的行的数据
    invalid_rows = df[df[column].isin(invalid_values)]

    # 保存到新的 Excel 表中
    invalid_rows.to_excel(f"invalid_rows_{column}.xlsx", index=False)

