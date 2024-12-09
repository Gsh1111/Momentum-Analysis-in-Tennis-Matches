import pandas as pd

# 读取 Excel 文件
file_path = 'processed_data.xlsx'
df = pd.read_excel(file_path)

# # 1. elapsed_time 统一转化为按分钟计数
# df['elapsed_time'] = df['elapsed_time'] / 60

# 2. winner_shot_type F：正手1；B：反手2
df['winner_shot_type'] = df['winner_shot_type'].map({'F': 1, 'B': 2})

# 3. serve_depth CTL：1；NCTL：2
df['serve_depth'] = df['serve_depth'].map({'CTL': 1, 'NCTL': 2})

# 4. return_depth D：1；ND：2
df['return_depth'] = df['return_depth'].map({'D': 1, 'ND': 2})

# 5. serve_width 表示发球的方向编码
serve_width_mapping = {'B': 1, 'BC': 2, 'BW': 3, 'C': 4, 'W': 5}
df['serve_width'] = df['serve_width'].map(serve_width_mapping)

# 保存修改后的数据
df.to_excel('processed_data_updated.xlsx', index=False)
