import numpy as np
import pandas as pd

df = pd.read_excel('1701-pro.xlsx')  # 替换为你的Excel文件路径

# 转换elapsed_time为timedelta类型
df['elapsed_time'] = pd.to_timedelta(df['elapsed_time'])

# 将 timedelta 转换为分钟
df['elapsed_time'] = df['elapsed_time'].dt.total_seconds() / 60

# 存储每个分组的结果
result_frames = []

# 按照 'set_no' 和 'game_no' 进行分组，计算时间间隔
df['time_interval'] = df['elapsed_time'].diff().fillna(0)

df.to_excel('1701-promax.xlsx')
