import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import interp1d

# 读取Excel表格
df = pd.read_excel('2.xlsx')  # 替换为你的Excel文件路径

# 在 set 变化的位置画一条虚线和更改背景色
df['set_change'] = df['set_no'].ne(df['set_no'].shift())
set_change_times = df[df['set_change']]['elapsed_time']

# 使用 seaborn 画图
plt.figure(figsize=(10,5),dpi=300)
# 找出 'p1_TPS' 为 1 的所有横坐标,找出 'p2_TPS' 为 1 的所有横坐标
p1_tps_time = df.loc[df['p1_TPS'] == 1, 'elapsed_time'].tolist()
p2_tps_time = df.loc[df['p2_TPS'] == 1, 'elapsed_time'].tolist()

# sns.scatterplot(x=p1_tps_time, y=df.loc[df['p1_TPS'] == 1, 'S2_sum'], color='#579AC3', marker='o', label='p1_TPS')
# sns.scatterplot(x=p2_tps_time, y=df.loc[df['p2_TPS'] == 1, 'S1_sum'], color='#E6756E', marker='o', label='p2_TPS')

sns.lineplot(x='elapsed_time', y='S1_sum', label='Performance of Player 1', data=df, errorbar=None)
sns.lineplot(x='elapsed_time', y='S2_sum', label='Performance of Player 2', data=df, color='#E6756E', errorbar=None)


# 计算两条曲线的差值
df['difference'] = df['S1_sum'] - df['S2_sum']

# 判断差值的正负关系，分别填充颜色
positive_indices = df.index[df['difference'] >= 0]
negative_indices = df.index[df['difference'] < 0]

# 填充正值的区域
plt.fill_between(x=df['elapsed_time'].iloc[positive_indices], y1=0, y2=df['difference'].iloc[positive_indices], color='#579AC3', alpha=0.3, label='Player 1 in advantage')

# 填充负值的区域
plt.fill_between(x=df['elapsed_time'].iloc[negative_indices], y1=0, y2=df['difference'].iloc[negative_indices], color='#E6756E', alpha=0.3, label='Player 2 in advantage')


# 设置图表标题和标签
plt.title('Performance of Player 1 and Player 2 ',fontsize= 16)
plt.xlabel('Elapsed Time (min)',fontsize= 14)
plt.ylabel('Performance',fontsize= 14)

# 在 set 变化的位置画一条虚线
for i in range(0, len(set_change_times), 2):
    start_time = set_change_times.iloc[i]
    end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df['elapsed_time'].max()

    plt.axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
    plt.axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

    color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
    plt.axvspan(start_time, end_time, facecolor=color, alpha=0.3)

plt.scatter(x=df['elapsed_time'].iloc[-1]+3, y=df['S1_sum'].iloc[-1], marker='*', color='#E6756E', label='p1 Win!', s=250 ,alpha=1)

# 显示图表
plt.legend(loc='upper left',fontsize=10)
plt.show()



plt.figure(figsize=(10, 6),dpi=300)

# sns.scatterplot(x=p1_tps_time, y=df.loc[df['p1_TPS'] == 1, 'Mg_2'], color='#579AC3', marker='o', label='p1_TPS')
# sns.scatterplot(x=p2_tps_time, y=df.loc[df['p2_TPS'] == 1, 'Mg_1'], color='#E6756E', marker='o', label='p2_TPS')

sns.lineplot(x='elapsed_time', y='Mg_1', label='Mg_1', data=df, errorbar=None)
sns.lineplot(x='elapsed_time', y='Mg_2', label='Mg_2', data=df, color='#E6756E', errorbar=None)

# # 计算两条曲线的差值
# df['difference'] = df['Mg_1'] - df['Mg_2']
#
# # 判断差值的正负关系，分别填充颜色
# positive_indices = df.index[df['difference'] >= 0]
# negative_indices = df.index[df['difference'] < 0]
#
# # 填充正值的区域
# plt.fill_between(x=df['elapsed_time'].iloc[positive_indices], y1=0, y2=df['difference'].iloc[positive_indices], color='#579AC3', alpha=0.3, label='Positive Area')
#
# # 填充负值的区域
# plt.fill_between(x=df['elapsed_time'].iloc[negative_indices], y1=0, y2=df['difference'].iloc[negative_indices], color='#E6756E', alpha=0.3, label='Negative Area')

# 设置图表标题和标签
plt.title('Line Plot of Mg_1 and Mg_2 Cumulative Sum',fontsize= 14)
plt.xlabel('Elapsed Time (min)',fontsize= 14)
plt.ylabel('Cumulative Sum',fontsize= 14)

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
