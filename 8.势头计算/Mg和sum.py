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

# sns.set_style("ticks")  # 设置背景样式

# 创建第一个坐标系
fig, ax1 = plt.subplots(figsize=(10,6),dpi=300)

# 在第一个坐标系上画第一条曲线
sns.lineplot(x='elapsed_time', y='S1_sum', label='Performance', data=df, ax=ax1, errorbar=None)

# 创建第二个坐标系
ax2 = ax1.twinx()

# 在第二个坐标系上画第二条曲线
sns.lineplot(x='elapsed_time', y='Mg_1', label='Momentum', data=df, color='#E6756E', ax=ax2, errorbar=None)

# 设置图表标题和标签
plt.title('Performance & Momentum of p1',fontsize= 16)
ax1.set_xlabel('Elapsed Time (min)',fontsize= 14)

# 更改第二个坐标系的 y 轴标签
ax1.set_ylabel('Performance',fontsize= 14)
ax2.set_ylabel('Momentum', fontsize=14)

# 在 set 变化的位置画一条虚线
for i in range(0, len(set_change_times), 2):
    start_time = set_change_times.iloc[i]
    end_time = set_change_times.iloc[i + 1] if i + 1 < len(set_change_times) else df['elapsed_time'].max()

    plt.axvline(x=start_time, linestyle='--', color='gray', linewidth=0.8)
    plt.axvline(x=end_time, linestyle='--', color='gray', linewidth=0.8)

    color = 'white' if (i+1) % 2 == 0 else '#F0F0F0' # 每两个区域之间交替使用浅灰色和白色
    plt.axvspan(start_time, end_time, facecolor=color, alpha=0.3)

ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 1), borderaxespad=0.5, fontsize=15)
ax2.legend(loc='upper left', bbox_to_anchor=(0.01, 0.9), borderaxespad=0.5, fontsize=15)
plt.tight_layout()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()



plt.figure(figsize=(10, 6),dpi=500)

# sns.scatterplot(x=p1_tps_time, y=df.loc[df['p1_TPS'] == 1, 'Mg_2'], color='#579AC3', marker='o', label='p1_TPS')
# sns.scatterplot(x=p2_tps_time, y=df.loc[df['p2_TPS'] == 1, 'Mg_1'], color='#E6756E', marker='o', label='p2_TPS')

sns.lineplot(x='elapsed_time', y='S2_sum', label='p2 performance', data=df, errorbar=None)
sns.lineplot(x='elapsed_time', y='Mg_2', label='Mg_2', data=df, color='#E6756E', errorbar=None)


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
