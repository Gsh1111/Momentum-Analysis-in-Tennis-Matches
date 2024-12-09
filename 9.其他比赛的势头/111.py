import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 使用 seaborn 画图
plt.figure(figsize=(12, 8),dpi=300)
# 创建第一个坐标系
fig, ax1 = plt.subplots()

# 在第一个坐标系上画第一条曲线
sns.lineplot(x='elapsed_time', y='S1_sum', label='p1 performance', data=df, ax=ax1, errorbar=None)

# 创建第二个坐标系
ax2 = ax1.twinx()

# 在第二个坐标系上画第二条曲线
sns.lineplot(x='elapsed_time', y='Mg_1', label='Mg_1', data=df, color='#E6756E', ax=ax2, errorbar=None)

# 设置图表标题和标签
plt.title('Performance of S1 and S2 ',fontsize= 16)
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

plt.show()