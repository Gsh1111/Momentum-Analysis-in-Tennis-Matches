import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 假设您有这两个列表
data = {
    'Factor': [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1],
    'Success_Rate': [0.913385827, 0.902362205, 0.91496063, 0.918110236, 0.899212598, 0.890866142, 0.883464567, 0.91496063, 0.921259843, 0.916535433, 0.935984252, 0.913385827, 0.908661417, 0.886614173, 0.899212598, 0.908661417, 0.91496063, 0.885590551, 0.878740157, 0.883464567, 0.891338583]
}

df = pd.DataFrame(data)
m = 0.935984252
# 画折线图（使用样条插值）
plt.figure(figsize=(16, 5),dpi=300)
color_rgb = [31/255, 119/255, 180/255]

# 使用 make_interp_spline 进行样条插值
xnew = np.linspace(min(df['Factor']), max(df['Factor']), 300)
spl = make_interp_spline(df['Factor'], df['Success_Rate'], k=3)
y_smooth = spl(xnew)

# 绘制平滑的曲线
plt.plot(xnew, y_smooth, color=color_rgb, label='Accuracy Curve')
plt.ylim(0.835,0.985)
# 添加参考线
plt.axvline(x=1, ymin=0, ymax=0.674, color='#D54A3B', linestyle='-.', label='Reference Line (Factor = 1)')
plt.axhline(y=m, color='#D54A3B', linestyle='--',linewidth=2)

factor_values = df['Factor']
success_rate_values = df['Success_Rate']
lower_bound = success_rate_values * 0.96
upper_bound = success_rate_values * 1.04

# 使用 fill_between 方法填充颜色范围
plt.fill_between(factor_values, lower_bound, upper_bound, color='gray', alpha=0.05)

# 增加刻度点密度
plt.xticks(factor_values)

# 设置图表标题和标签
plt.title('Sensitivity Analysis', fontsize=18)
plt.xlabel('volatility', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.8)
plt.legend(fontsize=14)  # 显示图例

# 显示图表
plt.show()
