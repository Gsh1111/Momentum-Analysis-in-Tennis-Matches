import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置风格
sns.set(style='white')

# 读取 Excel 文件
file_path = 'processed_data_updated.xlsx'
df = pd.read_excel(file_path)
# 列名长度从大到小排序
sorted_columns = sorted(df.columns, key=len, reverse=False)
df = df[sorted_columns]
# 选择要分析的列
# selected_columns = [ 'set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
#                      'server', 'serve_no', 'point_victor', 'p1_points_won', 'p2_points_won',
#                      'p1_points', 'p2_points', 'game_victor', 'set_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner',
#                      'winner_shot_type', 'p1_double_fault', 'p2_double_fault', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt',
#                      'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won',
#                      'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed', 'p1_distance_run', 'p2_distance_run',
#                      'rally_count', 'speed_mph', 'serve_width', 'serve_depth', 'return_depth']

# selected_columns = ['set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
#                     'server', 'serve_no', 'point_victor', 'p1_points', 'p2_points']
# selected_columns = ['rally_count', 'speed_mph', 'serve_width', 'serve_depth', 'return_depth']

# 假设你的数据框是 df

# 选择以 p1_ 开头的列
p1_columns = [col for col in df.columns if col.startswith('p1_')]

# 选择以 p2_ 开头的列
p2_columns = [col for col in df.columns if col.startswith('p2_')]

p1_columns.remove('p1_score')
p2_columns.remove('p2_score')


# 创建一个新的 DataFrame，移除 'p2_' 前缀
df_subset = df[p2_columns].rename(columns=lambda x: x.replace('p2_', '').replace('_', ' ')).rename(lambda x: x.title(), axis=1)

# 计算相关性矩阵
df_corr = df_subset.corr()

# 设置一个“上三角行”蒙版
mask = np.triu(np.ones_like(df_corr, dtype=bool))
mask = mask[1:, :-1]
corr = df_corr.iloc[1:, :-1].copy()

# 设置调色盘
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# 生成半边热图
plt.figure(figsize=(15, 14), dpi = 300)
sns.heatmap(corr,
            mask=mask,
            cmap=cmap,
            vmax=0.4,
            vmin=-0.4,
            center=0,
            square=True,
            linewidths=5,
            annot=True,
            annot_kws={'size': 12},
            cbar_kws={"shrink": .8}
            )
# 调整标签字体大小
plt.tick_params(axis='both', which='both', labelsize=14)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.title('Correlation Heat Map of $\mathbf{Player 2}$', fontsize = 20)
plt.show()
