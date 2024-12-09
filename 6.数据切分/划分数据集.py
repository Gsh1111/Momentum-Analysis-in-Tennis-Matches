import pandas as pd

df = pd.read_excel('Wimbledon_featured_matches.xlsx')

# print(df)

# 根据match_id分组并保存为不同的Excel文件

for match_id, group in df.groupby('match_id'):
    filename = f"{match_id[-4:]}.xlsx"  # 文件名使用match_id
    group.to_excel(filename, index=False)

