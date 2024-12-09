# -*- coding: utf-8 -*-

import pandas as pd

# 读取数据集
data = pd.read_excel('Wimbledon_featured_matches.xlsx')
# print(data)
# 检查每列的缺失值数量
missing_values = data.isnull().sum()
print(missing_values)
# # 打印出含有缺失值的列
# columns_with_missing_values = missing_values[missing_values > 0]
# print("Columns with missing values:")
# print(columns_with_missing_values)
#
# # 打印出所有含有缺失值的行
# rows_with_missing_values = data[data.isnull().any(axis=1)]
# print("\nRows with missing values:")
# print(rows_with_missing_values)
#
# # 针对 speed_mph 进行分类并打印含有缺失值的行
# rows_with_missing_speed_mph = data[data['speed_mph'].isnull()]
# print("Rows with missing values in 'speed_mph':")
# print(rows_with_missing_speed_mph)
#
# # 针对 serve_width 进行分类并打印含有缺失值的行
# rows_with_missing_serve_width = data[data['serve_width'].isnull()]
# print("\nRows with missing values in 'serve_width':")
# print(rows_with_missing_serve_width)
#
# # 针对 serve_depth 进行分类并打印含有缺失值的行
# rows_with_missing_serve_depth = data[data['serve_depth'].isnull()]
# print("\nRows with missing values in 'serve_depth':")
# print(rows_with_missing_serve_depth)
#
# # 针对 return_depth 进行分类并打印含有缺失值的行
# rows_with_missing_return_depth = data[data['return_depth'].isnull()]
# print("\nRows with missing values in 'return_depth':")
# print(rows_with_missing_return_depth)
