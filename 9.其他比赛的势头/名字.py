# 生成文件名列表
file_names_1301_1316 = [f'13{i}.xlsx' for i in range(1, 17)]
file_names_1401_1408 = [f'140{i}.xlsx' for i in range(1, 9)]
file_names_1501_1504 = [f'150{i}.xlsx' for i in range(1, 5)]
file_names_1601_1602 = [f'160{i}.xlsx' for i in range(1, 3)]
# 合并文件名列表
file_names = file_names_1301_1316 + file_names_1401_1408+file_names_1501_1504+file_names_1601_1602
print(file_names)