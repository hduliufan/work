#数据缺少/数据中含有非数值如NaN 
#dropna()删除行或者列（axis=1）
#how='all'删除全为nan的行
#thres=num 保留至少num个数字的行
#subset['c']对c列中的非数行删除所在行
#缺失数据填补均值插补Imputer 按特征均值补齐 missing_values= 非数值，
# strategy= 修改方法（mean median,most_frequent(出现次数频率最高的
# 替换NaN常用于类别特征)）
#isnull方法来测试数据是否为数值（数值返回0，非数字返回 1）
import pandas as pd
from io import StringIO
csv_data= '''a,b,c,d
            1.0,2.0,3.0,4.0
            1.09,2.0,,3.0
            1,,,'''
csv_data = pd.read_csv(StringIO(csv_data))
print(csv_data)
print(csv_data.isnull().sum())
print(csv_data.values)
print(csv_data.describe())
csv_datarow =csv_data.dropna(thresh=3)
print(csv_datarow)
csv_datacolumn= csv_data.dropna(subset=['b'])
print(csv_datacolumn)
from sklearn.preprocessing import Imputer
impute= Imputer (missing_values='NaN',strategy= 'mean',axis=0)
#建立相应数据模型
impute.fit(csv_data)
#对数据模型的应用transform
df= impute.transform(csv_data.values)
print(df)