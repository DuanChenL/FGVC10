"""
1. 读取全部的csv，按照指定的规则进行清洗（清洗目前包括，指定属性的列所在范围，图片是否失效等等）
2. 对上述清洗之后的结果正则化
3. 将最后的结果存为json
"""

import numpy as np
import pandas as pd
import json
import csv
from PIL import Image
import time
import os
from sklearn.preprocessing import StandardScaler

file_name = '../data/data_train_mean.csv'
img_path = '../data/01_data_train/'

def default_loader(path):
    return Image.open(path).convert('RGB')

"""
1. 读取全部的csv，按照指定的规则进行清洗（清洗目前包括，指定属性的列所在范围，图片是否失效等等）
"""
def wash_porps(file_name,img_path):
    df = pd.read_csv(file_name)
    
    #对属性值的清洗
    #尽量保持每行只有一个属性的判别，这样对于可读性会很好，到时候用for循环判定也更方便改动
    df = df.drop(df[(df['trait_144'] < 1) | (df['trait_144'] > 1000)].index)
    df = df.drop(df[(df['trait_145'] < 0.1) | (df['trait_145'] > 100)].index)

    #对图片的清洗

    # 干净的df索引
    filtered_df = pd.DataFrame(columns=df.columns)

    #清洗过程
    for index, row in df.iterrows():
        try:
            # 读取照片文件
            imgPath = os.path.join(img_path, row['pic_name'])
            img = default_loader(imgPath)

            # 将有效的行添加到新的DataFrame中  #在Pandas 1.0版本及之后的版本中，DataFrame对象的append()方法已经被废弃，不再被推荐使用。
            # filtered_df = filtered_df.append(row, ignore_index=True) #
            filtered_df = pd.concat([filtered_df, row.to_frame().T], ignore_index=True)

            # 显示读取成功的照片信息
            # print(f"Successfully read photo for ID {row['ID']}")
        except Exception as e:
            print(e)

            # 显示读取失败的照片信息
            print(f"Failed to read photo for ID {row['pic_name']}, skipping row...")
        pass

    # 保存干净的数据
    save_path = '../data_pro/'
    filtered_df.to_csv(save_path+'1_filtered_data.csv', index=False)  #对应第一步

    return filtered_df

"""
2. 对上述清洗之后的结果正则化
注意，本函数是对所有的列都进行了正则化
如果仅仅需要其中的一部分列，请额外调用 2.1
两者返回值将会保持一致
"""
def standardize_dataframe(df):
    # """
    # 对DataFrame中的每一列进行标准正则化
    # :param df: 待处理的DataFrame
    # :return: 标准化后的DataFrame
    # """
    df = convert_object_to_float(df)
    for col in df.columns:
        if df[col].dtype != 'object':
            # 对于数值型数据，进行标准化
            df[col] = (df[col] - df[col].mean()) / df[col].std()


    # save
    save_path = '../data_pro/'
    save_name = '2_normalized_all.csv'
    csv_path = save_path+save_name
    df.to_csv(csv_path, index=False)  #对应第二步

    return df, csv_path

"""
2.1 返回指定的列，返回值和2保持一致
"""
def get_dataframe_columns(df, column_names):
    """
    返回DataFrame中指定列的数据
    param df: 待处理的DataFrame
    param column_names: 需要返回数据的列名列表
    return: 返回指定列的数据
    """
    for col in column_names:
        if col not in df.columns:
            raise ValueError(f"{col} not found in DataFrame columns")
        
    # save
    save_path = '../data_pro/'
    save_name = '2_normalized_part.csv'
    csv_path = save_path + save_name
    df = df[column_names]
    df.to_csv(csv_path, index=False)  #对应第二步

    return df, csv_path


"""
3. 将最后的csv结果存为json
"""
def csv2json(csv_path):
    json_file = list()
    count = 0
    with open(csv_path, 'r') as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            ann_dict = dict()
            ann_label_name = dict()
            if count != 0:
                ann_dict['file_name'] = line[0]
                ann_dict['label'] = line[1:]  #指定lable列
                json_file.append(ann_dict)
            count += 1

    json_data = json.dumps(json_file)
    save_path = '../data_pro/'
    save_name = '3_normalized_data.json'
    json_path = save_path + save_name
    with open( json_path, 'w') as f_six:
        f_six.write(json_data)

    return json_path


"""
由于过程中会将df转变程object，在正则化的过程中，需要将其转换会float64
"""
def convert_object_to_float(df):
    """
    将DataFrame中的object类型的列转换为float64类型
    :param df: 待处理的DataFrame
    :return: 转换后的DataFrame
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # 将object类型的列转换为float64类型
            try:
                df[col] = df[col].astype('float64')
            except ValueError:
                print(f"Unable to convert column '{col}' to float64")

    return df


if __name__ == '__main__':
    #1. 读取全部的csv，按照指定的规则进行清洗（清洗目前包括，指定属性的列所在范围，图片是否失效等等）
    df = wash_porps(file_name=file_name,img_path=img_path)

    #2. 对上述清洗之后的结果正则化
    df, csv_path = standardize_dataframe(df)
    df_part, csv_path = get_dataframe_columns(df, ['pic_name','trait_4','trait_46'])

    #3. 将最后的结果存为json
    json_path = csv2json(csv_path)

    #4. 当然，如果你还需要划分训练集和验证集，请使用同目录下的 4_divede_train_Data.py，记得修改路径