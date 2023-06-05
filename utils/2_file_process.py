"""
将指定csv文件转为json格式，此处的文件包含所有实例
"""


import csv
# import config
import json
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    json_file = list()
    # label_list = list()
    # ann_dict = dict()
    count = 0
    with open('../data/001_train_mean_trait_5.csv', 'r') as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            ann_dict = dict()
            ann_label_name = dict()
            if count != 0:
                ann_dict['file_name'] = line[0]
                ann_dict['label'] = line[1:6]  #指定lable列
                # ann_dict['file_name'] = line[19]
                # ann_dict['label'] = line[0]
                # ann_dict['label_name'] = line[0]
                # label_list.append(line[0])
                json_file.append(ann_dict)
            count += 1
                # print(line[19:])


    # Creating a instance of label Encoder.
    # le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    # print(label_list)
    # labels = le.fit_transform(label_list)
    # for i in range(len(labels)):
    #     json_file[i]['label'] = str(labels[i])
    # print(json_file)
    json_data = json.dumps(json_file)
    with open('../data/001_train_norm_part.json', 'w') as f_six:
        f_six.write(json_data)