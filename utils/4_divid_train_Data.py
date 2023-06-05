"""
划分数据集，分为train和val
"""


import os
from PIL import Image
import random
import json

if __name__ == '__main__':
    path = '../data_pro'
    all_img = 0
    label = 0 
    val_data = list()
    train_data = list()
    with open(path + '/3_normalized_data.json', 'r') as file:
        file = json.load(file)
        # print(file)
        sample = random.sample(range(len(file)), int(len(file) * 0.2))
        # print(len(sample))
        for i in sample:
            val_data.append(file[i])
        # print(file)
        for _data in file:
            # print(_data)
            if _data not in val_data:
                train_data.append(_data)
        print(len(val_data))
        print(len(train_data))
    #     imgsize_count = dict()
    #     # print(dirpath + filepath)
    #     img_num = len(filenames)
    #     all_img += img_num
    #     sample = random.sample(range(len(filenames)), int(len(filenames) * 0.2))
    #     label = int(dirpath[-1])
    #     for name in sample:
    #         sample_name = dict()
    #         sample_name['file_name'] = dirpath + "/" + filenames[name]
    #         # print(dirpath)
    #         print(sample_name)
    #         sample_name['label'] = label
    #         val_data.append(sample_name)
    #     for i in range(len(filenames)):
    #         if i not in sample:
    #             train_name = dict()
    #             train_name['file_name'] = dirpath + "/" + filenames[i]
    #             train_name['label'] = label
    #             train_data.append(train_name)
    #     # val_data[label] = sample_name
    #     # train_data[label] = train_name
    #     # print(sample)
    #     # for filepath in filenames:
    #     #     img = Image.open(dirpath + "\\" + filepath)
    #     #     imgSize = img.size  # 图片的长和宽
    #     #     # print(imgSize)
    #     #     if imgSize not in imgsize_count.keys():
    #     #         imgsize_count[imgSize] = 0
    #     #     imgsize_count[imgSize] += 1
    #     #     # img.show()
    #     # print(imgsize_count)
    #     # print(img_num)
    # # print(all_img)
    # # print(val_data)
    # # print(train_data)
    json_data = json.dumps(val_data)
    with open(path + '/4_val_norm.json', 'w') as f_six:
        f_six.write(json_data)
    json_data = json.dumps(train_data)
    with open(path + '/4_train_norm.json', 'w') as f_six:
        f_six.write(json_data)