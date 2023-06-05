import os
from PIL import Image
import random
import json
import numpy as np

if __name__ == '__main__':
    path = './meta'
    all_img = 0
    label = 0
    val_data = list()
    train_data = list()
    val_meta = []
    train_meta = []
    with open(path + '/2_data_class_1.json', 'r') as file:
        file = json.load(file)
        # print(file)
        sample = random.sample(range(len(file)), int(len(file) * 0.2))
        # print(len(sample))
        for i in sample:
            temp=dict()
            temp['file_name'] = file[i]['file_name']
            temp['label'] = file[i]['label']
            val_data.append(temp)
            
            temp_meta = []
            temp_meta = [float(x)  for x in file[i]['bio']]
            val_meta.append(temp_meta)
        # print(file)
        # for _data in file:
        #     # print(_data)
        #     if _data not in val_data:
        #         # train_data.append(_data)
        #         temp=dict()
        #         temp['file_name'] = _data['file_name']
        #         temp['label'] = _data['label']
        #         train_data.append(temp)
                
        #         temp_meta = []
        #         temp_meta = [float(x) for x in _data['bio']]
        #         train_meta.append(temp_meta)
                
        for _data in file:
            # print(_data)
                # train_data.append(_data)
            temp=dict()
            temp['file_name'] = _data['file_name']
            temp['label'] = _data['label']
            train_data.append(temp)
                
            temp_meta = []
            temp_meta = [float(x) for x in _data['bio']]
            train_meta.append(temp_meta)
                
        print(len(val_data))
        print(len(train_data))

    json_data = json.dumps(val_data)
    with open('meta/3_val_data_class.json', 'w') as f_six:
        f_six.write(json_data)
    json_data = json.dumps(train_data)
    with open('meta/3_train_data_class.json', 'w') as f_six:
        f_six.write(json_data)
        
    a=np.array(val_meta)
    b=np.array(train_meta)
    print(a)
    print(a.shape)
    np.save('meta/3_val_meta.npy',a)   # 保存为.npy格式
    np.save('meta/3_train_meta.npy',b)   # 保存为.npy格式
    