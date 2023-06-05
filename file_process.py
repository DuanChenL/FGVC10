import csv
import config
import json
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    json_file = list()
    label_list = list()
    # ann_dict = dict()
    count = 0
    with open('data/data_train_mean.csv', 'r') as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            ann_dict = dict()
            ann_label_name = dict()
            if count != 0:
                # ann_dict['file_name'] = line[19]
                # ann_dict['label'] = line[20:]
                ann_dict['file_name'] = line[19]
                ann_dict['label'] = line[0]
                ann_dict['bio'] = line[2:19]
                label_list.append(line[0])
                json_file.append(ann_dict)
            count += 1
                # print(line[19:])


    # Creating a instance of label Encoder.
    le = LabelEncoder()

    # Using .fit_transform function to fit label
    # encoder and return encoded label
    print(label_list)
    labels = le.fit_transform(label_list)
    for i in range(len(labels)):
        json_file[i]['label'] = str(labels[i])
    # print(json_file)
    json_data = json.dumps(json_file)
    with open('meta/1_train_data_class.json', 'w') as f_six:
        f_six.write(json_data)