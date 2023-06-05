from PIL import Image
import json

def default_loader(path):
    return Image.open(path).convert('RGB')

if __name__ == "__main__":
    json_list = list()
    with open('../data/001_train_norm_part.json') as file:
        json_file = json.load(file)
        for file_name in json_file:
            # print(file_name['file_name'])
            try:
                img = default_loader('../data/01_data_train/' + file_name['file_name'])
                json_list.append(file_name)
            except Exception as e:
                print(e)
                print(json_list[-1])

    json_data = json.dumps(json_list)
    with open('../data/001_train_norm_part_clean.json', 'w') as f_six:
        f_six.write(json_data)