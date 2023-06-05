from PIL import Image
import json
import config

def default_loader(path):
    # image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    # return image
    return Image.open(path).convert('RGB')

if __name__ == "__main__":
    json_list = list()
    with open('./meta/1_train_data_class.json') as file:
        json_file = json.load(file)
        for file_name in json_file:
            # print(file_name['file_name'])
            try:
                img = default_loader(config.PATH['train_img'] + file_name['file_name'])
                json_list.append(file_name)
            except Exception as e:
                print(e)
                print(json_list[-1])

    json_data = json.dumps(json_list)
    with open('./meta/2_data_class_1.json', 'w') as f_six:
        f_six.write(json_data)