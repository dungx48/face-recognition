import json
import os
import sys

from requests import get

def get_details(data_path_root):
    details = []
    id=1
    for data_name in os.listdir(data_path_root):
        if os.path.isdir(os.path.join(data_path_root, data_name)) == True:
            # print(data_name)
            obj = {
                "id": id,
                "name": data_name
            }
            details.append(obj)
            id+=1
    # print(id-1)
    with open("./data_aia.json", "w", encoding='utf8') as file_json:
        json.dump(details, file_json, ensure_ascii=False)
    return details

if __name__ == "__main__":
    data_path_root = '/home/vdungx/Desktop/face-recognition/dataset/processed/'
    get_details(data_path_root)