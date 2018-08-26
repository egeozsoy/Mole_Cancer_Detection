import requests
from PIL import Image
import io
import os

# https://isic-archive.com/api/v1
IMAGE_LIST_LINK = 'https://isic-archive.com/api/v1/image?limit={}&offset={}&sort=name&sortdir=1&detail=true'  # can be used with detail=true to eliminate need for meta_link
IMAGE_DOWNLOAD_LINK = 'https://isic-archive.com/api/v1/image/{}/download'  # {} = id of image
IMAGE_META_LINK = 'https://isic-archive.com/api/v1/image/{}'  # {} = id of image
IMAGE_SAVE_PATH = '{}/images/'.format(os.path.dirname(os.path.abspath(__file__))) #Get current path
LABEL_SAVE_PATH = '{}/labels/'.format(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    offset_step = 2500
    start_from = 0

    benign_malignant_counter = {
        'benign': 1,
        'malignant': 1
    }

    for i in range(start_from, 30000, offset_step):
        response = requests.get(IMAGE_LIST_LINK.format(offset_step, i))
        image_data_list = response.json()

        print('Starting proceessing with offset: {}'.format(i))
        first_time = True
        for img_data in image_data_list:
            try:
                img_id = img_data['_id']
                mole_type = img_data['meta']['clinical']['benign_malignant']
                number = benign_malignant_counter[mole_type]

                # sample 2500 images from each category
                if number > 2500:
                    continue

                image_bytes = requests.get(IMAGE_DOWNLOAD_LINK.format(img_id))
                image = Image.open(io.BytesIO(image_bytes.content))
                name = '{type:}_{num:05d}'.format(type=mole_type, num=benign_malignant_counter[mole_type])
                image_save_path = '{}{}.jpg'.format(IMAGE_SAVE_PATH, name)
                image.save(image_save_path)
                if first_time:
                    print('First image of offset is called {}'.format(name))
                    first_time = False

                benign_malignant_counter[mole_type] = number + 1
            except Exception as e:
                print(e)
                continue
