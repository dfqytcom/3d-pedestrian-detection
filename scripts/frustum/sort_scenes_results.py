import os
import shutil
import numpy as np

if __name__ == '__main__':

    all_results_path = '/home/oscar/media/frustum/results/all'
    scenes_results_path = '/home/oscar/media/frustum/results/scenes'

    image_id_path = '/home/oscar/media/frustum/files/image_id.txt'

    with open(image_id_path, 'r') as f:
        image_ids = f.read().split('\n')[:-1]

    image_ids_dict = {}

    for i in image_ids:
        id, scene_image = i.split(' ')
        image_ids_dict[int(id)] = scene_image

    for filename in sorted(os.listdir(all_results_path)):
        id = int(filename.split('.')[0])
        new_path = image_ids_dict[id] + '.txt'
        shutil.copy(os.path.join(all_results_path, filename),
                    os.path.join(scenes_results_path, new_path))
