import os
import math
import pickle
import numpy as np

import downsample_bbp_frustum as utils

def get_frustum_angle(coords2d):
    [tlx,_,brx,_] = coords2d
    cx_bb = (brx+tlx)//2

    fx = 2457.005064
    cx = 615.208515

    if  cx_bb > cx:
        frustum_angle = -1 * math.atan(fx/(cx_bb-cx))
    elif cx_bb < cx:
        frustum_angle = -1 * (math.pi - math.atan(fx/(cx-cx_bb)))

    return frustum_angle

if __name__ == '__main__':

    media_path = os.environ['MEDIA']
    coords2d_path = os.path.join(media_path, 'frustum/coords2d/outdoor')
    frustums_path = os.path.join(media_path, 'frustum/ply_1024/outdoor')

    id_path = os.path.join(media_path, 'frustum/files/val.txt')
    image_id_path = os.path.join(media_path, 'frustum/files/image_id.txt')

    id_file = open(id_path, 'a+')
    image_id_file = open(image_id_path, 'a+')

    id_list = []
    box2d_list = []
    input_list = [] # frustum points (XYZ)
    type_list = []
    frustum_angle_list = [] # angle of 2d box center from pos x-axis
    prob_list = []

    id = -1
    last_image_id = ''

    for frustums_scene in sorted(os.listdir(frustums_path)):
        frustums_scene_path = os.path.join(frustums_path, frustums_scene)
        for frustum_filename in sorted(os.listdir(frustums_scene_path)):

            coords2d_filename = os.path.join(coords2d_path, frustums_scene, frustum_filename.split('.')[0] + '.txt')

            with open(coords2d_filename, 'rb') as f:
                coords2d = f.read()

            coords2d = np.asarray(coords2d.decode("utf-8").split(' '), dtype=np.float32)

            image_id = '_'.join(frustum_filename.split('_')[:-1])
            if last_image_id != image_id:
                last_image_id = image_id
                id += 1
                id_file.write(f'{id:06}\n')
                image_id_file.write("%d %s\n" % (id, os.path.join(frustums_scene, image_id)))

            id_list.append(id)
            box2d_list.append(coords2d)
            input_list.append(utils.read_ply(os.path.join(frustums_scene_path, frustum_filename)))
            type_list.append('Pedestrian')
            frustum_angle_list.append(get_frustum_angle(coords2d))
            prob_list.append(0.99)

    with open(os.path.join(media_path, 'frustum/files/frustum_beamagine.pickle'),'wb+') as fp:
        pickle.dump(id_list, fp, protocol=2)
        pickle.dump(box2d_list,fp, protocol=2)
        pickle.dump(input_list, fp, protocol=2)
        pickle.dump(type_list, fp, protocol=2)
        pickle.dump(frustum_angle_list, fp, protocol=2)
        pickle.dump(prob_list, fp, protocol=2)
