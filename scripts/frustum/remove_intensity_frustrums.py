import numpy as np
import sys
import os
from plyfile import PlyData, PlyElement

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)
    points = np.asarray(plydata['vertex'].data.tolist())
    return points

def write_ply(points,output_file):

    with open(output_file, 'w') as ply:
        header = ['ply']
        header.append('format ascii 1.0')
        header.append('element vertex {}'.format(points.shape[0]))
        header.append('property float x')
        header.append('property float y')
        header.append('property float z')
        # header.append('property float intensity')
        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

        for p in points:
            ply.write("{} {} {} {}\n".format(p[0], p[1], p[2]))

    return

def process_point_cloud(input_cluster_path, output_cluster_path):

    pcd = read_ply(input_cluster_path)
    write_ply(pcd, output_cluster_path)

    return

if __name__ == '__main__':

    # Path to input folder
    input_clusters_dir = sys.argv[1]

    # Path to output folder
    output_clusters_dir = sys.argv[2]

    # Iterate input folder
    for subdir, dirs, files in os.walk(input_clusters_dir):
        for file in files:
            if '.ply' in file:
                input_cluster_path = os.path.join(subdir,file)
                output_cluster_path = os.path.join(output_clusters_dir, file)

                process_point_cloud(input_cluster_path, output_cluster_path)
