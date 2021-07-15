"""
Function to downsample non-pedestrian bounding boxes using the Farthest Point Sampling algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import time
import sys
import math
import random
import os
from random import randrange
from matplotlib import interactive
from plyfile import PlyData, PlyElement
interactive(True)

pi = math.pi

def makeCircleDemo(number_of_points, number_of_downsampled_points):
    circle = np.asarray(create_circle_points(number_of_points))

    plt.scatter(circle[:,0], circle[:,1])

    print('\n------- {} points circle'.format(number_of_points))
    input('\nPress Enter to continue...\n')

    points = downsample(circle, number_of_downsampled_points)

    for point in circle[points]:
        plt.plot(point[0], point[1], 'ro', markersize=12)

    print('\n------- {} (red) points circle'.format(points.shape[0]))
    input('\nPress Enter to end...\n')

def create_circle_points(n):
    return [(math.cos(2*pi/n*x),math.sin(2*pi/n*x)) for x in range(0,n+1)]

def initializeSubset(X, K, distances_matrix, method = "K_random", first_point_init = None):
    # K points of the downsampled dataset (iteration #0) --> Random
    N = X.shape[0]
    if method == "K_random":
        points = np.random.permutation(np.arange(N))
        points = points[:K]
        first_point = points[0]
        distances = distances_matrix[first_point, :]
        closest_points = np.empty(N)
        closest_points.fill(first_point)
        # Update distances and closest_points with the K points of the downsampled dataset
        for i in points[1:]:
            old_distances = distances
            distances = np.minimum(distances, distances_matrix[i, :])
            changes = np.not_equal(old_distances, distances)
            closest_points[np.where(changes == True)] = i

    # K points of the downsampled dataset (iteration #0) --> Furthest Point Sampling
    elif method == "furthest":
        # First point of the downsampled dataset: X[0]
        if first_point_init == "first":
            points = np.zeros(K, dtype=np.int64)
            distances = distances_matrix[0, :]
            closest_points = np.zeros(N)

        # First point of the downsampled dataset: X[randrange(N)]
        elif first_point_init == "rand":
            rand_point = randrange(N)
            points = np.zeros(K, dtype=np.int64)
            distances = distances_matrix[rand_point, :]
            closest_points = np.empty(N)
            closest_points.fill(rand_point)
            points[0] = rand_point

        else:
            return

        for i in range(1, K):
            furthest_point = np.argmax(distances)
            points[i] = furthest_point
            old_distances = distances
            distances = np.minimum(distances, distances_matrix[furthest_point, :])
            changes = np.not_equal(old_distances, distances)
            closest_points[np.where(changes)] = furthest_point
    else:
        return

    return [points, distances, closest_points]

def downsample(X, K):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    points : ndarray (K)
        Indices of the K resulting points
    """

    distances_matrix = pairwise_distances(X, metric='euclidean')

    [points, distances, closest_points] = initializeSubset(X, K, distances_matrix)

    iter = 0
    count = 0
    convergence = False

    # Until convergence --> distances don't change during a whole iteration (through the K points of the downsampled dataset)
    while True:
        iter += 1
        print('------- ITERATION #{}'.format(iter))
        old_distances = distances

        for i, point in enumerate(points):
            closest_indices = np.where(closest_points == point)[0]
            valid = False
            for cl_i in closest_indices:
                if cl_i not in points:
                    valid = True
                    break
            if valid == True:
                new_closest_index_point_distance = []
                for closest_index in closest_indices:
                    sorted_point_distances = np.argsort(distances_matrix[closest_index,:])
                    # Excluding the closest point ("point")
                    sorted_point_distances = sorted_point_distances[sorted_point_distances != point]
                    for new_closest_point in sorted_point_distances:
                        if new_closest_point in points:
                            break
                    new_closest_index_point_distance.append([closest_index, new_closest_point, distances_matrix[closest_index, new_closest_point]])

                new_closest_index_point_distance = np.asarray(new_closest_index_point_distance)
                furthest_point = int(new_closest_index_point_distance[np.argmax(new_closest_index_point_distance[:,2]), 0])

                if point != furthest_point:
                    points[i] = furthest_point
                    aux_to_update_distances = np.unique(new_closest_index_point_distance[:,1]).astype(int)
                    aux_to_update_distances = np.append(aux_to_update_distances, furthest_point)
                    distances[np.where(closest_points == point)] = float("inf")
                    for index in aux_to_update_distances:
                        old_distances = distances
                        distances = np.minimum(distances, distances_matrix[index, :])
                        changes = np.not_equal(old_distances, distances)
                        closest_points[np.where(changes)] = index
                else:
                    count += 1

        changes = np.not_equal(old_distances, distances)
        if not np.any(changes):
            break

    print('\n******* #ITERATIONS UNTIL CONVERGENCE: {}'.format(iter))
    return points

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0].data
    points = np.delete(np.asarray(vertex.tolist()),3,1)
    return points

def write_ply(points,output_file):

    with open(output_file, 'w') as ply:
        header = ['ply']
        header.append('format ascii 1.0')
        header.append('element vertex {}'.format(points.shape[0]))
        header.append('property float x')
        header.append('property float y')
        header.append('property float z')
        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

        for p in points:
            ply.write("{} {} {}\n".format(p[0], p[1], p[2]))

    return

def processPointCloud(cluster_path, cluster_1024_path):

    pcd = read_ply(cluster_path)
    print('\nLoading point cloud from {}'.format(cluster_path))

    points = pcd.shape[0]
    print('\nNumber of points (pre-downsampling): {}\n'.format(points))

    if points > 1024 * 1.05: # 5% (maybe 10%?)
        distances_matrix = pairwise_distances(pcd, metric='euclidean')
        [points_1024,_,_] = initializeSubset(pcd, 1024, distances_matrix, method = "furthest", first_point_init = "rand")
        pcd_1024 = pcd[points_1024]

    elif points > 1024 and points <= 1024 * 1.05:
        points_rnd = np.random.permutation(np.arange(points))
        points_1024 = points_rnd[:1024]
        pcd_1024 = pcd[points_1024]

    elif points == 1024:
        pcd_1024 = pcd

    if pcd_1024.shape[0] != 1024 or points < 1024:
        print("ERROR --> #points_1024: {} , #points: {}".format(pcd_1024.shape[0], points))
        exit()

    write_ply(pcd_1024, cluster_1024_path)

    return

if __name__ == '__main__':
    clusters_dir = os.environ['MEDIA'] + '/pointcloud/clusters/not_pedestrian/outdoor/'
    for subdir, dirs, files in os.walk(clusters_dir):
        for file in files:
            cluster_path = os.path.join(subdir,file)
            aux_path = cluster_path.split('/clusters/')
            cluster_1024_path = os.path.join(aux_path[0], os.path.join('clusters_1024', aux_path[1]))
            processPointCloud(cluster_path, cluster_1024_path)

    # makeCircleDemo(100, 20)
