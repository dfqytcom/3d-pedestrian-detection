#pragma once

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/filesystem.hpp>

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void);

void loadPointCloud(const std::string &point_cloud_filename,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &pcdPtr);

void paintBlackPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &pcd,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcd_RGB);

void displayPointCloud(const std::string &point_cloud_path,
                       pcl::visualization::PCLVisualizer::Ptr &viewer,
                       bool is_bin_file);

void displayScenePointClouds(const std::string &scene_point_clouds_path,
                             pcl::visualization::PCLVisualizer::Ptr &viewer);

void displayDatasetPointClouds(const std::string &dataset_point_clouds_path,
                               pcl::visualization::PCLVisualizer::Ptr &viewer);

void displayAllPointClouds(const std::string &point_clouds_path,
                           pcl::visualization::PCLVisualizer::Ptr &viewer);

void displayFolderPointClouds(const std::string &dataset_pcd_path,
                              pcl::visualization::PCLVisualizer::Ptr &viewer);