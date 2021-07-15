/*
Utility functions to load and display point clouds.
  - .bin files.
  - .ply files.
  - Folders containing .bin or/and .ply files.
  - Datasets (highway, indoor, outdoor) and scenes (scene1, scene2, ...)
*/

#include "point_cloud_pedestrian_detector/point_cloud_utils.hpp"

#include <iostream>
#include <fstream>
#include <unistd.h>

#include <pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/filesystem.hpp>

// auxiliar parameters to control the display of the point clouds
int display_all = 1;
int display_dataset = 1;
int display_scene = 1;
int display_pcd = 1;

// ----------------------- keyboardEventOccurred -----------------------

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);

  // to control the pressed key
  if(event.keyDown()) {
    std::string key_pressed = event.getKeySym();
    if (key_pressed == "space") { // 'space bar' --> next point cloud
      display_pcd = 0;
    }
    if (key_pressed == "s" || key_pressed == "S") { // 's' --> next scene
      display_scene = 0;
    }
    if (key_pressed == "d" || key_pressed == "D") { // 'd' --> next dataset
      display_dataset = 0;
    }
    if (key_pressed == "Escape") { // 'ESC' --> stop
      display_all = 0;
    }
  }
}

// ----------------------- loadPointCloud -----------------------

void loadPointCloud(const std::string &pcd_path,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &pcd)
{
  try {
    pcl::PointXYZI curr_point;
    int32_t number_of_points, value;

    std::ifstream point_cloud_file(pcd_path, std::ios::binary | std::ios::in);

    point_cloud_file.read(reinterpret_cast<char *>(&number_of_points), sizeof(number_of_points));
    double cnt = 0;
    for(int i=0; i<number_of_points; ++i){
      ++cnt;
      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      // curr_point.x = value/1000.0;
      curr_point.x = value;

      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      // curr_point.y = value/1000.0;
      curr_point.y = value;

      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      // curr_point.z = value/1000.0;
      curr_point.z = value;

      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      curr_point.intensity = value;

      pcd->points.push_back(curr_point);
    }

    // pcl::io::savePLYFile("/home/oscar/Desktop/frustum_beamagine/20200219_174329_871/20200219_174329_907.ply", *pcd);

    return;
  }
  catch(const std::ifstream::failure &e) {
    std::cerr << "ERROR LOADING POINT CLOUD: " << e.what() << '\n';
  }
}

// ----------------------- paintBlackPointCloud -----------------------

void paintBlackPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &pcd,
                          pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcd_RGB)
{
  // "paint" points of a cloud as black
  for (auto & point : pcd->points) {
    pcl::PointXYZRGB point_RGB;
    point_RGB.x = point.x;
    point_RGB.y = point.y;
    point_RGB.z = point.z;
    point_RGB.r = 0;
    point_RGB.g = 0;
    point_RGB.b = 0;
    pcd_RGB->points.push_back(point_RGB);
  }
  return;
}

// ----------------------- displayPointCloud -----------------------

void displayPointCloud(const std::string &pcd_path,
                       pcl::visualization::PCLVisualizer::Ptr &viewer,
                       bool is_bin_file)
{
  display_pcd = 1;

  // load point cloud
  pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZI>);

  if (is_bin_file) {
    loadPointCloud(pcd_path, point_cloud);
  }
  else {
    pcl::io::loadPLYFile(pcd_path, *point_cloud);
    // for (auto & point : point_cloud->points) {
    //   point.x = point.x/1000.0;
    //   point.y = point.y/1000.0;
    //   point.z = point.z/1000.0;
    // }
  }

  // paint black point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB>);
  paintBlackPointCloud(point_cloud, point_cloud2);
  viewer->addPointCloud<pcl::PointXYZRGB>(point_cloud2);

  // viewer->addPointCloud<pcl::PointXYZI>(point_cloud);

  // verbose
  std::cout << "    - POINT CLOUD: " << pcd_path << '\n';
  std::cout << "      * Number of points: " << point_cloud->size() << std::string(2, '\n');

  // wait until 'ESC', 'd', 's' or 'space bar' key is pressed
  while(display_all && display_dataset && display_scene && display_pcd) {
    viewer->spinOnce();
    usleep(100);
  }

  // viewer->saveCameraParameters("/home/oscar/Desktop/qt_visualizer/PCL_Viewer_Parameters.txt");

  viewer->removeAllPointClouds();
  return;
}

// ----------------------- displayScenePointClouds -----------------------

void displayScenePointClouds(const std::string &scene_pcd_path,
                             pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  try {
    display_scene = 1;

    // to store the paths to each point cloud of a scene
    std::vector<boost::filesystem::path> pcd_paths;

    std::copy(boost::filesystem::directory_iterator(scene_pcd_path),
      boost::filesystem::directory_iterator(), std::back_inserter(pcd_paths));

    // sort aphabetically
    std::sort(pcd_paths.begin(), pcd_paths.end());

    // iterate through paths
    for(auto const &pcd_path : pcd_paths) {

      // to control the displaying
      if(display_all && display_dataset && display_scene) {
        displayPointCloud(pcd_path.string(), viewer, true);
      }

      // if 'ESC', 'd' or 's' key is pressed --> return
      else {
        return;
      }
    }
    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- displayDatasetPointClouds -----------------------

void displayDatasetPointClouds(const std::string &dataset_pcd_path,
                               pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  try {
    display_dataset = 1;

    // to store the paths to each scene of a dataset
    std::vector<boost::filesystem::path> scenes_pcd_paths;

    std::copy(boost::filesystem::directory_iterator(dataset_pcd_path),
      boost::filesystem::directory_iterator(), std::back_inserter(scenes_pcd_paths));

    // sort alphabetically (scene1, scene10, scene2, ...)
    std::sort(scenes_pcd_paths.begin(), scenes_pcd_paths.end());

    // iterate through scene paths
    for(auto const &scene_pcd_path : scenes_pcd_paths) {

      // to control the displaying
      if(display_all && display_dataset) {
        std::cout << "  · SCENE: " << scene_pcd_path.filename().string() << std::string(2,'\n');
        displayScenePointClouds(scene_pcd_path.string(), viewer);
      }

      // if 'ESC' or 'd' key is pressed --> return
      else {
        return;
      }
    }
    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- displayAllPointClouds -----------------------

void displayAllPointClouds(const std::string &all_pcd_path,
                           pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  try {
    display_all = 1;

    // to store the paths to each dataset
    std::vector<boost::filesystem::path> datasets_pcd_paths;

    std::copy(boost::filesystem::directory_iterator(all_pcd_path),
      boost::filesystem::directory_iterator(), std::back_inserter(datasets_pcd_paths));

    // sort aphabetically
    std::sort(datasets_pcd_paths.begin(), datasets_pcd_paths.end());

    // iterate through each dataset (highway, indoor, outdoor)
    for(auto const &dataset_pcd_path : datasets_pcd_paths) {

      // to control the displaying
      if(display_all) {
        std::cout << std::endl << "# DATASET: " << dataset_pcd_path.filename().string() << std::string(2, '\n') << std::flush;
        displayDatasetPointClouds(dataset_pcd_path.string(), viewer);
      }

      // if 'ESC' key is pressed --> stop
      else {
        return;
      }
    }
    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- displayFolderPointClouds -----------------------

void displayFolderPointClouds(const std::string &dataset_pcd_path,
                               pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  try {
    // path to the folder
    boost::filesystem::recursive_directory_iterator path_it(dataset_pcd_path);

    // iterate through the folder
    for(auto const & path : path_it) {

      // if it's a regular file --> it's not a folder --> it's a point cloud
      if(boost::filesystem::is_regular_file(path)) {

          if(display_all) {
            std::cout << path.path().string() << '\n';
          // if it's a .bin point_cloud
          if (path.path().string().find(".bin") != std::string::npos) {
            displayPointCloud(path.path().string(), viewer, true); // .bin --> true
          }

          // if it's a .ply point_cloud
          else if (path.path().string().find(".ply") != std::string::npos) {
            displayPointCloud(path.path().string(), viewer, false); // .ply --> false
          }
        }
      }
    }
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}
