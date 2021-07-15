/*
Main function to display point clouds.
  - .bin files.
  - .ply files.
  - Folders containing .bin or/and .ply files.
  - Datasets (highway, indoor, outdoor) and scenes (scene1, scene2, ...)
*/

#include "point_cloud_pedestrian_detector/point_cloud_utils.hpp"

#include <iostream>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/filesystem.hpp>

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  if (argc > 3) {
    std::cout << "Usage: [" << argv[0] << "] <path_to_folder/path_to_point_cloud/dataset> <scene>" << std::endl;
    return 0;
  }

  // path to the datasets folders (highway, indoor, outdoor)
  std::string path_to_point_clouds = std::string(std::getenv("MEDIA")) + "/pointcloud/datasets/";

  // PCL visualizer parameters
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("Point cloud viewer"));
  viewer->addCoordinateSystem();
  viewer->setBackgroundColor(255, 255, 255); // white background
  // viewer->loadCameraParameters("../PCL_Viewer_Parameters.txt");
  viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

  switch (argc) {
    case 1:
    {
      displayAllPointClouds(path_to_point_clouds, viewer);
      break;
    }

    case 2:
    {
      std::string s = argv[1];

      // if it's a path to a point_cloud (.bin)
      if (std::string(s).find(".bin") != std::string::npos) {
        displayPointCloud(s, viewer, true); // .bin --> true
      }

      // if it's a path to a point_cloud (.ply)
      else if (std::string(s).find(".ply") != std::string::npos) {
        displayPointCloud(s, viewer, false); // .ply --> false
      }

      // specific dataset
      else if ((s == "highway") ||
               (s == "indoor")  ||
               (s == "outdoor")) {
        displayDatasetPointClouds(path_to_point_clouds + s, viewer);
      }

      // if it's a path to a folder
      else {
        displayFolderPointClouds(s, viewer);
      }
      break;
    }

    case 3:
    {
      std::string d = argv[1];

      if ((d == "highway") ||
          (d == "indoor")  ||
          (d == "outdoor")) {

        displayScenePointClouds(path_to_point_clouds + d + "/" + argv[2], viewer);
      }
      else {
        std::cerr << "No dataset named: " << d << '\n';
      }
      break;
    }
  }
  return 1;
}
