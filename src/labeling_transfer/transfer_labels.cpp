/*
Main function to detect pedestrians in RGB images using YOLOv3.
*/

#include "labeling_transfer/labeling_transfer.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{

  if ((argc != 3) || ((std::string(argv[1]) != "bbp") && (std::string(argv[1]) != "bbnp")) ||
      ((std::string(argv[2]) != "display") && (std::string(argv[2]) != "save"))) {

    std::cout << "Usage: [" << argv[0] << "] [bbp/bbnp] [display/save]" << std::endl;
    return 0;
  }

  // bbp / bbnp
  std::string type = argv[1];

  // display / save
  std::string option = argv[2];

  // path to images datasets
  std::string media_path = std::string(std::getenv("MEDIA")) + "/image/datasets/";

  // path to bounding boxes maps
  std::string results_maps_path;

  // path to save the 3D clusters after the labeling transfer
  std::string clusters_path;

  // path to save the 2d coordinates of the bounding boxes corresponding to each 3d cluster
  std::string coords2d_path;

  if (type == "bbp") {
    // pedestrian bounding boxes (YOLO detections)
    results_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/results/";

    // pedestrian 3D clusters
    // clusters_path = std::string(std::getenv("MEDIA")) + "/pointcloud/clusters/pedestrian/";
    clusters_path = std::string(std::getenv("MEDIA")) + "/frustum/ply/";

    coords2d_path = std::string(std::getenv("MEDIA")) + "/frustum/coords2d/";
  }

  else if (type == "bbnp") {
    // non-pedestrian bounding boxes
    results_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/bbnp/";

    // non-pedestrian 3D clusters
    clusters_path = std::string(std::getenv("MEDIA")) + "/pointcloud/clusters/not_pedestrian/";
  }

  // path to matching maps (that match each image with its corresponding point cloud)
  std::string matching_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/matching/";

  // path to the projection matrices
  // std::string projection_matrices_path = "../Neofusion_RGB_25_02_2020_INFO.txt";
  std::string projection_matrices_path = "../Neofusion_RGB_25_02_2020_INFO.txt";

  // minimum intersection area between a bounding box and the LIDAR FOV to transfer the bounding box
  float box_intersection_area_threshold = 0.7;

  // minimum YOLO confidence to transfer a bounding box
  float confidence_threshold = 0.8;

  TransferParameters params(type, option, media_path, results_maps_path, matching_maps_path, clusters_path, coords2d_path,
    projection_matrices_path, box_intersection_area_threshold, confidence_threshold);

  // 'display' mode
  if(params.option == "display") {

    // auxiliar parameters to control the display of the detections
    params.display_all = 1;
    params.display_dataset = 1;
    params.display_scene = 1;

    // initialize PCL visualizer, set white background and load parameters
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point cloud viewer"));
    viewer->setBackgroundColor (255, 255, 255);
    viewer->loadCameraParameters("../PCL_Viewer_Parameters.txt");

    transferLabels(params, viewer);
  }

  // 'save' mode
  else if(params.option == "save"){

    // need this "aux" PCL visualizer as an argument, but we aren't displaying anything... (should refactor the functions)
    pcl::visualization::PCLVisualizer::Ptr aux;

    transferLabels(params, aux);
  }

  // show (console) some transfer stats: #clusters with more than 1024 points, max/min #points...
  outputTransferStats(params);

  if(params.option == "save") {
    std::cout << "Pedestrian point cloud clusters saved in: " << params.clusters_path << std::string(2, '\n') << std::flush;
  }

  return 1;
}
