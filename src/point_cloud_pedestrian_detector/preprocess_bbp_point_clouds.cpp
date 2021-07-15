/*
Functions to preprocess (center, normalize) pedestrian and non-pedestrian 3D clusters.
*/

#include "utils/maps.hpp"
#include "utils/boxes.hpp"

#include <iostream>
#include <algorithm>
#include <random>

#include <Eigen/Dense>

#include <pcl/common/centroid.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <boost/filesystem.hpp>

// ----------------------- normalizeCloud -----------------------

void normalizeCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud) {
  Eigen::Vector4f cloud_centroid;
  pcl::compute3DCentroid(*cloud, cloud_centroid);

  double furthest_distance = 0;

  for(auto &point : cloud->points) {
    point.x -= cloud_centroid[0];
    point.y -= cloud_centroid[1];
    point.z -= cloud_centroid[2];

    double distance = sqrt(pow(abs(point.x), 2) + pow(abs(point.y), 2) + pow(abs(point.z), 2));
    if(distance > furthest_distance) {
      furthest_distance = distance;
    }
  }

  for(auto &point : cloud->points) {
    point.x /= furthest_distance;
    point.y /= furthest_distance;
    point.z /= furthest_distance;
  }
}

// ----------------------- saveTestDataset -----------------------

void saveTestDataset(std::string & save_path,
                     std::vector<std::string> & test_clouds_paths,
                     std::string & mode)
{
  for(auto & test_cloud_path: test_clouds_paths) {
    std::string aux = test_cloud_path.substr(test_cloud_path.find("scene"));
    std::string current_scene = aux.substr(0, aux.find("/"));
    std::string test_cloud_filename = aux.substr(aux.find("/")+1);

    // load the cloud and normalize it
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(test_cloud_path, *cloud);
    normalizeCloud(cloud);

    // mode = pedestrian / not_pedestrian
    pcl::io::savePLYFile(save_path + "/test/" + mode + "/" + current_scene + "_" + test_cloud_filename, *cloud);
  }
}

// ----------------------- splitTrainingEvalDatasets -----------------------

void splitTrainingEvalDatasets(std::string & save_path,
                               std::vector<std::string> & train_clouds_paths,
                               std::string & mode)
{
  // 80% of the scenes 1-9 to the training dataset, 20% to the validation dataset
  int to_train_clouds = int(train_clouds_paths.size() * 0.8);
  int train_saved_clouds = 0;

  for(auto & train_cloud_path : train_clouds_paths) {
    std::string aux = train_cloud_path.substr(train_cloud_path.find("scene"));
    std::string current_scene = aux.substr(0, aux.find("/"));
    std::string gt_cloud_filename = aux.substr(aux.find("/")+1);

    // load the cloud and normalize it
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile(train_cloud_path, *cloud);
    normalizeCloud(cloud);

    // if < 80% --> train dataset
    if(train_saved_clouds < to_train_clouds) {

      // mode = pedestrian / not_pedestrian
      pcl::io::savePLYFile(save_path + "/train/" + mode + "/" + current_scene + "_" + gt_cloud_filename, *cloud);
      ++train_saved_clouds;
    }

    // if >= 80% --> validation (eval) dataset
    else {
      pcl::io::savePLYFile(save_path + "/eval/" + mode + "/" + current_scene + "_" + gt_cloud_filename, *cloud);
    }
  }
  return;
}

// ----------------------- preprocessBbnpClouds -----------------------

void preprocessBbnpClouds(std::string & base_path,
                          std::string & save_path)
{
  auto rng = std::default_random_engine {};

  std::string clusters_1024_path = base_path + "/not_pedestrian/outdoor";

  std::string scene;

  // test_clouds_paths --> to the test dataset
  // train_clouds_paths --> to the train/validation datasets
  std::vector<std::string> test_clouds_paths, train_clouds_paths;

  boost::filesystem::recursive_directory_iterator clusters_1024_it{clusters_1024_path};

  // iterate through the outdoor clusters folder
  for(auto & it_path : clusters_1024_it) {

    // level 0 --> scene1, scene2, ...
    if(clusters_1024_it.level() == 0) {
      scene = it_path.path().filename().string();
    }

    // level 1 --> clusters
    else if(clusters_1024_it.level() == 1) {

      // e.g. "20200219_183319_804_1.ply"
      std::string image_filename = it_path.path().filename().string().substr(0, it_path.path().filename().string().find("."));

      // if scene10--> test
      if(scene == "scene10") {
        test_clouds_paths.push_back(it_path.path().string());
      }

      // if scene1,2,..,9 --> train/validation
      else {
        train_clouds_paths.push_back(it_path.path().string());
      }
    }
  }

  // shuffle the train/validation clusters
  std::shuffle(train_clouds_paths.begin(), train_clouds_paths.end(), rng);

  std::string mode = "not_pedestrian";

  // save train_gt and train_not_gt not_pedestrian clusters into training and validation datasets
  splitTrainingEvalDatasets(save_path, train_clouds_paths, mode);

  // save test not_pedestrian clusters into test dataset
  saveTestDataset(save_path, test_clouds_paths, mode);

  return;
}

// ----------------------- preprocessBbpClouds -----------------------

void preprocessBbpClouds(std::string & base_path,
                         std::string & save_path)
{
  auto rng = std::default_random_engine {};

  std::string clusters_1024_path = base_path + "/pedestrian/outdoor";

  // to store the paths to the ground truth maps
  std::vector<boost::filesystem::path> gt_maps_paths;

  std::copy(boost::filesystem::directory_iterator(std::string(std::getenv("MEDIA")) + "/image/maps/gt/outdoor"),
    boost::filesystem::directory_iterator(), std::back_inserter(gt_maps_paths));

  // to load the ground truth scene map
  std::unordered_map<std::string, std::vector<Gt_box>> scene_gt_map;

  std::string scene;

  // test_clouds_paths --> to the test dataset
  // train_gt_clouds_paths / train_not_gt_clouds_paths --> to the train/validation datasets
  std::vector<std::string> test_clouds_paths, train_gt_clouds_paths, train_not_gt_clouds_paths;

  boost::filesystem::recursive_directory_iterator clusters_1024_it{clusters_1024_path};

  // iterate through the outdoor clusters folder
  for(auto & it_path : clusters_1024_it) {

    // level 0 --> scene1, scene2, ...
    if(clusters_1024_it.level() == 0) {
      scene = it_path.path().filename().string();

      // to load the corresponding ground truth scene map
      for(auto & gt_map_path : gt_maps_paths) {
        std::string gt_map_filename = gt_map_path.string().substr(gt_map_path.string().find_last_of("/")+1);

        // e.g. if we are preprocessing scene9 --> load scene9_184_gt_map.txt
        if(gt_map_filename.find(scene) != std::string::npos) {
          scene_gt_map = loadGtMap(gt_map_path.string());
          break;
        }
      }
    }

    // level 1 --> clusters
    else if(clusters_1024_it.level() == 1) {
      // e.g. "20200219_183319_804_1.ply"
      std::string image_filename = it_path.path().filename().string().substr(0, it_path.path().filename().string().find("."));

      // if scene10--> test
      if(scene == "scene10") {
        test_clouds_paths.push_back(it_path.path().string());
      }

      // if scene1,2,..,9 --> train/validation
      else {

        // if the image is found inside the gt map (manually labeled) --> train_gt_clouds_paths
        if(scene_gt_map.find(image_filename.substr(0, image_filename.find_last_of("_")) + ".txt") != scene_gt_map.end()) {
          train_gt_clouds_paths.push_back(it_path.path().string());
        }

        // if not (not manually labeled) --> train_not_gt_clouds_paths
        else {
          train_not_gt_clouds_paths.push_back(it_path.path().string());
        }
      }
    }
  }

  // shuffle the train_gt and train_not_gt clouds to mix everything
  std::shuffle(train_gt_clouds_paths.begin(), train_gt_clouds_paths.end(), rng);
  std::shuffle(train_not_gt_clouds_paths.begin(), train_not_gt_clouds_paths.end(), rng);

  std::string mode = "pedestrian";

  // save train_gt and train_not_gt pedestrian clusters into training and validation datasets
  splitTrainingEvalDatasets(save_path, train_gt_clouds_paths, mode);
  splitTrainingEvalDatasets(save_path, train_not_gt_clouds_paths, mode);

  // save test pedestrian clusters into test dataset
  saveTestDataset(save_path, test_clouds_paths, mode);

  return;
}

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  if ((argc != 2) ||
      ((std::string(argv[1]) != "bbp") &&
        std::string(argv[1]) != "bbnp")) {
    std::cout << "Usage: [" << argv[0] << "] [bbp/bbnp]" << std::endl;
    return 0;
  }

  // bbp / bbnp
  std::string mode = argv[1];

  // path to the clusters that have to be preprocessed
  std::string base_path = std::string(std::getenv("MEDIA")) + "/pointcloud/clusters_1024/";

  // path to save the preprocessed clusters
  std::string save_path = std::string(std::getenv("MEDIA")) + "/pointcloud/clusters_1024_norm";

  if (mode == "bbp") {
    preprocessBbpClouds(base_path, save_path);
  }
  else if (mode == "bbnp") {
    preprocessBbnpClouds(base_path, save_path);
  }
  return 1;
}
