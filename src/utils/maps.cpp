/*
Utility functions to manage, serialize (save) and deserialize (load) maps:
  - Gt       --> ground truth bounding boxes manually labeled.
  - Results  --> YOLO detections bounding boxes
  - Matching --> images paired with their corresponding point clouds.
*/

#include "utils/maps.hpp"
#include "utils/boxes.hpp"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/base_object.hpp>

MapsParameters::MapsParameters(const std::string &opt,
                               const std::string &data_path) :
        option(opt),
        base_path(data_path)
{
}

//--------------------------------- Free functions ---------------------------------

// ----------------------- addGtMapEntry -----------------------

int addGtMapEntry(const MapsParameters &params,
                  std::unordered_map<std::string,
                  std::vector<Gt_box>> &gt_map)
{
  std::string line;
  std::vector<std::string> lineSplit;
  std::vector<Gt_box> boxes;

  // file containing the ground truth bounding boxes of an image
  std::ifstream file(params.scene_path + "/" + params.filename);

  // one line = one bounding box
  while(std::getline(file, line)){
    boost::split(lineSplit, line, boost::is_any_of(" "));

    // create a ground truth box using the values read from the file
    Gt_box box(std::stoi(lineSplit[0]),  // classID
               std::stof(lineSplit[1]),  // center_x
               std::stof(lineSplit[2]),  // center_y
               std::stof(lineSplit[3]),  // width
               std::stof(lineSplit[4])); // height
    boxes.push_back(box);
  }

  // store the gt bounding boxes into the gt map
  gt_map[params.filename] = boxes;
  return boxes.size();
}

// ----------------------- addResultsMapEntry -----------------------

int addResultsMapEntry(const MapsParameters &params,
                       std::unordered_map<std::string,
                       std::vector<Results_box>> &results_map)
{
  std::string line;
  std::vector<std::string> lineSplit;
  std::vector<Results_box> boxes;

  // file containing the YOLO detections bounding boxes of an image
  std::ifstream file(params.scene_path + "/" + params.filename);

  // one line = one bounding box
  while(std::getline(file, line)){
    boost::split(lineSplit, line, boost::is_any_of(" "));

    // create a results box using the values read from the line
    Results_box box(std::stoi(lineSplit[0]),  // classID
                    std::stof(lineSplit[1]),  // center_x
                    std::stof(lineSplit[2]),  // center_y
                    std::stof(lineSplit[3]),  // width
                    std::stof(lineSplit[4]),  // height
                    std::stof(lineSplit[5])); // confidence
    boxes.push_back(box);
  }

  // store the results bounding boxes into the results map
  results_map[params.filename] = boxes;
  return boxes.size();
}

// ----------------------- generateGtSceneMap -----------------------

void generateGtSceneMap(MapsParameters &params)
{
  try {
    std::unordered_map<std::string, std::vector<Gt_box>> gt_map;
    boost::filesystem::directory_iterator it{params.scene_path};

    // to store the number of gt bounding boxes saved in a map
    int scene_boxes = 0;

    // iterate through each image of a scene and save the gt bounding boxes into a gt map
    while(it != boost::filesystem::directory_iterator{}) {
      params.filename = (*it++).path().filename().string();
      scene_boxes += addGtMapEntry(params, gt_map);
    }

    // output file to store the gt map --> serialize
    std::string gt_map_path = params.maps_path + params.scene + "_" + std::to_string(scene_boxes) + "_" + params.option + "_map.txt";
    std::ofstream gt_map_file{gt_map_path};
    boost::archive::text_oarchive oa{gt_map_file};
    oa << gt_map;

    std::cout << "OK!  " << params.scene << "_" << scene_boxes << "_" << params.option <<  "_map.txt created (" << gt_map_path << ")" << std::endl;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }

  return;
}

// ----------------------- generateResultsSceneMap -----------------------

void generateResultsSceneMap(MapsParameters &params)
{
  try {
    std::unordered_map<std::string, std::vector<Results_box>> results_map;
    boost::filesystem::directory_iterator it{params.scene_path};

    // to store the number of results bounding boxes saved in a map
    int scene_boxes = 0;

    // iterate through each image of a scene and save the results bounding boxes into a results map
    while(it != boost::filesystem::directory_iterator{}) {
      params.filename = (*it++).path().filename().string();
      scene_boxes += addResultsMapEntry(params, results_map);
    }

    // output file to store the results map --> serialize
    std::string results_map_path = params.maps_path + params.scene + "_" + std::to_string(scene_boxes) + "_" + params.option + "_map.txt";
    std::ofstream results_map_file{results_map_path};
    boost::archive::text_oarchive oa{results_map_file};
    oa << results_map;

    std::cout << "OK!  " << params.scene << "_" << scene_boxes << "_" << params.option <<  "_map.txt created (" << results_map_path << ")" << std::endl;

    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- generateMatchingSceneMap -----------------------

void generateMatchingSceneMap(MapsParameters &params)
{
  try {
    // path to the folder containing the point clouds of a specific scene
    std::string pcd_scene_path = std::string(std::getenv("MEDIA")) + "/pointcloud/datasets/" + params.dataset + "/" + params.scene;

    // path to save the matching map
    std::string matching_map_path = params.maps_path + params.scene + "_" + params.option + "_map.txt";

    std::unordered_map<std::string, std::string> matching_map;
    boost::filesystem::directory_iterator image_it{params.scene_path};

    // iterate through scene images
    while (image_it != boost::filesystem::directory_iterator{}) {
      std::string image_path = (*image_it).path().string();

      // e.g. "20200110_135543_421.png"
      std::string image_filename = (*image_it++).path().filename().string();

      // minimum difference between image/pointcloud time stamps to match them
      int32_t min_diff = 200; // ms

      bool match = false;
      std::string pcd_candidate;

      std::vector<std::string> image_filename_split;
      boost::split(image_filename_split, image_filename, boost::is_any_of("_"));

      // e.g. "20200110135543421"
      long image_timestamp = stol(boost::join(image_filename_split, ""));

      boost::filesystem::directory_iterator pcd_it{pcd_scene_path};

      // iterate through the point clouds
      while(pcd_it != boost::filesystem::directory_iterator{})
      {
        // e.g. "20200110_135543_434.bin"
        std::string pcd_filename = (*pcd_it).path().filename().string();

        std::vector<std::string> pcd_filename_split;
        boost::split(pcd_filename_split, pcd_filename, boost::is_any_of("_"));

        // e.g. "20200110135543434"
        long pcd_timestamp = stol(boost::join(pcd_filename_split, ""));

        // e.g. "20200110135543434" - "20200110135543421" = "13" ms
        int32_t time_diff = abs(image_timestamp-pcd_timestamp);

        // we store the point cloud with the minimum difference (and < 200ms)
        if(time_diff < min_diff)
        {
          match = true;
          min_diff = time_diff;
          pcd_candidate = (*pcd_it).path().string();
        }
        *pcd_it++;
      }

      // if a match is found --> pair the image and the point cloud and save them into the matching map
      if (match) {
        matching_map[image_path] = pcd_candidate;
      }
    }

    // output file to store the matching map --> serialize
    std::ofstream matching_map_file{matching_map_path};
    boost::archive::text_oarchive oa{matching_map_file};
    oa << matching_map;

    std::cout << "OK!  " << params.scene << "_" << params.option <<  "_map.txt created (" << matching_map_path << ")" << std::endl;

    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- generateSceneMap -----------------------

void generateSceneMap(MapsParameters &params)
{
  if(params.option == "gt") {
    generateGtSceneMap(params);
  }
  else if(params.option == "results") {
    generateResultsSceneMap(params);
  }
  else if(params.option == "matching") {
    generateMatchingSceneMap(params);
  }
  return;
}

// ----------------------- generateDatasetMaps -----------------------

void generateDatasetMaps(MapsParameters &params)
{
  std::cout << std::endl << "----------------------" << params.option << ": " << params.dataset << " ---------------------- " << std::endl << std::endl;

  try {

    // path to scenes folders
    std::vector<boost::filesystem::path> scenes_paths;

    std::copy(boost::filesystem::directory_iterator(params.dataset_path),
      boost::filesystem::directory_iterator(), std::back_inserter(scenes_paths));

    // sort alphabetically (scene1, scene10, scene2, ...)
    std::sort(scenes_paths.begin(), scenes_paths.end());

    for(auto const &scene_path : scenes_paths) {
      params.scene_path = scene_path.string();

      // scene# (1, 2, ...)
      params.scene = scene_path.filename().string();

      generateSceneMap(params);
    }
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
  return;
}

// ----------------------- generateMaps -----------------------

void generateMaps(MapsParameters &params)
{
  try {
    // check if maps already exist
    std::string maps_folder = std::string(std::getenv("MEDIA")) + "/image/maps/" + params.option + "/";
    if(boost::filesystem::is_directory(maps_folder)) {
      std::cout << '\n' << "WARNING: All previous maps in " << maps_folder << " folders will be removed, are you sure? (y/n): ";
      std::string remove;
      std::cin >> remove;

      if(remove != "y" && remove != "Y") {
        std::cout << "OPERATION ABORTED" << '\n';
        return;
      }
    }

    // path to dataset (highway, indoor, outdoor) folders
    std::vector<boost::filesystem::path> datasets_paths;

    std::copy(boost::filesystem::directory_iterator(params.base_path),
      boost::filesystem::directory_iterator(), std::back_inserter(datasets_paths));

    // sort them alphabetically
    std::sort(datasets_paths.begin(), datasets_paths.end());

    for(auto const &dataset_path : datasets_paths) {
      params.dataset_path = dataset_path.string();

      // highway, indoor or outdoor
      params.dataset = dataset_path.filename().string();

      // path to save the maps
      params.maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/" + params.option + "/" + params.dataset + "/";

      // remove the existing maps
      if (boost::filesystem::is_directory(params.maps_path)) {
        boost::filesystem::remove_all(params.maps_path);
      }
      boost::filesystem::create_directories(params.maps_path);
      std::cout << std::string(2, '\n') << params.maps_path << " directory created" << std::string(2, '\n');

      generateDatasetMaps(params);
    }
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
  return;
}

//--------------------------------- maps.hpp ---------------------------------

// ----------------------- generateGtMaps -----------------------

void generateGtMaps()
{
  // path to the dataset folders (highway, indoor, outdoor) containing the
  // ground truth bounding boxes of each dataset and scene
  std::string base_path = std::string(std::getenv("MEDIA")) + "/image/gt/";

  MapsParameters gt_params("gt", base_path);
  generateMaps(gt_params);
  return;
}

// ----------------------- generateResultsMaps -----------------------

void generateResultsMaps()
{
  // path to the dataset folders (highway, indoor, outdoor) containing the
  // results bounding boxes (YOLO detections) of each dataset and scene
  std::string base_path = std::string(std::getenv("MEDIA")) + "/image/results/";

  MapsParameters results_params("results", base_path);
  generateMaps(results_params);
  return;
}

// ----------------------- generateMatchingMaps -----------------------

void generateMatchingMaps()
{
  // path to the dataset folders (highway, indoor, outdoor) containing the .png images
  std::string base_path = std::string(std::getenv("MEDIA")) + "/image/datasets/";

  MapsParameters matching_params("matching", base_path);
  generateMaps(matching_params);
  return;
}

// ----------------------- loadGtMap -----------------------

std::unordered_map<std::string, std::vector<Gt_box>> loadGtMap(const std::string &gt_map_filename)
{
  try {
    // deserialize a ground truth map and return it
    std::ifstream gt_map_file(gt_map_filename);
    boost::archive::text_iarchive ia{gt_map_file};
    std::unordered_map<std::string, std::vector<Gt_box>> gt_map;
    ia >> gt_map;
    return gt_map;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- loadResultsMap -----------------------

std::unordered_map<std::string, std::vector<Results_box>> loadResultsMap(const std::string &results_map_filename)
{
  try {
    // deserialize a results map and return it
    std::ifstream results_map_file(results_map_filename);
    boost::archive::text_iarchive ia{results_map_file};
    std::unordered_map<std::string, std::vector<Results_box>> results_map;
    ia >> results_map;
    return results_map;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- loadMaps -----------------------

std::pair<std::unordered_map<std::string, std::vector<Gt_box>>,
          std::unordered_map<std::string, std::vector<Results_box>>>
        loadMaps(const std::string &gt_scene_map_path,
                 const std::string &results_scene_map_path)
{
  // deserialize a ground truth and a results map and return them
  auto gt_map = loadGtMap(gt_scene_map_path);
  auto results_map = loadResultsMap(results_scene_map_path);

  return std::make_pair(gt_map, results_map);
}

// ----------------------- loadMatchingMap -----------------------

std::unordered_map<std::string, std::string> loadMatchingMap(const std::string &matching_map_path)
{
  try {
    // deserialize a matching map and return it
    std::ifstream matching_map_file(matching_map_path);
    boost::archive::text_iarchive ia{matching_map_file};
    std::unordered_map<std::string, std::string> matching_map;
    ia >> matching_map;
    return matching_map;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}
