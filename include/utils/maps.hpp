#pragma once

#include "utils/boxes.hpp"

#include <unordered_map>
#include <vector>

struct MapsParameters {
  const std::string option;
  const std::string base_path;
  std::string dataset_path;
  std::string scene_path;
  std::string maps_path;
  std::string dataset, scene, filename;

  MapsParameters(const std::string &opt,
                 const std::string &data_path);
};

void generateGtMaps();

void generateResultsMaps();

void generateMatchingMaps();

std::unordered_map<std::string, std::vector<Gt_box>> loadGtMap(const std::string &gt_map_filename);

std::unordered_map<std::string, std::vector<Results_box>> loadResultsMap(const std::string &results_map_filename);

std::pair<std::unordered_map<std::string, std::vector<Gt_box>>,
          std::unordered_map<std::string, std::vector<Results_box>>>
    loadMaps(const std::string &gt_scene_map_path, 
             const std::string &results_scene_map_path);

std::unordered_map<std::string, std::string> loadMatchingMap(const std::string &matching_map_filename);
