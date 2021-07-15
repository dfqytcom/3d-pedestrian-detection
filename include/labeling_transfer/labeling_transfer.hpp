#pragma once

#include <string>
#include <unordered_map>
#include <opencv2/core/types.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

struct TransferParameters
{
  const std::string type, option;
  const std::string media_path, results_maps_path, matching_maps_path, clusters_path, coords2d_path, projection_matrices_path;
  const float box_intersection_area_threshold, confidence_threshold;

  std::vector<int> number_of_points;

  int display_all, display_dataset, display_scene;
  std::string dataset, scene, image_filename;
  std::string image_path, point_cloud_path, results_scene_map_path, matching_map_path;

  cv::Point FOV_top_left, FOV_top_right, FOV_bot_right, FOV_bot_left;

  TransferParameters(const std::string &bb,
                     const std::string &opt,
                     std::string &media_p,
                     std::string &res_maps_p,
                     std::string &match_maps_p,
                     const std::string &clusters_p,
                     const std::string &coords2d_p,
                     const std::string &proj_matrix_p,
                     float box_inters_area_th,
                     float conf_th) :
      type(bb),
      option(opt),
      media_path(media_p),
      results_maps_path(res_maps_p),
      matching_maps_path(match_maps_p),
      clusters_path(clusters_p),
      coords2d_path(coords2d_p),
      projection_matrices_path(proj_matrix_p),
      box_intersection_area_threshold(box_inters_area_th),
      confidence_threshold(conf_th),
      display_all(0),
      display_dataset(0),
      display_scene(0),
      FOV_top_left(628,430),
      FOV_top_right(628,430),
      FOV_bot_right(628,430),
      FOV_bot_left(628,430)
  {
  }
};

void outputTransferStats(TransferParameters &params);

void transferLabels(TransferParameters &params,
                    pcl::visualization::PCLVisualizer::Ptr &viewer);
