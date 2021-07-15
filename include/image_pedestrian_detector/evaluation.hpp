#pragma once

#include <unordered_map>
#include <vector>

struct EvalParameters
{
  const std::string option; // display/save
  const std::string gt_maps_path, results_maps_path;

  std::string dataset, scene;
  std::string gt_scene_map_filename, results_scene_map_filename, image_filename;
  std::string evaluation_filename, precision_recall_filename;

  int display_scene, display_dataset, display_all, create_new_file;

  float iou_threshold, conf_threshold;

  EvalParameters(const std::string &opt,
                 const std::string &gt_maps_p,
                 const std::string &results_maps_p) :
      option(opt),
      gt_maps_path(gt_maps_p),
      results_maps_path(results_maps_p),
      display_scene(0),
      display_dataset(0),
      display_all(0),
      create_new_file(0)
  {
  }
};

struct Eval
{
  int tp, fp, fn;
  int gt_boxes, results_boxes, results_boxes_above_conf_threshold;
  float precision, recall;
  std::unordered_map<int, std::pair<int, float>> true_positives;
  std::vector<int> false_positives, false_negatives;

  Eval() :
    tp(0),
    fp(0),
    fn(0),
    precision(0.0),
    recall(0.0),
    gt_boxes(0),
    results_boxes(0),
    results_boxes_above_conf_threshold(0)
  {
  }
};

Eval evaluateAll(EvalParameters &params);

Eval evaluateDataset(EvalParameters &params);

Eval evaluateScene(EvalParameters &params);

bool sceneMapExists(EvalParameters &params,
                    const std::string &opt);
