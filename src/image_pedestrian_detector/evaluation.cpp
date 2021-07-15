/*
Auxiliar functions to evaluate YOLO detections.

Main: evaluate_results.cpp
*/

#include "image_pedestrian_detector/evaluation.hpp"
#include "utils/maps.hpp"
#include "utils/boxes.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <vector>

#include <boost/filesystem.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

//--------------------------------- Free functions ---------------------------------

// ----------------------- drawBoxes -----------------------

void drawBoxes(cv::Mat &image,
               const std::pair<std::vector<Gt_box>,
               std::vector<Results_box>> &gt_and_results_boxes,
               const Eval &image_eval)
{
  std::vector<Gt_box> gt_boxes = gt_and_results_boxes.first;
  std::vector<Results_box> results_boxes = gt_and_results_boxes.second;

  for(auto const tp_index : image_eval.true_positives) {
    results_boxes[tp_index.first].drawBox(image, true);
    gt_boxes[tp_index.second.first].drawBox(image, true);
  }

  for(auto const fp_index : image_eval.false_positives) {
    results_boxes[fp_index].drawBox(image, false);
  }

  for(auto const fn_index : image_eval.false_negatives) {
    gt_boxes[fn_index].drawBox(image, false);
  }

  return;
}

// ----------------------- displayImage -----------------------

void displayImage(EvalParameters &params,
                  const Eval &image_eval,
                  const std::pair<std::vector<Gt_box>,
                  std::vector<Results_box>> &gt_and_results_boxes)
{
  std::string image_path = std::string(std::getenv("MEDIA")) + "/image/datasets/" + params.dataset +
    "/" + params.scene + "/" + params.image_filename.substr(0, params.image_filename.find(".")) + ".png";

  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    std::cout << std::endl << "WARNING: No such image " << image_path << std::endl;
    return;
  }

  drawBoxes(image, gt_and_results_boxes, image_eval);

  // set evaluation results (tp, fp, fn, precision, recall) and the image filename as the title of the figure
  std::stringstream figure_title_aux;
  figure_title_aux << std::fixed << std::setprecision(2) << "TP: " << image_eval.tp << "  FP: " << image_eval.fp << "  FN: " <<
    image_eval.fn << "  Precision: " << image_eval.precision << "  Recall: " << image_eval.recall << "  Image: " << image_path;

  std::string figure_title = figure_title_aux.str();
  cv::namedWindow(figure_title, cv::WINDOW_NORMAL);
  cv::resizeWindow(figure_title, 1920, 1080);
  cv::imshow(figure_title, image);
  // cv::imwrite("example.png", image); // uncomment to save the image with the evaluation boxes

  // wait until 'ESC', 'd', 's' or 'space bar' key is pressed
  int key_pressed;
  do {
    key_pressed = cv::waitKey();
  }
  while(key_pressed != 32 && key_pressed != 27 && key_pressed != 100 && key_pressed != 115);
  cv::destroyAllWindows();

  if (key_pressed == 27) { // ESC --> stop
    params.display_all = 0;
  }
  else if (key_pressed == 100) { // d --> change dataset
    params.display_dataset = 0;
  }
  else if (key_pressed == 115) { // s --> change scene
    params.display_scene = 0;
  }
  return;
}

// ----------------------- outputEvaluation -----------------------

void outputEvaluation(const EvalParameters &params,
                      const Eval &eval,
                      int level)
{
  std::string s = "(" + std::to_string(eval.results_boxes) + ")";
  switch (level) {

    // first line: style stuff
    case 0:
      std::cout << std::string(150, '-') << std::endl;
      std::cout << std::left << std::setw(15) << params.dataset << std::setw(15) << "gt_boxes" << std::setw(40) <<
      "results_boxes_above_conf_threshold" << std::setw(15) << "(total)" << std::setw(10) << "tp" << std::setw(10) <<
      "fp" << std::setw(10) << "fn" << std::setw(20) << "precision" << std::setw(20) << "recall" << std::endl;
      break;

    // to show the evaluation per scene
    case 1:
      std::cout << std::string(150, '-') << std::endl;
      std::cout << std::left << std::setw(15) << params.scene << std::setw(15) << eval.gt_boxes << std::setw(40) <<
        eval.results_boxes_above_conf_threshold << std::setw(15) << s << std::setw(10) << eval.tp << std::setw(10) <<
        eval.fp << std::setw(10) << eval.fn << std::setw(20) << eval.precision << std::setw(20) << eval.recall << std::endl;
      break;

    // to show the average evaluation
    case 2:
      std::cout << std::string(150, '-') << std::endl;
      std::cout << std::left << std::setw(15) << "Total" << std::setw(15) << eval.gt_boxes << std::setw(40) <<
        eval.results_boxes_above_conf_threshold << std::setw(15) << s << std::setw(10) << eval.tp << std::setw(10) <<
        eval.fp << std::setw(10) << eval.fn << std::setw(20) << eval.precision << std::setw(20) << eval.recall << std::endl;
      std::cout << std::string(150, '-') << std::string(2, '\n') << std::flush;
      break;
  }
  return;
}

// ----------------------- saveEvaluationIntoFile -----------------------

void saveEvaluationIntoFile(EvalParameters &params,
                            const Eval &eval,
                            int level)
{
  std::ofstream file;

  // create new file
  if(params.create_new_file) {
    file.open(params.evaluation_filename);
    params.create_new_file = 0;
  }

  // open existing file in 'append' mode
  else {
    file.open(params.evaluation_filename, std::ios::app);
  }

  std::string s = "(" + std::to_string(eval.results_boxes) + ")";
  switch (level) {

    // first line: style stuff
    case 0:
      file << std::string(170, '-') << std::endl;
      file << std::left << std::setw(15) << params.dataset << std::setw(15) << "gt_boxes" << std::setw(40) <<
        "results_boxes_above_conf_threshold" << std::setw(15) << "(total)" << std::setw(10) << "tp" << std::setw(10) <<
        "fp" << std::setw(10) << "fn" << std::setw(20) << "precision" << std::setw(20) << "recall" << std::endl;
      break;

    // to save the evaluation per scene
    case 1:
      file << std::string(170, '-') << std::endl;
      file << std::left << std::setw(15) << params.scene << std::setw(15) << eval.gt_boxes << std::setw(40) <<
        eval.results_boxes_above_conf_threshold << std::setw(15) << s << std::setw(10) << eval.tp << std::setw(10) <<
        eval.fp << std::setw(10) << eval.fn << std::setw(20) << eval.precision << std::setw(20) << eval.recall << std::endl;
      break;

    // to save the average evaluation
    case 2:
      file << std::string(170, '-') << std::endl;
      file << std::left << std::setw(15) << "Total" << std::setw(15) << eval.gt_boxes << std::setw(40) <<
        eval.results_boxes_above_conf_threshold << std::setw(15) << s << std::setw(10) << eval.tp << std::setw(10) <<
        eval.fp << std::setw(10) << eval.fn << std::setw(20) << eval.precision << std::setw(20) << eval.recall << std::endl;
      file << std::string(170, '-') << std::string(2, '\n');
      break;
  }
  return;
}

// ----------------------- computeUnionArea -----------------------

float computeUnionArea(const cv::Rect &gt_rect,
                       const cv::Rect &results_rect)
{
  return ((gt_rect.width+1)*(gt_rect.height+1)) + ((results_rect.width+1)*(results_rect.height+1));
}

// ----------------------- computeIntersectionArea -----------------------

float computeIntersectionArea(const cv::Rect &gt_rect,
                              const cv::Rect &results_rect)
{
  cv::Rect intersection_rect = gt_rect & results_rect;
  return (intersection_rect.width+1)*(intersection_rect.height+1); // Pixels --> +1
}

// ----------------------- computeIou -----------------------

float computeIou(Gt_box &gt_box,
                 Results_box &results_box)
{
  cv::Rect gt_rect = cv::Rect(gt_box.getLeftX(), gt_box.getTopY(), gt_box.getWidth(), gt_box.getHeight());
  cv::Rect results_rect = cv::Rect(results_box.getLeftX(), results_box.getTopY(), results_box.getWidth(), results_box.getHeight());

  float intersection_area = computeIntersectionArea(gt_rect, results_rect);
  float union_area = computeUnionArea(gt_rect, results_rect);

  return (intersection_area/(union_area-intersection_area));

}

// ----------------------- updateEval -----------------------

void updateEval(Eval &eval_out,
                const Eval &eval_in)
{
  eval_out.gt_boxes += eval_in.gt_boxes;
  eval_out.results_boxes += eval_in.results_boxes;
  eval_out.results_boxes_above_conf_threshold += eval_in.results_boxes_above_conf_threshold;
  eval_out.tp += eval_in.tp;
  eval_out.fp += eval_in.fp;
  eval_out.fn += eval_in.fn;
  eval_out.precision = std::max(float(0), float(eval_out.tp)/eval_out.results_boxes_above_conf_threshold);
  eval_out.recall = std::max(float(0), float(eval_out.tp)/eval_out.gt_boxes);
}

// ----------------------- evaluateImage -----------------------

Eval evaluateImage(EvalParameters &params,
                   const std::pair<std::vector<Gt_box>,
                   std::vector<Results_box>> &gt_and_results_boxes)
{
  Eval image_eval;

  std::vector<Gt_box> gt_boxes = gt_and_results_boxes.first;
  std::vector<Results_box> results_boxes = gt_and_results_boxes.second;

  // number of gt and results bounding boxes
  image_eval.gt_boxes += gt_boxes.size();
  image_eval.results_boxes += results_boxes.size();

  // to just consider the YOLO detections with a confidence > conf_threshold (e.g. 0.8)
  auto results_boxes_it = results_boxes.begin();
  while(results_boxes_it != results_boxes.end()) {
    // confidence of the results bounding box
    float conf = (*results_boxes_it).getConfidence();
    // if confidence < conf_threshold, we erase the bounding box from the vector
    if (conf < params.conf_threshold) {
      results_boxes_it = results_boxes.erase(results_boxes_it);
    }
    else {
      ++results_boxes_it;
    }
  }

  // iterate gt boxes
  for(int i = 0; i < gt_boxes.size(); ++i) {
    Gt_box gt_box = gt_boxes[i];
    float iou_max = params.iou_threshold;

    // match_results_box: index of the results box that matches with this ground truth box
    int match_results_box;
    bool match = false;

    // iterate results boxes
    for(int j = 0; j < results_boxes.size(); ++j) {
      Results_box results_box = results_boxes[j];

      // compute intersection over union (iou) between gt and results bounding box
      float iou = computeIou(gt_box, results_box);

      // if iou > last iou_max --> save index of the results box (j) and iou
      if (iou > iou_max) {
        iou_max = iou;
        match_results_box = j;
        match = true;
      }
    } // end results boxes iteration

    // if there's a results box that matched with this gt box
    if (match) {

      // i: index of the gt, iou_max: iou between this gt (i) and the matched results box
      auto gt_box_iou_pair = std::make_pair(i, iou_max);

      // if the matched results box is not matched with another gt
      if (image_eval.true_positives.find(match_results_box) == image_eval.true_positives.end()) {

        // OK --> ++ true positive
        image_eval.true_positives[match_results_box] = gt_box_iou_pair;
      }

      // if the matched results box is also matched with another gt
      else {

        // if the iou between the matched results box and this gt box is greater than that of the other gt box
        if (iou_max > image_eval.true_positives[match_results_box].second) {

          // the gt box that was previously matched with the matched results box --> ++ false  negative
          image_eval.false_negatives.push_back(image_eval.true_positives[match_results_box].first);

          // this gt box --> ++ true positive
          image_eval.true_positives[match_results_box] = gt_box_iou_pair;
        }

        // if it's smaller
        else {

          // this gt box --> ++ false negatives (and the previous gt box remain as a true positive)
          image_eval.false_negatives.push_back(i);
        }
      }
    }

    // if there isn't any results box that matched with this gt box
    else {

      // this gt box --> ++ false negatives
      image_eval.false_negatives.push_back(i);
    }
  } // end gt boxes iteration

  // the results boxes that didn't match with any gt box --> ++ false positives
  for (int p = 0; p < results_boxes.size(); p++) {
    if (image_eval.true_positives.find(p) == image_eval.true_positives.end()) {
      image_eval.false_positives.push_back(p);
    }
  }

  image_eval.results_boxes_above_conf_threshold = results_boxes.size();
  image_eval.tp = image_eval.true_positives.size();
  image_eval.fp = image_eval.false_positives.size();
  image_eval.fn = image_eval.false_negatives.size();
  image_eval.precision = std::max(float(0), float(image_eval.tp)/(image_eval.tp+image_eval.fp));
  image_eval.recall = std::max(float(0), float(image_eval.tp)/(image_eval.tp+image_eval.fn));

  // 'display' mode
  if(params.display_scene && params.display_dataset && params.display_all) {
    displayImage(params, image_eval, std::make_pair(gt_boxes, results_boxes));
  }

  return image_eval;
}

//--------------------------------- evaluation.hpp ---------------------------------

// ----------------------- sceneMapExists -----------------------

bool sceneMapExists(EvalParameters &params,
                    const std::string &opt)
{
  // path to the gt/results maps of a dataset
  std::string dataset_maps_path;

  if (opt == "gt") {
    dataset_maps_path = params.gt_maps_path + params.dataset;
  }
  else if (opt == "results") {
    dataset_maps_path = params.results_maps_path + params.dataset;
  }

  boost::filesystem::directory_iterator it{dataset_maps_path};
  while(it != boost::filesystem::directory_iterator{}) {
    std::string map_name = (*it++).path().filename().string();

    // if map exists
    if(map_name.find(params.scene + "_") != map_name.npos) {

      if (opt == "gt") {
        params.gt_scene_map_filename = map_name;
      }

      else if(opt == "results") {
        params.results_scene_map_filename = map_name;
      }

      return true;
    }
  }

  // if not
  return false;
}

// ----------------------- evaluateScene -----------------------

Eval evaluateScene(EvalParameters &params)
{
  Eval scene_eval;

  // check if results (YOLO detections) map of a specific dataset and scene exists
  if(!sceneMapExists(params, "results")) {
    std::cerr << "ERROR: No results" << " " << params.dataset << " " << params.scene << " map" << '\n';
    return scene_eval;
  }

  // path to specific ground truth (manual label) map
  std::string gt_scene_map_path = params.gt_maps_path + params.dataset + "/" + params.gt_scene_map_filename;

  // path to specific results (YOLO detections) map
  std::string results_scene_map_path = params.results_maps_path + params.dataset + "/" + params.results_scene_map_filename;

  // maps to store ground truth and results bounding boxes:
  // std::string      --> image filename
  // std::vector<Box> --> bounding boxes (gt or results) for that image
  std::unordered_map<std::string, std::vector<Gt_box>> gt_map;
  std::unordered_map<std::string, std::vector<Results_box>> results_map;

  // deserialize maps
  tie(gt_map, results_map) = loadMaps(gt_scene_map_path, results_scene_map_path);

  // iterate through gt map entries (one entry = one image from a specific dataset and scene)
  for (auto const &it: gt_map) {

    // image filename
    params.image_filename = it.first;

    // vector containing the gt boxes of that image
    auto gt_boxes = it.second;

    // vector containing the results boxes of that image (obtained from the results map)
    auto results_boxes = results_map[params.image_filename];

    // evaluate specific image
    Eval image_eval = evaluateImage(params, std::make_pair(gt_boxes, results_boxes));

    // update scene evaluation (taking into account the new evaluated image)
    updateEval(scene_eval, image_eval);
  }

  // write the evaluation results into the .txt
  if(params.option == "save") {
    saveEvaluationIntoFile(params, scene_eval, 1);
  }

  // show the evaluation results (console)
  outputEvaluation(params, scene_eval, 1);
  return scene_eval;
}

// ----------------------- evaluateDataset -----------------------

Eval evaluateDataset(EvalParameters &params)
{
  try{
    Eval dataset_eval;

    // to store the path to each scene folder (of one specific dataset)
    std::vector<boost::filesystem::path> scenes_paths;

    std::copy(boost::filesystem::directory_iterator(params.gt_maps_path + params.dataset),
      boost::filesystem::directory_iterator(), std::back_inserter(scenes_paths));

    // sort paths alphabetically (1, 10, 2, 3, 4..)
    std::sort(scenes_paths.begin(), scenes_paths.end());

    // iterate through the scenes folders
    for(auto const &scene_path : scenes_paths) {

      params.gt_scene_map_filename = scene_path.filename().string();

      // scene# (1, 2, ...)
      params.scene = params.gt_scene_map_filename.substr(0, params.gt_scene_map_filename.find("_"));

      // need this 'if' to display the next scene when you press 's' while displaying the results
      if(params.option == "display") {
        params.display_scene = 1;
      }

      // evaluate specific scene
      Eval scene_eval = evaluateScene(params);

      // update dataset evaluation (taking into account the new evaluated scene)
      updateEval(dataset_eval, scene_eval);
    }

    // write the evaluation results into the .txt
    if(params.option == "save") {
      saveEvaluationIntoFile(params, dataset_eval, 2);
    }

    // show the evaluation results (console)
    outputEvaluation(params, dataset_eval, 2);
    return dataset_eval;
  }

  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- evaluateAll -----------------------

Eval evaluateAll(EvalParameters &params)
{
  try {
    Eval total_eval;

    std::cout << '\n' << "iou: " << params.iou_threshold << "    " << "confidence_threshold: " << params.conf_threshold << std::string(2, '\n') << std::flush;

    // to store the path to each dataset folder (highway, indoor and outdoor)
    std::vector<boost::filesystem::path> datasets_paths;

    std::copy(boost::filesystem::directory_iterator(params.gt_maps_path),
      boost::filesystem::directory_iterator(), std::back_inserter(datasets_paths));

    // sort paths alphabetically
    std::sort(datasets_paths.begin(), datasets_paths.end());

    // iterate through the datasets folders
    for(auto const &dataset_path : datasets_paths) {

      //highway, indoor or outdoor
      params.dataset = dataset_path.filename().string();

      // we are not writing any evaluation results into the .txt, just some style stuff
      if(params.option == "save") {
        saveEvaluationIntoFile(params, total_eval, 0);
      }

      // need this 'if' to display the next dataset when you press 'd' while displaying the results
      else if(params.option == "display") {
        params.display_dataset = 1;
      }

      // we are not showing any evaluation results (console), just some style stuff
      outputEvaluation(params, total_eval, 0);

      // evaluate specific dataset
      Eval dataset_eval = evaluateDataset(params);

      // update general evaluation (taking into account the new evaluated dataset)
      updateEval(total_eval, dataset_eval);
    }

    // now we are writing the evaluation results into the .txt
    if(params.option == "save") {
      saveEvaluationIntoFile(params, total_eval, 2);
    }

    // now we are showing the evaluation results (console)
    outputEvaluation(params,total_eval, 2);
    return total_eval;
  }

  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}
