#pragma once

#include <opencv4/opencv2/dnn.hpp>

struct DetectorParameters
{
  const std::string option; // display/save
  const std::string datasets_path, results_path;

  std::string dataset, scene, image_filename, image_path;

  int display_scene, display_dataset, display_all; // auxiliar parameters to control the display of the detections

  std::string yolo_base_path, yolo_classes_file, yolo_model_configuration, yolo_model_weights;
  std::vector<std::string> classes;
  float conf_threshold, nms_threshold;

  DetectorParameters(const std::string &opt,
                     const std::string &d_path,
                     const std::string &r_path);
  ~DetectorParameters() = default;
};

void loadYoloData(DetectorParameters &params,
                  cv::dnn::Net &yolo_net);

void processScene(DetectorParameters &params,
                  cv::dnn::Net &yolo_net);

void processDataset(DetectorParameters &params,
                    cv::dnn::Net &yolo_net);

void processAll(DetectorParameters &params,
                cv::dnn::Net &yolo_net);
