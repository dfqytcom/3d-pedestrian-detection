/*
Main function to detect pedestrians in RGB images using YOLOv3.
*/

#include "image_pedestrian_detector/image_pedestrian_detection.hpp"

#include <iostream>

// ----------------------- main -----------------------

int main(int argc,
         char *argv[])
{
  if (argc < 2 || argc > 4 || ((std::string(argv[1]) != "display") && (std::string(argv[1]) != "save"))) {
    std::cout << "[./image_pedestrian_detector] [display/save] <dataset> <scene>" << std::endl;
    return 0;
  }

  // path to the "datasets" folder, containing "highway", "indoor" and "outdoor" folders
  std::string datasets_path = std::string(std::getenv("MEDIA")) + "/image/datasets/";

  // path to the folder where the results are saved
  std::string results_path = std::string(std::getenv("MEDIA")) + "/image/results/";

  // argv[1] = option (display/save)
  DetectorParameters params(argv[1], datasets_path, results_path);

  // auxiliar parameters to control the display of the detections
  if(params.option == "display") {
    params.display_all = 1;
    params.display_dataset = 1;
    params.display_scene = 1;
  }

  cv::dnn::Net yolo_net;
  loadYoloData(params, yolo_net);

  switch (argc) {

    // process all datasets
    case 2:
      processAll(params, yolo_net);
      break;

    // process one dataset (argv[2])
    case 3:
      params.dataset = argv[2];
      processDataset(params, yolo_net);
      break;

    // process one scene (argv[3]) from one dataset (argv[2])
    case 4:
      params.dataset = argv[2];
      params.scene = argv[3];
      processScene(params, yolo_net);
      break;
  }

  if(params.option == "save") {
    std::cout << '\n' << "Results saved in: " << params.results_path << std::string(2, '\n') << std::flush;
  }

  return 1;
}
