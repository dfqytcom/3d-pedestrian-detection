/*
Main function to evaluate YOLO detections.
*/

#include "image_pedestrian_detector/evaluation.hpp"
#include "utils/maps.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <boost/filesystem.hpp>

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  if ((argc < 2) || (argc > 4) || ((std::string(argv[1]) != "display") && (std::string(argv[1])) != "save")) {
    std::cout << "Usage: [" << argv[0] << "] [display/save] <dataset> <scene>" << std::endl;
    return 0;
  }

  // path to ground truth (manual label) maps
  std::string gt_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/gt/";

  // path to results (YOLO detections) maps
  std::string results_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/results/";

  // argv[1] = option (display/save)
  EvalParameters params(argv[1], gt_maps_path, results_maps_path);

  params.iou_threshold = 0.5; // default
  params.conf_threshold = 0.8; // default

  // auxiliar parameters to control the display of the evaluation
  if(params.option == "display") {
    params.display_all = 1;
    params.display_dataset = 1;
    params.display_scene = 1;
  }

  // if 'save' mode --> check if folder to save the evaluation results exists
  else if (params.option == "save") {
    params.create_new_file = 1;

    std::string evaluation_path = "../results/evaluation/";
    if (!boost::filesystem::is_directory(evaluation_path)) {
      std::cerr << std::endl << "WARNING: No such directory " << evaluation_path << " to save evaluation results" << std::string(2, '\n') << std::flush;
      std::cout << "Do you want to create it? (y/n): ";
      std::string create;
      std::cin >> create;
      if (create == "y" || create == "Y") {
        boost::filesystem::create_directories(evaluation_path);
        std::cout << "Directory " << evaluation_path << " created!" << '\n';
      }
      else {
        std::cout << std::string(2, '\n') << "OPERATION ABORTED" << std::string(2, '\n');
        return 0;
      }
    }

    // path + filename to save the evaluation results
    std::stringstream evaluation_filename;
    evaluation_filename << std::fixed << std::setprecision(2) << evaluation_path << "evaluation_iou_" << params.iou_threshold << "_conf_" << params.conf_threshold << ".txt";
    params.evaluation_filename = evaluation_filename.str();
  }

  switch(argc) {

    // evaluate all datasets
    case 2:
      evaluateAll(params);
      break;

    // evaluate one dataset (argv[2])
    case 3:
      params.dataset = argv[2];
      evaluateDataset(params);
      break;

    // evaluate one scene (argv[3]) from one dataset (argv[2])
    case 4:
      params.dataset = argv[2];
      params.scene = argv[3];

      // check if ground truth map of a specific dataset and scene exists
      if(!sceneMapExists(params, "gt")) {
        std::cerr << "ERROR: No gt " << " " << params.dataset << " " << params.scene << " map" << '\n';
        return 0;
      }
      evaluateScene(params);
      break;
  }

  if(params.option == "save") {
    std::cout << "Evaluation results saved in: " << params.evaluation_filename << std::string(2, '\n') << std::flush;
  }

  return 1;
}
