/*
Compute and save the precision and recall values with different confidence thresholds
and a fixed iou threshold (0.5).
*/

#include "image_pedestrian_detector/evaluation.hpp"
#include "utils/maps.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <boost/filesystem.hpp>

// ----------------------- fillVector -----------------------

void fillVector(std::vector<float> &v,
                float init,
                float end,
                float inc)
{
  for(float num = init; num < end; num+=inc) {
    v.push_back(num);
  }
}

// ----------------------- sortBySecond -----------------------

bool sortBySecond(const std::tuple<float,float,float> &a,
                  const std::tuple<float,float,float> &b)
{
    return (std::get<1>(a) < std::get<1>(b));
}

// ----------------------- savePrecisionRecall -----------------------

void savePrecisionRecall(const EvalParameters &params,
                        std::vector<std::tuple<float, float, float>> &precision_recall)
{
  sort(precision_recall.begin(), precision_recall.end(), sortBySecond);

  // output filename
  std::ofstream file(params.precision_recall_filename);

  // save precision value
  for (int i = 0; i < precision_recall.size(); ++i) {
    file << std::get<0>(precision_recall[i]);
    if (i != precision_recall.size()-1) {
      file << ",";
    }
    else {
      file << '\n';
    }
  }

  // save recall value
  for (int j = 0; j < precision_recall.size(); ++j) {
    file << std::get<1>(precision_recall[j]);
    if (j != precision_recall.size()-1) {
      file << ",";
    }
    else {
      file << '\n';
    }
  }

  // save confidence threshold value
  for (int k = 0; k < precision_recall.size(); ++k) {
    file << std::get<2>(precision_recall[k]);
    if (k != precision_recall.size()-1) {
      file << ",";
    }
  }
  return;
}

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  if (argc != 1) {
    std::cout << "Usage: [" << argv[0] << "]" << std::endl;
    return 0;
  }

  std::vector<float> conf_thresholds;

  fillVector(conf_thresholds, 0.3, 0.99, (0.99-0.3)/20); // 20 different confidence thresholds

  std::string gt_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/gt/";
  std::string results_maps_path = std::string(std::getenv("MEDIA")) + "/image/maps/results/";

  EvalParameters params("x", gt_maps_path, results_maps_path);

  params.create_new_file = 1;

  std::string precision_recall_path = "../results/precision_recall/";

  if(!boost::filesystem::is_directory(precision_recall_path)) {
    std::cerr << std::endl << "WARNING: No such directory " <<  precision_recall_path <<  " to save precision_recall results" << std::string(2, '\n') << std::flush;
    std::cout << "Do you want to create it? (y/n): ";
    std::string create;
    std::cin >> create;
    if (create == "y" || create == "Y") {
      boost::filesystem::create_directories(precision_recall_path);
      std::cout << "Directory " << precision_recall_path << " created!" << '\n';
    }
    else {
      std::cout << std::string(2, '\n') << "OPERATION ABORTED" << std::string(2, '\n');
      return 0;
    }
  }

  params.precision_recall_filename = precision_recall_path + "precision_recall_confidence_thresholds.txt";

  std::vector<std::tuple<float, float, float>> precision_recall_iou;

  // fixed iou threhold (typical value)
  params.iou_threshold = 0.5;

  // varying confidence threshold
  for (auto const &conf_threshold : conf_thresholds) {
    params.conf_threshold = conf_threshold;

    // evaluate the results per each confidence threshold
    Eval total_eval = evaluateAll(params);

    // save the precision and recall values obtained with a specific confidence threshold
    precision_recall_iou.push_back(std::make_tuple(total_eval.precision, total_eval.recall, params.conf_threshold));
  }

  savePrecisionRecall(params, precision_recall_iou);

  std::cout << "Precision_recall results saved in: " << params.precision_recall_filename << std::string(2, '\n') << std::flush;

  return 1;
}
