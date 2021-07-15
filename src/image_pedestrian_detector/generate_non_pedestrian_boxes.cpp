/*
Function to generate non-pedestrian bounding boxes based on YOLO pedestrian detections.
*/

#include "utils/boxes.hpp"
#include "utils/maps.hpp"
#include "utils/histogram.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <map>
#include <cmath>
#include <random>
#include <iterator>

#include <boost/filesystem.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/base_object.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

const int image_width = 1224;
const int image_height = 1024;

// Random generator
template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

// ----------------------- getMeanStd -----------------------

std::tuple<double, double> getMeanStd(const std::vector<double> &v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });

  double squared_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double std = std::sqrt(squared_sum / v.size());

  return std::make_pair(mean, std);
}

// ----------------------- main -----------------------

int main(int argc, char const *argv[]) {

  // obtained from the function "get_boxes_stats.cpp"
  // must refactor this to get the values automatically from the "get_boxes_stats.cpp" functions
  // ---
    int width_min = 9;
    int width_max = 690;
    float width_mean = 95.6;
    float width_std = 57.1;

    int height_min = 17;
    int height_max = 1118;
    float height_mean = 267.1;
    float height_std = 145.9;

    float ratio_mean = 2.94;
    float ratio_std = 1.086;

    // number of pedestrian bounding boxes
    int total_bbp = 13232;

    // percentage of pixels occupied by pedestrian bounding boxes (11.47%)
    float perc_occ_pxs = 0.1147;
  // ---

  // number of bbnp to generate
  double total_bbnp = total_bbp / perc_occ_pxs * (1.0 - perc_occ_pxs);
  std::cout << "- Total non-pedestrian bounding boxes to generate: " << total_bbnp << '\n';

  // random generator
  std::random_device rd{};
  std::mt19937 gen{rd()};

  // gaussian (normal) width and height distributions
  std::normal_distribution<> width_distribution{width_mean,width_std};
  std::normal_distribution<> height_distribution{height_mean,height_std};

  // path to the results (YOLO detections) bounding boxes
  boost::filesystem::directory_iterator results_maps_it{std::string(std::getenv("MEDIA")) + "/image/maps/results/outdoor"};

  // to store the images and pedestrian bounding boxes of all outdoor scenes
  std::unordered_map<std::string, std::vector<Results_box>> full_map;

  // iterate through bbp maps
  for (auto const &results_map_path : results_maps_it) {
    std::string map_path = results_map_path.path().string();
    std::string map_filename = results_map_path.path().filename().string();
    std::string scene = map_filename.substr(0, map_filename.find("_"));

    // load pedestrian bounding boxes map
    std::unordered_map<std::string, std::vector<Results_box>> results_map = loadResultsMap(results_map_path.path().string());

    for (auto &results_map_it : results_map) {
      full_map[scene + "/" + results_map_it.first] = results_map_it.second;
    }
  }

  double generated_bbnp = 0;

  std::unordered_map<std::string, std::vector<cv::Rect>> bbnp_full_map;

  // until all bbnp are generated
  while(generated_bbnp <= total_bbnp) {

    // generate random width with the same statistics as the pedestrian bounding boxes (normal distribution)
    int width = std::round(width_distribution(gen));
    if(width >= width_min && width <= width_max) {

      // generate random height with the same statistics as the pedestrian bounding boxes (normal distribution)
      int height = std::round(height_distribution(gen));
      if(height >= height_min && height <= height_max) {

        // check if the ratio has the same statistics as the pedestrian bounding boxes
        float ratio = float(height)/width;
        if(ratio >= (ratio_mean - ratio_std) && ratio <= (ratio_mean + ratio_std)) {

          bool valid = false;

          // to generate random pixels
          std::uniform_int_distribution<> x_uniform_distr(0, image_width-width-1);
          std::uniform_int_distribution<> y_uniform_distr(0, image_height-height-1);

          std::string image_name;
          cv::Rect bbnp;

          // until a valid bbnp is found...
          while(!valid) {

            // random image
            std::pair<std::string, std::vector<Results_box>> image_boxes = *select_randomly(full_map.begin(), full_map.end());
            image_name = image_boxes.first;

            // get the pedestrian bounding boxes of that image
            std::vector<Results_box> boxes = full_map[image_name];
            int pixel_x, pixel_y;

            // try 10 times per image
            for(int iter = 0; iter <= 10; ++iter) {
              bool overlapped = false;

              // generate random pixel coordinates (x,y)
              pixel_x = x_uniform_distr(gen);
              pixel_y = y_uniform_distr(gen);

              // create the candidate non-pedestrian bounding box with the width and height previously generated
              bbnp = cv::Rect(pixel_x, pixel_y, width, height);

              // check if any pedestrian bounding box is overlapped with the candidate non-pedestrian
              for(auto &results_box : image_boxes.second) {

                // pedestrian bounding box
                cv::Rect bbp = cv::Rect(results_box.getLeftX(), results_box.getTopY(), results_box.getWidth(), results_box.getHeight());

                // intersection between the bbp and the bbnp
                cv::Rect intersection_rect = bbp & bbnp;

                // if the intersection area > 0 (could be changed to another value) --> there is overlap --> discard bbnp
                if(intersection_rect.area() > 0) {
                  overlapped = true;
                  break;
                }
              }

              // if the non-pedestrian bounding box is not overlapped with any pedestrian bounding box
              if(!overlapped) {

                // if there aren't others bbnp in this image
                if (bbnp_full_map.find(image_name) == bbnp_full_map.end()) {

                  // accept the bbnp as valid --> ++generated_bbnp
                  ++generated_bbnp;
                  bbnp_full_map[image_name].push_back(bbnp);
                  valid = true;
                  break;
                }

                // if there are others bbnp in this image
                else {
                  bool overlapped_bbnp = false;

                  // iterate through the already generated bbnp in the same image
                  for(auto const &old_bbnp : bbnp_full_map[image_name]) {

                    // intersection between the candidate bbnp and the already generated bbnp
                    cv::Rect intersection_bbnp = bbnp & old_bbnp;
                    float perc_overlap = float(intersection_bbnp.area())/bbnp.area();
                    float perc_other_overlap = float(intersection_bbnp.area())/old_bbnp.area();

                    // if the overlapping percentage between the candidate bbnp and any already generated bbnp > 0.7 (could be changed to another value) --> discard bbnp
                    if(perc_overlap > 0.7){
                      overlapped_bbnp = true;
                      break;
                    }
                  }

                  // if the bbnp candidate is NOT overlapped with any other already generated bbnp more than 0.7 --> accept the bbnp as valid --> ++generated_bbnp
                  if(!overlapped_bbnp) {
                    ++generated_bbnp;
                    bbnp_full_map[image_name].push_back(bbnp);
                    valid = true;
                    break;
                  }
                }
              }
            }
          }
          // end of while(!valid) {...} loop
          std::cout << " ... " << generated_bbnp << '\n';
        }
      }
    }
  }

  std::unordered_map<std::string, std::unordered_map<std::string, std::vector<Results_box>>> to_save_bbnp;

  // iterate through each image
  for(auto const &bbnp_full_map_it : bbnp_full_map) {

    // e.g. "scene1/20200110_135503_101.txt"
    std::string scene_image_filename = bbnp_full_map_it.first;

    // e.g. "scene1"
    std::string scene = scene_image_filename.substr(0, scene_image_filename.find("/"));

    // iterate through each non-pedestrian bounding box generated in the image
    for (auto const & bbnp_rectangle : bbnp_full_map_it.second) {

      // Save them as Results_box (need to refactor these)
      Results_box rbox(1, //classID
                       float((bbnp_rectangle.x + bbnp_rectangle.width/2))/image_width,
                       float((bbnp_rectangle.y+bbnp_rectangle.height/2))/image_height,
                       float(bbnp_rectangle.width)/image_width,
                       float(bbnp_rectangle.height)/image_height,
                       1.0); //confidence

      // pair the non-pedestrian bounding bounding box with its corresponding scene
      to_save_bbnp[scene][scene_image_filename].push_back(rbox);
    }
  }

  // save the generated non-pedestrian bounding boxes per scene --> serialize
  for(auto & to_save_bbnp_scene : to_save_bbnp) {

    // output file path
    std::string save_bbnp_map_path = std::string(std::getenv("MEDIA")) + "/image/maps/bbnp/outdoor/" + to_save_bbnp_scene.first + "_bbnp_map.txt";

    std::ofstream save_bbnp_map_file{save_bbnp_map_path};
    boost::archive::text_oarchive oa{save_bbnp_map_file};
    oa << to_save_bbnp_scene.second;
  }

  std::cout << "--> Generated non-pedestrian bounding boxes: " << generated_bbnp << '\n';
  return 0;
}
