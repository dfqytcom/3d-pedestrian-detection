/*
Utility functions to get pedestriand and non bounding boxes stats per scene (must be refactored).
*/

#include "utils/boxes.hpp"
#include "utils/maps.hpp"
#include "utils/histogram.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <numeric>

#include <boost/filesystem.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// ----------------------- getMeanStd -----------------------

std::tuple<double, double> getMeanStd(const std::vector<double> &v)
{
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
  try{
    // pedestrian bounding boxes stats

    std::vector<double> bbp_width_vector, bbp_height_vector, bbp_ratio_vector, bbp_per_image_vector;
    std::vector<std::string> scenes;
    std::map<std::string, std::vector<double>> scenes_bbp_width, scenes_bbp_height, scenes_bbp_ratio;
    std::map<std::string, std::vector<double>> scenes_bbp_per_image, scenes_bbp_overlapped_perc, scenes_bbp_occupied_pixels;
    std::map<std::string, double> scenes_number_images, scenes_number_bbp;

    double total_images = 0;

    // to store the path to each results (pedestrian YOLO detections) map
    std::vector<boost::filesystem::path> results_maps_paths;

    std::copy(boost::filesystem::directory_iterator(std::string(std::getenv("MEDIA")) + "/image/maps/results/outdoor"),
      boost::filesystem::directory_iterator(), std::back_inserter(results_maps_paths));

    // sort paths alphabetically
    std::sort(results_maps_paths.begin(), results_maps_paths.end());

    for(auto const &results_map_path : results_maps_paths) {

      // scene# (1, 2,...)
      std::string scene = results_map_path.filename().string().substr(0, results_map_path.filename().string().find("_"));

      // load scene results map
      std::unordered_map<std::string, std::vector<Results_box>> results_map = loadResultsMap(results_map_path.string());

      // store the scene name
      scenes.push_back(scene);

      // to store the number of images per scene
      scenes_number_images[scene] = results_map.size();

      total_images += results_map.size();

      // to store the number of pedestrian bounding boxes per scene
      scenes_number_bbp[scene] = 0;

      // iterate through the images of the scene (map entries)
      for (auto const & results_map_it : results_map) {

        // results boxes (YOLO detections) of an image
        auto results_boxes = results_map_it.second;

        // to store the number of pedestrian bounding boxes per image
        bbp_per_image_vector.push_back(results_boxes.size());

        // to store the number of pedestrian bounding boxes per image for each scene
        scenes_bbp_per_image[scene].push_back(results_boxes.size());

        // to store the total number of pedestrian bounding boxes per scene
        scenes_number_bbp[scene] += results_boxes.size();

        // iterate through the bounding boxes of the image
        for (int i = 0; i < results_boxes.size(); ++i) {
          auto box = results_boxes[i];

          // to store the width, height and ratio of each bounding box
          bbp_width_vector.push_back(box.getWidth());
          bbp_height_vector.push_back(box.getHeight());
          bbp_ratio_vector.push_back(box.getHeight()/box.getWidth());

          // to store the width, height and ratio of each bounding box per scene
          scenes_bbp_width[scene].push_back(box.getWidth());
          scenes_bbp_height[scene].push_back(box.getHeight());
          scenes_bbp_ratio[scene].push_back(box.getHeight()/box.getWidth());

          // compute the overlapping between pedestrian bounding boxes
          // ---
            cv::Rect box_rect = cv::Rect(0, 0, box.getWidth(), box.getHeight());

            cv::Mat aux_image_A = cv::Mat::zeros(box_rect.height, box_rect.width, CV_32F);
            cv::rectangle(aux_image_A, box_rect, 1, cv::FILLED);

            int x_offset = box.getLeftX();
            int y_offset = box.getTopY();

            cv::Mat aux_image_B = cv::Mat::zeros(box_rect.height, box_rect.width, CV_32F);

            std::vector<Results_box> aux_boxes{results_boxes};
            aux_boxes.erase(aux_boxes.begin() + i);

            for (auto &aux_box : aux_boxes) {
              cv::Rect aux_rect = cv::Rect(aux_box.getLeftX() - x_offset, aux_box.getTopY() - y_offset, aux_box.getWidth(), aux_box.getHeight());
              cv::rectangle(aux_image_B, aux_rect, 1, cv::FILLED);
            }

            cv::Mat aux_image_C;
            cv::bitwise_and(aux_image_A, aux_image_B, aux_image_C);

            double overlapped_pixels = cv::countNonZero(aux_image_C);

            // resulting percentage of overlapping of the pedestrian bounding boxes
            double overlapped_perc = overlapped_pixels/box_rect.area();
          // ---

          // to store the overlapping percentage of pedestrian bounding boxes per scene
          scenes_bbp_overlapped_perc[scene].push_back(overlapped_perc*100);

          // to store the number pixels occupied by the pedestrian bounding boxes per scene
          scenes_bbp_occupied_pixels[scene].push_back(box_rect.area());
        }
      }
    }

    // non-pedestrian bounding boxes stats

    std::vector<double> bbnp_width_vector, bbnp_height_vector, bbnp_ratio_vector, bbnp_per_image_vector;
    std::map<std::string, std::vector<double>> scenes_bbnp_width, scenes_bbnp_height, scenes_bbnp_ratio;
    std::map<std::string, std::vector<double>> scenes_bbnp_per_image, scenes_bbnp_overlapped_perc, scenes_bbnp_occupied_pixels;
    std::map<std::string, double> scenes_number_bbnp;

    // to iterate to the non-pedestrian maps
    boost::filesystem::directory_iterator bbnp_maps_it{std::string(std::getenv("MEDIA")) + "/image/maps/bbnp/outdoor"};

    for(auto const &bbnp_map_path : bbnp_maps_it) {

      // scene# (1, 2,...)
      std::string scene = bbnp_map_path.path().filename().string().substr(0, bbnp_map_path.path().filename().string().find("_"));

      // load scene bbnp map
      std::unordered_map<std::string, std::vector<Results_box>> bbnp_map = loadResultsMap(bbnp_map_path.path().string());

      // to store the number of non-pedestrian bounding boxes per scene
      scenes_number_bbnp[scene] = 0;

      // iterate through the images of the scene (map entries)
      for (auto const & bbnp_map_it : bbnp_map) {

        // non-pedestrian boundingx boxes of an image
        auto bbnp = bbnp_map_it.second;

        // to store the number of non-pedestrian bounding boxes per image
        bbnp_per_image_vector.push_back(bbnp.size());

        // to store the number of non-pedestrian bounding boxes per image for each scene
        scenes_bbnp_per_image[scene].push_back(bbnp.size());

        // to store the total number of non-pedestrian bounding boxes per scene
        scenes_number_bbnp[scene] += bbnp.size();

        // iterate through the bounding boxes of the image
        for (int i = 0; i < bbnp.size(); ++i) {
          auto box = bbnp[i];

          // to store the width, height and ratio of each bounding box
          bbnp_width_vector.push_back(box.getWidth());
          bbnp_height_vector.push_back(box.getHeight());
          bbnp_ratio_vector.push_back(box.getHeight()/box.getWidth());

          // to store the width, height and ratio of each bounding box per scene
          scenes_bbnp_width[scene].push_back(box.getWidth());
          scenes_bbnp_height[scene].push_back(box.getHeight());
          scenes_bbnp_ratio[scene].push_back(box.getHeight()/box.getWidth());

          // compute the overlapping between non-pedestrian bounding boxes
          // ---
            cv::Rect box_rect = cv::Rect(0, 0, box.getWidth(), box.getHeight());

            cv::Mat aux_image_A = cv::Mat::zeros(box_rect.height, box_rect.width, CV_32F);
            cv::rectangle(aux_image_A, box_rect, 1, cv::FILLED);

            int x_offset = box.getLeftX();
            int y_offset = box.getTopY();

            cv::Mat aux_image_B = cv::Mat::zeros(box_rect.height, box_rect.width, CV_32F);

            std::vector<Results_box> aux_boxes{bbnp};
            aux_boxes.erase(aux_boxes.begin() + i);

            for (auto &aux_box : aux_boxes) {
              cv::Rect aux_rect = cv::Rect(aux_box.getLeftX() - x_offset, aux_box.getTopY() - y_offset, aux_box.getWidth(), aux_box.getHeight());
              cv::rectangle(aux_image_B, aux_rect, 1, cv::FILLED);
            }

            cv::Mat aux_image_C;
            cv::bitwise_and(aux_image_A, aux_image_B, aux_image_C);

            double overlapped_pixels = cv::countNonZero(aux_image_C);

            // resulting percentage of overlapping of the non-pedestrian bounding boxes
            double overlapped_perc = overlapped_pixels/box_rect.area();
          // ---

          // to store the overlapping percentage of pedestrian bounding boxes per scene
          scenes_bbnp_overlapped_perc[scene].push_back(overlapped_perc*100);

          // to store the number pixels occupied by the non-pedestrian bounding boxes per scene
          scenes_bbnp_occupied_pixels[scene].push_back(box_rect.area());
        }
      }
    }

    // ----------------------------------------------------------------------------------------------

    // pedestrian bounding boxes stats

      // width
      double bbp_width_mean, bbp_width_std;
      double bbp_width_max =  *std::max_element(bbp_width_vector.begin(), bbp_width_vector.end());
      double bbp_width_min = *std::min_element(bbp_width_vector.begin(), bbp_width_vector.end());
      std::tie(bbp_width_mean, bbp_width_std) = getMeanStd(bbp_width_vector);

      // height
      double bbp_height_mean, bbp_height_std;
      double bbp_height_max = *std::max_element(bbp_height_vector.begin(), bbp_height_vector.end());
      double bbp_height_min = *std::min_element(bbp_height_vector.begin(), bbp_height_vector.end());
      std::tie(bbp_height_mean, bbp_height_std) = getMeanStd(bbp_height_vector);

      // ratio
      double bbp_ratio_mean, bbp_ratio_std;
      double bbp_ratio_max =  *std::max_element(bbp_ratio_vector.begin(), bbp_ratio_vector.end());
      double bbp_ratio_min = *std::min_element(bbp_ratio_vector.begin(), bbp_ratio_vector.end());
      std::tie(bbp_ratio_mean, bbp_ratio_std) = getMeanStd(bbp_ratio_vector);

      // bounding boxes per image
      double bbp_per_image_mean, bbp_per_image_std;
      double bbp_per_image_max = *std::max_element(bbp_per_image_vector.begin(), bbp_per_image_vector.end());
      double bbp_per_image_min = *std::min_element(bbp_per_image_vector.begin(), bbp_per_image_vector.end());
      std::tie(bbp_per_image_mean, bbp_per_image_std) = getMeanStd(bbp_per_image_vector);

    // non-pedestrian bounding boxes stats

      // width
      double bbnp_width_mean, bbnp_width_std;
      double bbnp_width_max =  *std::max_element(bbnp_width_vector.begin(), bbnp_width_vector.end());
      double bbnp_width_min = *std::min_element(bbnp_width_vector.begin(), bbnp_width_vector.end());
      std::tie(bbnp_width_mean, bbnp_width_std) = getMeanStd(bbnp_width_vector);

      // height
      double bbnp_height_mean, bbnp_height_std;
      double bbnp_height_max = *std::max_element(bbnp_height_vector.begin(), bbnp_height_vector.end());
      double bbnp_height_min = *std::min_element(bbnp_height_vector.begin(), bbnp_height_vector.end());
      std::tie(bbnp_height_mean, bbnp_height_std) = getMeanStd(bbnp_height_vector);

      // ratio
      double bbnp_ratio_mean, bbnp_ratio_std;
      double bbnp_ratio_max =  *std::max_element(bbnp_ratio_vector.begin(), bbnp_ratio_vector.end());
      double bbnp_ratio_min = *std::min_element(bbnp_ratio_vector.begin(), bbnp_ratio_vector.end());
      std::tie(bbnp_ratio_mean, bbnp_ratio_std) = getMeanStd(bbnp_ratio_vector);

      // bounding boxes per image
      double bbnp_per_image_mean, bbnp_per_image_std;
      double bbnp_per_image_max = *std::max_element(bbnp_per_image_vector.begin(), bbnp_per_image_vector.end());
      double bbnp_per_image_min = *std::min_element(bbnp_per_image_vector.begin(), bbnp_per_image_vector.end());
      std::tie(bbnp_per_image_mean, bbnp_per_image_std) = getMeanStd(bbnp_per_image_vector);

    // other stats
    double overall_overlapped_perc = 0;
    double overall_occupied_pixels_perc = 0;
    double overall_bbnp_overlapped_perc = 0;
    double overall_bbnp_occupied_pixels_perc = 0;
    double total_bbp = 0;
    double total_bbnp = 0;

    // per scene stats
    std::cout << '\n' << "SCENES" << std::string(2,'\n');
    for(auto const & scene : scenes) {

      // pedestrian bounding boxes

        // width
        auto scene_bbp_width_vector = scenes_bbp_width[scene];
        double scene_bbp_width_mean, scene_bbp_width_std;
        double scene_bbp_width_max = *std::max_element(scene_bbp_width_vector.begin(), scene_bbp_width_vector.end());
        double scene_bbp_width_min = *std::min_element(scene_bbp_width_vector.begin(), scene_bbp_width_vector.end());
        std::tie(scene_bbp_width_mean, scene_bbp_width_std) = getMeanStd(scene_bbp_width_vector);

        // height
        auto scene_bbp_height_vector = scenes_bbp_height[scene];
        double scene_bbp_height_mean, scene_bbp_height_std;
        double scene_bbp_height_max = *std::max_element(scene_bbp_height_vector.begin(), scene_bbp_height_vector.end());
        double scene_bbp_height_min = *std::min_element(scene_bbp_height_vector.begin(), scene_bbp_height_vector.end());
        std::tie(scene_bbp_height_mean, scene_bbp_height_std) = getMeanStd(scene_bbp_height_vector);

        // ratio
        auto scene_bbp_ratio_vector = scenes_bbp_ratio[scene];
        double scene_bbp_ratio_mean, scene_bbp_ratio_std;
        double scene_bbp_ratio_max = *std::max_element(scene_bbp_ratio_vector.begin(), scene_bbp_ratio_vector.end());
        double scene_bbp_ratio_min = *std::min_element(scene_bbp_ratio_vector.begin(), scene_bbp_ratio_vector.end());
        std::tie(scene_bbp_ratio_mean, scene_bbp_ratio_std) = getMeanStd(scene_bbp_ratio_vector);

        // bounding boxes per image
        auto scene_bbp_per_image_vector = scenes_bbp_per_image[scene];
        double scene_bbp_per_image_mean, scene_bbp_per_image_std;
        double scene_bbp_per_image_max = *std::max_element(scene_bbp_per_image_vector.begin(), scene_bbp_per_image_vector.end());
        double scene_bbp_per_image_min = *std::min_element(scene_bbp_per_image_vector.begin(), scene_bbp_per_image_vector.end());
        std::tie(scene_bbp_per_image_mean, scene_bbp_per_image_std) = getMeanStd(scene_bbp_per_image_vector);

      // non-pedestrian bounding boxes

        // width
        auto scene_bbnp_width_vector = scenes_bbnp_width[scene];
        double scene_bbnp_width_mean, scene_bbnp_width_std;
        double scene_bbnp_width_max = *std::max_element(scene_bbnp_width_vector.begin(), scene_bbnp_width_vector.end());
        double scene_bbnp_width_min = *std::min_element(scene_bbnp_width_vector.begin(), scene_bbnp_width_vector.end());
        std::tie(scene_bbnp_width_mean, scene_bbnp_width_std) = getMeanStd(scene_bbnp_width_vector);

        // height
        auto scene_bbnp_height_vector = scenes_bbnp_height[scene];
        double scene_bbnp_height_mean, scene_bbnp_height_std;
        double scene_bbnp_height_max = *std::max_element(scene_bbnp_height_vector.begin(), scene_bbnp_height_vector.end());
        double scene_bbnp_height_min = *std::min_element(scene_bbnp_height_vector.begin(), scene_bbnp_height_vector.end());
        std::tie(scene_bbnp_height_mean, scene_bbnp_height_std) = getMeanStd(scene_bbnp_height_vector);

        // ratio
        auto scene_bbnp_ratio_vector = scenes_bbnp_ratio[scene];
        double scene_bbnp_ratio_mean, scene_bbnp_ratio_std;
        double scene_bbnp_ratio_max = *std::max_element(scene_bbnp_ratio_vector.begin(), scene_bbnp_ratio_vector.end());
        double scene_bbnp_ratio_min = *std::min_element(scene_bbnp_ratio_vector.begin(), scene_bbnp_ratio_vector.end());
        std::tie(scene_bbnp_ratio_mean, scene_bbnp_ratio_std) = getMeanStd(scene_bbnp_ratio_vector);

        // bounding boxes per image
        auto scene_bbnp_per_image_vector = scenes_bbnp_per_image[scene];
        double scene_bbnp_per_image_mean, scene_bbnp_per_image_std;
        double scene_bbnp_per_image_max = *std::max_element(scene_bbnp_per_image_vector.begin(), scene_bbnp_per_image_vector.end());
        double scene_bbnp_per_image_min = *std::min_element(scene_bbnp_per_image_vector.begin(), scene_bbnp_per_image_vector.end());
        std::tie(scene_bbnp_per_image_mean, scene_bbnp_per_image_std) = getMeanStd(scene_bbnp_per_image_vector);

      // other stats

      auto scene_overlapped_perc_vector = scenes_bbp_overlapped_perc[scene];
      auto scene_occupied_pixels_vector = scenes_bbp_occupied_pixels[scene];
      double scene_ocuppied_pixels = std::accumulate(scene_occupied_pixels_vector.begin(), scene_occupied_pixels_vector.end(), decltype(scene_occupied_pixels_vector)::value_type(0));
      double scene_occupied_pixels_perc = 100 * scene_ocuppied_pixels / (scenes_number_images[scene] * 1224 * 1024);

      overall_overlapped_perc      += std::accumulate(scene_overlapped_perc_vector.begin(), scene_overlapped_perc_vector.end(), decltype(scene_overlapped_perc_vector)::value_type(0));
      overall_occupied_pixels_perc += scene_ocuppied_pixels;

      auto scene_bbnp_overlapped_perc_vector = scenes_bbnp_overlapped_perc[scene];
      auto scene_bbnp_occupied_pixels_vector = scenes_bbnp_occupied_pixels[scene];
      double scene_bbnp_ocuppied_pixels = std::accumulate(scene_bbnp_occupied_pixels_vector.begin(), scene_bbnp_occupied_pixels_vector.end(), decltype(scene_occupied_pixels_vector)::value_type(0));
      double scene_bbnp_occupied_pixels_perc = 100 * scene_bbnp_ocuppied_pixels / (scenes_number_images[scene] * 1224 * 1024);

      overall_bbnp_overlapped_perc      += std::accumulate(scene_bbnp_overlapped_perc_vector.begin(), scene_bbnp_overlapped_perc_vector.end(), decltype(scene_overlapped_perc_vector)::value_type(0));
      overall_bbnp_occupied_pixels_perc += scene_bbnp_ocuppied_pixels;

      total_bbp  += scene_overlapped_perc_vector.size();
      total_bbnp += scenes_number_bbnp[scene];

      double scene_overlapped_perc_min = *std::min_element(scene_overlapped_perc_vector.begin(), scene_overlapped_perc_vector.end());
      double scene_overlapped_perc_max = *std::max_element(scene_overlapped_perc_vector.begin(), scene_overlapped_perc_vector.end());
      double scene_overlapped_perc_mean, scene_overlapped_perc_std;
      std::tie(scene_overlapped_perc_mean, scene_overlapped_perc_std) = getMeanStd(scene_overlapped_perc_vector);

      double scene_bbnp_overlapped_perc_min = *std::min_element(scene_bbnp_overlapped_perc_vector.begin(), scene_bbnp_overlapped_perc_vector.end());
      double scene_bbnp_overlapped_perc_max = *std::max_element(scene_bbnp_overlapped_perc_vector.begin(), scene_bbnp_overlapped_perc_vector.end());
      double scene_bbnp_overlapped_perc_mean, scene_bbnp_overlapped_perc_std;
      std::tie(scene_bbnp_overlapped_perc_mean, scene_bbnp_overlapped_perc_std) = getMeanStd(scene_bbnp_overlapped_perc_vector);

      std::cout << std::string(50, '-') << std::string(2,'\n');

      std::cout << "* " << scene << " (" << scenes_number_images[scene] << " images) "  << std::string(2, '\n');

      std::cout << "--> Bbp ("                         << scenes_number_bbp[scene] << ")"  << '\n';
      std::cout << "  - #Bbp per image      --> min: " << scene_bbp_per_image_min          << ", max: " << scene_bbp_per_image_max << ", mean: " << scene_bbp_per_image_mean << ", std: " << scene_bbp_per_image_std << '\n';
      std::cout << "  - Overlapped pxs %    --> min: " << scene_overlapped_perc_min       << ", max: " << scene_overlapped_perc_max << ", mean: " << scene_overlapped_perc_mean << ", std: " << scene_overlapped_perc_std << '\n';
      std::cout << "  - Occ pxs % bbp/total --> "      << scene_occupied_pixels_perc      << '\n';
      std::cout << "  - Occ pxs % bbp/bb    --> "      << 100 * scene_occupied_pixels_perc/(scene_occupied_pixels_perc + scene_bbnp_occupied_pixels_perc) << '\n';
      std::cout << "  - Weight              --> min: " << scene_bbp_width_min  << ", max: "    << scene_bbp_width_max  << ", mean: " << scene_bbp_width_mean  << ", std: " << scene_bbp_width_std << '\n';
      std::cout << "  - Height              --> min: " << scene_bbp_height_min << ", max: "    << scene_bbp_height_max << ", mean: " << scene_bbp_height_mean << ", std: " << scene_bbp_height_std << '\n';
      std::cout << "  - Ratio               --> min: " << scene_bbp_ratio_min  << ", max: "    << scene_bbp_ratio_max  << ", mean: " << scene_bbp_ratio_mean  << ", std: " << scene_bbp_ratio_std << std::string(2, '\n');

      std::cout << "--> Bbnp ("                       << scenes_number_bbnp[scene] << ")" << '\n';
      std::cout << "  - #Bbnp per image     --> min: "   << scene_bbnp_per_image_min         << ", max: " << scene_bbnp_per_image_max  << ", mean: " << scene_bbnp_per_image_mean  << ", std: " << scene_bbnp_per_image_std << '\n';
      std::cout << "  - Overlapped pxs %    --> min: "   << scene_bbnp_overlapped_perc_min   << ", max: " << scene_bbnp_overlapped_perc_max << ", mean: " << scene_bbnp_overlapped_perc_mean << ", std: " << scene_bbnp_overlapped_perc_std << '\n';
      std::cout << "  - Weight              --> min: " << scene_bbnp_width_min  << ", max: "    << scene_bbnp_width_max  << ", mean: " << scene_bbnp_width_mean  << ", std: " << scene_bbnp_width_std << '\n';
      std::cout << "  - Height              --> min: " << scene_bbnp_height_min << ", max: "    << scene_bbnp_height_max << ", mean: " << scene_bbnp_height_mean << ", std: " << scene_bbnp_height_std << '\n';
      std::cout << "  - Ratio               --> min: " << scene_bbnp_ratio_min  << ", max: "    << scene_bbnp_ratio_max  << ", mean: " << scene_bbnp_ratio_mean  << ", std: " << scene_bbnp_ratio_std << '\n';

      std::cout << '\n';

      std::cout << "--> % #bbp/bb: "                   << 100 * scenes_number_bbp[scene]/(scenes_number_bbp[scene] + scenes_number_bbnp[scene]) << std::string(2,'\n');
    }

    std::cout << std::string(50, '*') << std::string(2,'\n');

    std::cout << "* OVERALL" << " (" << total_images << " images) " << std::string(2, '\n');

    std::cout << "--> Bbp ("                          << total_bbp << ")" << '\n';
    std::cout << "  - #Bbp per image      --> min: " << bbp_per_image_min << ", max: " << bbp_per_image_max << ", mean: " << bbp_per_image_mean << ", std: " << bbp_per_image_std << '\n';
    std::cout << "  - Overlapped pxs %    --> "      << overall_overlapped_perc / total_bbp << '\n';
    std::cout << "  - Occ pxs % bbp/total --> "      << 100 * overall_occupied_pixels_perc / (total_images*1224*1024) << '\n';
    std::cout << "  - Occ pxs % bbp/bb    --> "      << 100 * overall_occupied_pixels_perc / (overall_bbnp_occupied_pixels_perc + overall_occupied_pixels_perc) << '\n';
    std::cout << "  - Width               --> min: " << bbp_width_min  << ", max: " << bbp_width_max  << ", mean: " << bbp_width_mean  << " and std: " << bbp_width_std << '\n';
    std::cout << "  - Height              --> min: " << bbp_height_min << ", max: " << bbp_height_max << ", mean: " << bbp_height_mean << " and std: " << bbp_height_std << '\n';
    std::cout << "  - Ratio               --> min: " << bbp_ratio_min  << ", max: " << bbp_ratio_max  << ", mean: " << bbp_ratio_mean  << " and std: " << bbp_ratio_std << std::string(2, '\n');

    std::cout << "--> Bbnp ("                        << total_bbnp << ")"   << '\n';
    std::cout << "  - #Bbnp per image     --> min: "    << bbnp_per_image_min  << ", max: " << bbnp_per_image_max  << ", mean: " << bbnp_per_image_mean  << ", std: " << bbnp_per_image_std  << '\n';
    std::cout << "  - Overl % bbnp        --> "         << overall_bbnp_overlapped_perc / total_bbnp << '\n';
    std::cout << "  - Width               --> min: " << bbnp_width_min  << ", max: " << bbnp_width_max  << ", mean: " << bbnp_width_mean  << " and std: " << bbnp_width_std << '\n';
    std::cout << "  - Height              --> min: " << bbnp_height_min << ", max: " << bbnp_height_max << ", mean: " << bbnp_height_mean << " and std: " << bbnp_height_std << '\n';
    std::cout << "  - Ratio               --> min: " << bbnp_ratio_min  << ", max: " << bbnp_ratio_max  << ", mean: " << bbnp_ratio_mean  << " and std: " << bbnp_ratio_std << '\n';

    std::cout << '\n';

    std::cout << "--> % #bbp/bb: "                   << 100 * total_bbp/(total_bbp + total_bbnp)   << std::string(2,'\n');

    Histogram width_histogram(0, bbp_width_max, 10);
    Histogram height_histogram(0, bbp_height_max, 10);
    Histogram ratio_histogram(0, bbp_ratio_max, 10);
    Histogram bbp_per_image_histogram(0, bbp_per_image_max, 10);

    for(auto const &width : bbp_width_vector) {
      width_histogram.addValue(width);
    }

    for(auto const &height : bbp_height_vector) {
      height_histogram.addValue(height);
    }

    for(auto const &ratio : bbp_ratio_vector) {
      ratio_histogram.addValue(ratio);
    }

    for(auto const &bbp_per_image : bbp_per_image_vector) {
      bbp_per_image_histogram.addValue(bbp_per_image);
    }

    if(!width_histogram.save("../results/histograms/bbp/txt/width_histogram.txt")) {
      std::cerr << "ERROR: Couldn't save width histogram" << '\n';
    }
    if(!height_histogram.save("../results/histograms/bbp/txt/height_histogram.txt")) {
      std::cerr << "ERROR: Couldn't save height histogram" << '\n';
    }
    if(!ratio_histogram.save("../results/histograms/bbp/txt/ratio_histogram.txt")) {
      std::cerr << "ERROR: Couldn't save ratio histogram" << '\n';
    }
    if(!bbp_per_image_histogram.save("../results/histograms/bbp/txt/bbp_per_image_histogram.txt")) {
      std::cerr << "ERROR: Couldn't save ratio histogram" << '\n';
    }

  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }

  return 0;
}
