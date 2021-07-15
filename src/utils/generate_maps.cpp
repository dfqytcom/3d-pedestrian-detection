/*
Main function to generate maps.
  - Ground truth: bounding boxes manually labeled.
  - Results: YOLO detections.
  - Matching: images paired with their corresponding point clouds.
*/

#include "utils/maps.hpp"

#include <iostream>

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  if ((argc != 2) ||
      ((std::string(argv[1]) != "gt") &&
       (std::string(argv[1]) != "results") &&
       (std::string(argv[1]) != "matching") &&
       (std::string(argv[1]) != "all")))
  {
    std::cout << "Usage: [" << argv[0] << "] [gt/results/matching/all]" << std::endl;
    return 0;
  }

  std::string mode = argv[1];
  if ((mode == "gt") || (mode == "all")) {
    generateGtMaps();
  }
  if ((mode == "results") || (mode == "all")) {
    generateResultsMaps();
  }
  if ((mode == "matching") || (mode == "all")) {
    generateMatchingMaps();
  }

  return 1;
}
