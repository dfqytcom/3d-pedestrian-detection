#pragma once

#include <vector>
#include <string>

struct Histogram {
  int min, max, num_elem, threshold;
  std::vector<std::pair<int, int>> hist;

  Histogram(int min,
            int max,
            int num_elem);

  void addValue(int value);
  int save(const std::string &histogram_path);

};
