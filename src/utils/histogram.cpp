/*
Utility functions to get and save an histogram from a vector.
*/

#include "utils/histogram.hpp"

#include <iostream>
#include <fstream>
#include <string>

// Histogram constructor

Histogram::Histogram(int min,
          int max,
          int num_elem) :
    min(min),
    max(max),
    num_elem(num_elem), // number of bins
    threshold((max-min)/(num_elem*2)) // to delimitate each bin
{
  // create the bins and initialize their values to 0
  // * if min = 0, max = 5 and num_elem (bins) = 3 --> bins: 0, 2.5, 5
  for (int i = 0; i <= num_elem; ++i) {
    hist.push_back(std::make_pair(i*(max-min)/num_elem, 0));
  }
}

// ----------------------- addValue -----------------------

void Histogram::addValue (int value)
{
  // if value falls inside a range --> ++ counter
  for(auto &it : Histogram::hist) {
    int range = it.first;
    if((value >= range - Histogram::threshold) &&
       (value <= range + Histogram::threshold)) {
      ++it.second;
      return;
    }
  }
  return;
}

// ----------------------- save -----------------------

int Histogram::save(const std::string &histogram_path)
{
  try{
    // save histogram bins and values into .txt file
    std::ofstream histogram_file(histogram_path);
    for(auto const &it : Histogram::hist) {
      histogram_file << it.first << "," << it.second << '\n';
    }
    return 1;
  }
  catch(const std::ifstream::failure &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 0;
  }
}
