/*
Functions to organize .ply bbp and bbnp clusters to save them as a .h5 file easily (with "ply_to_hdf5.py" python script).
*/

#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <random>

#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  try
  {
    auto rng = std::default_random_engine {};

    // paths to the train/eval/test datasets containing the .ply 3D clusters
    std::string ply_datasets_paths = std::string(std::getenv("MEDIA")) + "/pointcloud/clusters_1024_norm/";

    // path to save the .txt files (that contain the clusters paths needed to create the HDF5 files)
    std::string h5_path = std::string(std::getenv("MEDIA"))  + "/pointcloud/hdf5/";

    // if it doesn't exist --> create it
    if (!boost::filesystem::is_directory(h5_path)) {
      boost::filesystem::create_directories(h5_path);
      std::cout << '\n' << "* "<< h5_path << " directory created" << '\n';
    }

    std::string dataset, type;

    boost::filesystem::directory_iterator ply_datasets_it{ply_datasets_paths};
    for(auto & ply_dataset_path : ply_datasets_it) {

      // train / eval / test
      dataset = ply_dataset_path.path().filename().string();

      // path to the pedestrian/not_pedestrian folders of this dataset (train, eval or test)
      std::string bbp_dir = ply_dataset_path.path().string() + "/pedestrian";
      std::string bbnp_dir = ply_dataset_path.path().string() + "/not_pedestrian";

      // to store the bbp and bbnp paths (separated)
      std::vector<boost::filesystem::path> bbp_paths, bbnp_paths;

      // copy the bbp and bbnp paths
      std::copy(boost::filesystem::directory_iterator(bbnp_dir), boost::filesystem::directory_iterator(), std::back_inserter(bbnp_paths));
      std::copy(boost::filesystem::directory_iterator(bbp_dir),  boost::filesystem::directory_iterator(), std::back_inserter(bbp_paths));

      // to store the bbp and bbnp paths (merged)
      std::vector<boost::filesystem::path> ply_paths;
      ply_paths.insert(ply_paths.end(), bbnp_paths.begin(), bbnp_paths.end());
      ply_paths.insert(ply_paths.end(), bbp_paths.begin() , bbp_paths.end());
      std::shuffle(ply_paths.begin(), ply_paths.end(), rng);

      // output .txt file (HDF5)
      std::ofstream h5_file(h5_path + "ply_data_" + dataset + "_0_id2file.txt", std::ofstream::out);

      // iterate bbp and bbnp paths
      for(auto const ply_path : ply_paths) {
        std::string ply_filename;

        // if it's a bbnp cluster
        if(ply_path.string().find("/not_pedestrian/") != std::string::npos) {
          ply_filename = ply_path.string().substr(ply_path.string().find("not_pedestrian"));
          h5_file << ply_filename << '\n';
        }

        // if it's a bbp cluster
        else {
          ply_filename = ply_path.string().substr(ply_path.string().find("pedestrian"));
          h5_file << ply_filename << '\n';
        }
      }
      std::cout << '\n' << "- File " << h5_path << "ply_data_" << dataset << "_0_id2file.txt created successfully" << '\n';
    }

    return 1;
  }

  catch(const std::ifstream::failure &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}
