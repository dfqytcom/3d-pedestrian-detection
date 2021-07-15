#include <iostream>

#include <pcl/common/common_headers.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <boost/filesystem.hpp>

// ----------------------- loadPointCloud -----------------------

void loadPointCloud(const std::string &pcd_path,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &pcd)
{
  try {
    pcl::PointXYZI curr_point;
    int32_t number_of_points, value;

    std::ifstream point_cloud_file(pcd_path, std::ios::binary | std::ios::in);

    point_cloud_file.read(reinterpret_cast<char *>(&number_of_points), sizeof(number_of_points));
    double cnt = 0;
    for(int i=0; i<number_of_points; ++i){
      ++cnt;
      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      curr_point.x = value;

      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      curr_point.y = value;

      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      curr_point.z = value;

      point_cloud_file.read(reinterpret_cast<char *>(&value), sizeof(value));
      curr_point.intensity = value;

      pcd->points.push_back(curr_point);
    }

    std::cout << pcd->points.size() << '\n';

    return;
  }
  catch(const std::ifstream::failure &e) {
    std::cerr << "ERROR LOADING POINT CLOUD: " << e.what() << '\n';
  }
}

// ----------------------- main -----------------------

int main(int argc,
         char const *argv[])
{
  if (argc != 3) {
    std::cout << "Usage: [" << argv[0] << "] <path_to_bin_folder> <path_to_ply_folder>" << std::endl;
    return 0;
  }

  std::string bin_folder = argv[1]; // folder containing .bin files
  std::string ply_folder = argv[2]; // folder to save .ply files

  // path to the .bin folder
  boost::filesystem::directory_iterator path_it(bin_folder);
  for(auto const & path : path_it) {

    // if it's a regular file --> it's not a folder --> it's a point cloud
    if(boost::filesystem::is_regular_file(path)) {

      // if it's a .bin point_cloud
      if (path.path().string().find(".bin") != std::string::npos) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZI>);
        loadPointCloud(path.path().string(), point_cloud);

        std::string ply_filename = path.path().filename().string().substr(0, path.path().filename().string().find(".")) + ".ply";
        pcl::io::savePLYFile(ply_folder + '/' + ply_filename, *point_cloud);
      }
    }
  }
  return 1;
}
