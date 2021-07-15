/*
Function to check if the point clouds (pedestrian and non-pedestrian) are properly normalized:
  - Centered             --> centroid coordinates = (0,0,0)
  - Inside a unit sphere --> euclidean distance from the center (0,0,0) to the farthest point <= 1.0
*/

#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include <pcl/common/centroid.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <boost/filesystem.hpp>

// ----------------------- main -----------------------

int main(int argc,
     char const *argv[])
{
  try {
    double farthest_centroid_x = 0;
    double farthest_centroid_y = 0;
    double farthest_centroid_z = 0;
    double farthest_distance = 0;

    int checked_clouds = 0;

    // path to the normalized clouds
    std::string normalized_clouds_path = std::string(std::getenv("MEDIA")) + "/pointcloud/clusters_1024_norm/";

    boost::filesystem::recursive_directory_iterator normalized_clouds_it(normalized_clouds_path);

    // iterate through the clouds (pedestrian and non-pedestrian)
    for(auto const & normalized_cloud_path : normalized_clouds_it) {

      // if it's a regular file --> it's not a folder --> it's a point cloud (.ply)
      if(boost::filesystem::is_regular_file(normalized_cloud_path)) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPLYFile(normalized_cloud_path.path().string(), *cloud);

        // get the centroid of the cloud
        Eigen::Vector4f cloud_centroid;
        pcl::compute3DCentroid(*cloud, cloud_centroid);

        // to store the farthest centroid coordinates of all point clouds
        // ⁺ if they are properly normalized, the farthest centroid coordinates should be (0,0,0)
        if(cloud_centroid[0] > farthest_centroid_x) {
          farthest_centroid_x = cloud_centroid[0];
        }
        if(cloud_centroid[1] > farthest_centroid_y) {
          farthest_centroid_y = cloud_centroid[1];
        }
        if(cloud_centroid[2] > farthest_centroid_z) {
          farthest_centroid_z = cloud_centroid[2];
        }

        // to get the euclidean distance from the origin (0,0,0) to the farthest point of all point clouds
        // * if they are properly normalized, the maximum distance should be <= 1.0
        for(auto &point : cloud->points) {
          double distance = sqrt(pow(abs(point.x), 2) + pow(abs(point.y), 2) + pow(abs(point.z), 2));
          if(distance > farthest_distance) {
            farthest_distance = distance;
          }
        }
        ++checked_clouds;
      }

      else {
        std::cout << '\n' << "- Checking " << normalized_cloud_path.path().filename().string() << '\n';
      }
    }

    // output the checking results

    std::cout << '\n' << "*** Point clouds normalization checking ***" << std::string(2, '\n');
    std::cout << "- " << checked_clouds << " clouds checked" << std::string(2, '\n');
    std::cout << "- Max centroid coordinates (x, y, z): ("  << farthest_centroid_x << ", " << farthest_centroid_y << ", " << farthest_centroid_z << ")" << std::string(2, '\n');
    std::cout << "- Farthest distance: "      << farthest_distance << std::string(2, '\n');

    return 1;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}
