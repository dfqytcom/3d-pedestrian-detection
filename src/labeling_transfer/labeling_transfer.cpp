/*
Auxiliar functions to transfer labels from RGB images onto 3D point clouds.

Main: transfer_labels.cpp
*/

#include "labeling_transfer/labeling_transfer.hpp"
#include "point_cloud_pedestrian_detector/point_cloud_utils.hpp"
#include "utils/maps.hpp"
#include "utils/boxes.hpp"
#include "utils/histogram.hpp"

#include <unordered_map>
#include <iostream>
#include <fstream>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

struct PointClouds
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcd; // original point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd_projected; // projected point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd_clusters; // clusters (pedestrians or non-pedestrians)
  std::vector<std::pair<pcl::PointXYZI, cv::Point>> points_pixels_projections; // to pair each point with its corresponding pixel (projection)

  PointClouds() :
      pcd(new pcl::PointCloud<pcl::PointXYZI>),
      pcd_projected(new pcl::PointCloud<pcl::PointXYZRGB>),
      pcd_clusters(new pcl::PointCloud<pcl::PointXYZRGB>)
  {
  }
};

// FUSION MATRICES
struct ProjectionMatrices
{
  // opencv matrices
  cv::Mat m_rgb_camera_matrix_cv = cv::Mat(3,3,CV_32F);
  cv::Mat m_rgb_distortion_coef_cv = cv::Mat(1,5,CV_32F);
  boost::numeric::ublas::matrix<float> m_rgb_projection_matrix;
  boost::numeric::ublas::matrix<float> m_rgb_camera_intrinsic_matrix;
  boost::numeric::ublas::matrix<float> m_rgb_camera_matrix;
  boost::numeric::ublas::matrix<float> m_rgb_fusion_matrix;

  // lidar matrices
  boost::numeric::ublas::matrix<float> m_lidar_matrix;
  boost::numeric::ublas::vector<float> m_lidar_translation;
  boost::numeric::ublas::matrix<float> m_lidar_homogenea;

  ProjectionMatrices() :
      m_rgb_projection_matrix(4,3),
      m_rgb_camera_intrinsic_matrix(3,3),
      m_rgb_camera_matrix(4,3),
      m_rgb_fusion_matrix(4,3),
      m_lidar_matrix(3,3),
      m_lidar_translation(4),
      m_lidar_homogenea(4,4)
  {
  }
};

//--------------------------------- Free functions ---------------------------------

// ----------------------- readRgbFusionMatrix -----------------------

void readRgbFusionMatrix(const std::string &projection_matrices_filename,
                         ProjectionMatrices &matrices)
{
  try {
    std::ifstream projection_matrices_file(projection_matrices_filename);
    std::string line;
    std::vector<std::string> matrix_values;

    while(!projection_matrices_file.eof()){
      std::getline(projection_matrices_file, line);
      matrix_values.clear();

      if(line == "----- LIDAR ROTATION MATRIX ----- "){
        std::getline(projection_matrices_file, line);
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_lidar_homogenea(0,0) = std::stof(matrix_values[0]);
        matrices.m_lidar_homogenea(1,0) = std::stof(matrix_values[1]);
        matrices.m_lidar_homogenea(2,0) = std::stof(matrix_values[2]);
        matrices.m_lidar_homogenea(0,3) = 0;

        std::getline(projection_matrices_file, line);
        matrix_values.clear();
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_lidar_homogenea(0,1) = std::stof(matrix_values[0]);
        matrices.m_lidar_homogenea(1,1) = std::stof(matrix_values[1]);
        matrices.m_lidar_homogenea(2,1) = std::stof(matrix_values[2]);
        matrices.m_lidar_homogenea(1,3) = 0;

        std::getline(projection_matrices_file, line);
        matrix_values.clear();
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_lidar_homogenea(0,2) = std::stof(matrix_values[0]);
        matrices.m_lidar_homogenea(1,2) = std::stof(matrix_values[1]);
        matrices.m_lidar_homogenea(2,2) = std::stof(matrix_values[2]);
        matrices.m_lidar_homogenea(2,3) = 0;
      }

      if(line == "----- LIDAR TRANSLATION VECTOR ----- "){
        std::getline(projection_matrices_file, line);
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_lidar_homogenea(3,0) = std::stof(matrix_values[0]);
        matrices.m_lidar_homogenea(3,1) = std::stof(matrix_values[1]);
        matrices.m_lidar_homogenea(3,2) = std::stof(matrix_values[2]);
        matrices.m_lidar_homogenea(3,3) = 1;

      }

      if(line == "----- CAMERA INTRINSIC MATRIX [fx 0 0; s fy 0; cx cy 1] ----- "){
        std::getline(projection_matrices_file, line);
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_intrinsic_matrix(0,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_intrinsic_matrix(0,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_intrinsic_matrix(0,2) = std::stof(matrix_values[2]);

        std::getline(projection_matrices_file, line);
        matrix_values.clear();
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_intrinsic_matrix(1,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_intrinsic_matrix(1,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_intrinsic_matrix(1,2) = std::stof(matrix_values[2]);

        std::getline(projection_matrices_file, line);
        matrix_values.clear();
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_intrinsic_matrix(2,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_intrinsic_matrix(2,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_intrinsic_matrix(2,2) = std::stof(matrix_values[2]);

        matrices.m_rgb_camera_matrix_cv.row(0).col(0) = matrices.m_rgb_camera_intrinsic_matrix(0,0);
        matrices.m_rgb_camera_matrix_cv.row(0).col(1) = 0;
        matrices.m_rgb_camera_matrix_cv.row(0).col(2) = matrices.m_rgb_camera_intrinsic_matrix(2,0);
        matrices.m_rgb_camera_matrix_cv.row(1).col(0) = matrices.m_rgb_camera_intrinsic_matrix(0,1);
        matrices.m_rgb_camera_matrix_cv.row(1).col(1) = matrices.m_rgb_camera_intrinsic_matrix(1,1);
        matrices.m_rgb_camera_matrix_cv.row(1).col(2) = matrices.m_rgb_camera_intrinsic_matrix(2,1);
        matrices.m_rgb_camera_matrix_cv.row(2).col(0) = matrices.m_rgb_camera_intrinsic_matrix(0,2);
        matrices.m_rgb_camera_matrix_cv.row(2).col(1) = matrices.m_rgb_camera_intrinsic_matrix(1,2);
        matrices.m_rgb_camera_matrix_cv.row(2).col(2) = matrices.m_rgb_camera_intrinsic_matrix(2,2);
      }

      if(line == "----- CAMERA DISTORTION VECTOR [k1 k2 p1 p2 k3] ----- "){
        std::getline(projection_matrices_file, line);
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_distortion_coef_cv.row(0).col(0) = std::stof(matrix_values[0]);
        matrices.m_rgb_distortion_coef_cv.row(0).col(1) = std::stof(matrix_values[1]);
        matrices.m_rgb_distortion_coef_cv.row(0).col(2) = std::stof(matrix_values[2]);
        matrices.m_rgb_distortion_coef_cv.row(0).col(3) = std::stof(matrix_values[3]);
        matrices.m_rgb_distortion_coef_cv.row(0).col(4) = std::stof(matrix_values[4]);
      }

      if(line == "----- CAMERA ROTATION MATRIX ----- "){
        std::getline(projection_matrices_file, line);
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_matrix(0,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_matrix(0,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_matrix(0,2) = std::stof(matrix_values[2]);

        std::getline(projection_matrices_file, line);
        matrix_values.clear();
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_matrix(1,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_matrix(1,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_matrix(1,2) = std::stof(matrix_values[2]);

        std::getline(projection_matrices_file, line);
        matrix_values.clear();
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_matrix(2,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_matrix(2,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_matrix(2,2) = std::stof(matrix_values[2]);
      }

      if(line == "----- CAMERA TRANSLATION VECTOR ----- "){
        std::getline(projection_matrices_file, line);
        boost::split(matrix_values, line, boost::is_any_of("\t"));
        matrices.m_rgb_camera_matrix(3,0) = std::stof(matrix_values[0]);
        matrices.m_rgb_camera_matrix(3,1) = std::stof(matrix_values[1]);
        matrices.m_rgb_camera_matrix(3,2) = std::stof(matrix_values[2]);
      }
    }

    //multiplicamos las matrices cuando acabamos de leer
    matrices.m_rgb_projection_matrix = boost::numeric::ublas::prod(matrices.m_rgb_camera_matrix, matrices.m_rgb_camera_intrinsic_matrix);
    matrices.m_rgb_fusion_matrix = boost::numeric::ublas::prod(matrices.m_lidar_homogenea, matrices.m_rgb_projection_matrix);

    return;
  }
  catch(const std::ifstream::failure &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- outputTransferStats -----------------------

void outputTransferStats(TransferParameters &params)
{
  int clusters_above_1024_points = 0;

  long total_points = 0;

  for (auto const & number_of_points : params.number_of_points) {
    total_points += number_of_points;

    if (number_of_points >= 1024) {
      ++clusters_above_1024_points;
    }
  }

  long mean = total_points / params.number_of_points.size();
  long deviation = 0;

  for(auto const & number_of_points : params.number_of_points) {
    deviation += pow((number_of_points - mean), 2);
  }

  deviation = sqrt(deviation/params.number_of_points.size());

  int max = *std::max_element(params.number_of_points.begin(), params.number_of_points.end());
  int min = *std::min_element(params.number_of_points.begin(), params.number_of_points.end());

  std::cout << '\n';
  std::cout << "*** TRANSFER STATS ***" << std::string(2, '\n');;
  std::cout << "  - Total #clusters (with valid projection): " << params.number_of_points.size() << std::string(2, '\n');
  std::cout << "  - #Clusters with +=1024 points: " << clusters_above_1024_points << std::string(2, '\n');
  std::cout << "  - Mean #points: " << mean << " / Standard deviation #points: " << deviation << std::string(2, '\n');
  std::cout << "  - Max #points of a cluster: " << max << std::string(2, '\n');;
  std::cout << "  - Min #points of a cluster: " << min << std::string(2, '\n');;

  std::cout << "* Histogram of the number of points per cluster not saved. Uncomment these lines to save it." << std::string(2, '\n');

  // Histogram histogram(0, max, 10);
  // for(auto const &points : params.number_of_points) {
  //   histogram.addValue(points);
  // }
  // std::string histogram_path = "../results/histograms/" + params.type + "/txt/" + params.type + "_points_per_cluster_histogram.txt";
  // if(histogram.save(histogram_path)) {
  //   std::cout << "*** Histogram saved in " << histogram_path << " ***" << std::string(2, '\n');
  // }
}

// ----------------------- resultsSceneMapExists -----------------------

bool resultsSceneMapExists(TransferParameters &params)
{
  try {
    // path to the folder that contains the maps of a specific dataset
    std::string dataset_maps_path = params.results_maps_path + params.dataset;
    boost::filesystem::directory_iterator it{dataset_maps_path};

    while(it != boost::filesystem::directory_iterator{}) {
      std::string map_path = (*it).path().string();
      std::string map_name = (*it++).path().filename().string();

      // if map exists
      if(map_name.find(params.scene + "_") != map_name.npos) {
        params.results_scene_map_path = map_path;
        return true;
      }
    }

    // if not
    return false;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- displayPointClouds -----------------------

void displayPointClouds(const TransferParameters &params,
                        PointClouds &point_clouds,
                        pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  // cloud with black points (white background)
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcd_black (new pcl::PointCloud<pcl::PointXYZRGB>);
  paintBlackPointCloud(point_clouds.pcd, pcd_black);
  viewer->addPointCloud<pcl::PointXYZRGB>(pcd_black);
  viewer->spin(); // press 'q' to skip
  viewer->removeAllPointClouds();

  // projected point cloud
  viewer->addPointCloud<pcl::PointXYZRGB>(point_clouds.pcd_projected);
  viewer->spin();
  viewer->removeAllPointClouds();

  // clusters (pedestrians or non-pedestrians)
  viewer->addPointCloud<pcl::PointXYZRGB>(point_clouds.pcd_clusters);
  viewer->spin();
  viewer->removeAllPointClouds();

  return;
}

// ----------------------- displayImage -----------------------

bool displayImage(TransferParameters &params,
                  const cv::Mat &image_undistorted)
{
  // set the image and point cloud paths as the title of the figure
  std::string figure_title = "Image: " + params.image_path + ", Point cloud: " + params.point_cloud_path;

  cv::namedWindow(figure_title, cv::WINDOW_NORMAL);
  cv::resizeWindow(figure_title, 1920, 1080);
  cv::imshow(figure_title, image_undistorted);
  // cv::imwrite("example.png", image_undistorted); // uncomment to save the image with YOLO detections, LIDAR FOV and intersection areas

  // wait until 'ESC', 'd', 's', 'space bar' or 'n' key is pressed
  int key_pressed;
  do {
    key_pressed = cv::waitKey();
  }
  while(key_pressed != 32 && key_pressed != 27 && key_pressed != 100 && key_pressed != 115 && key_pressed != 110);
  cv::destroyAllWindows();

  // if 'space bar' is pressed --> change image and display the point cloud projections (return true)
  if(key_pressed == 32) {
    return true;
  }

  // if 'ESC', 'd', 's' or 'n' is pressed the point cloud projections are not displayed (return false)
  else {
    if (key_pressed == 27) { // ESC --> stop
      params.display_all = 0;
    }
    else if (key_pressed == 100) { // d --> change dataset
      params.display_dataset = 0;
    }
    else if (key_pressed == 115) { // s --> change scene
      params.display_scene = 0;
    }
    return false;
  }
}

// ----------------------- getBoxesIntersectionAreas -----------------------

void getBoxesIntersectionAreas(const TransferParameters &params,
                               std::vector<Results_box> &boxes,
                               cv::Mat &image_undistorted)
{
  // "create" the LIDAR FOV and fill aux_image_A with it
  std::vector<cv::Point> FOV_vertices;
  FOV_vertices.push_back(params.FOV_top_left);
  FOV_vertices.push_back(params.FOV_top_right);
  FOV_vertices.push_back(params.FOV_bot_right);
  FOV_vertices.push_back(params.FOV_bot_left);
  cv::Mat aux_image_A = cv::Mat::zeros(image_undistorted.rows, image_undistorted.cols, CV_32F);
  fillConvexPoly(aux_image_A, FOV_vertices, FOV_vertices.size());

  // iterate through YOLO detections
  auto boxes_it = boxes.begin();
  while(boxes_it != boxes.end()) {

    auto box = (*boxes_it);

    // only transfer YOLO detections with a confidence > confidence_threshold (e.g. 0.8)
    if(box.getConfidence() >= params.confidence_threshold) {

      // "create" a YOLO bounding box and fill aux_image_B with it
      std::vector<cv::Point> box_vertices;
      box_vertices.push_back(cv::Point(box.getLeftX(), box.getTopY()));
      box_vertices.push_back(cv::Point(box.getRightX(), box.getTopY()));
      box_vertices.push_back(cv::Point(box.getRightX(), box.getBotY()));
      box_vertices.push_back(cv::Point(box.getLeftX(), box.getBotY()));
      cv::Mat aux_image_B = cv::Mat::zeros(image_undistorted.rows, image_undistorted.cols, CV_32F);
      fillConvexPoly(aux_image_B, box_vertices, box_vertices.size());

      // aux_image_C --> image filled with the intersection between the LIDAR FOV (aux_image_A) and a bounding box (aux_image_B)
      cv::Mat aux_image_C;
      cv::bitwise_and(aux_image_A, aux_image_B, aux_image_C);

      // intersection area (percentage) = pixels from aux_image_C that are non zero divided by the bounding box area
      float intersection_area = cv::countNonZero(aux_image_C) / box.getArea();

      // if we are transfering pedestrians
      if (params.type == "bbp") {

        // if the intersection_area > threshold (e.g. 0.7)
        if(intersection_area >= params.box_intersection_area_threshold) {

          // draw the bounding box on the image to be displayed (image_undistorted) as green
          cv::polylines(image_undistorted, box_vertices, true, cv::Scalar(0, 255, 0), 2);
          ++boxes_it;
        }

        // if intersection_area < threshold (e.g. 0.7) --> draw it as red and erase the bounding box
        else {
          cv::polylines(image_undistorted, box_vertices, true, cv::Scalar(0, 0, 255), 2);
          boxes_it = boxes.erase(boxes_it);
        }
      }

      // if we are transfering non-pedestrians --> don't need the intersection threshold (could be defined but we don't do it)
      else if (params.type == "bbnp") {
        cv::polylines(image_undistorted, box_vertices, true, cv::Scalar(0, 255, 0), 2);
        ++boxes_it;
      }

      // draw two labels per each bounding box: intersection area (top) and confidence (bottom)
      int top = box.getTopY();
      int left = box.getLeftX();
      int height = box.getHeight();
      int area_base_line, conf_base_line;

      std::string area_label = cv::format("Area: %.2f", intersection_area);
      std::string conf_label = cv::format("conf: %.2f", box.getConfidence());

      cv::Size area_label_size = getTextSize(area_label, cv::FONT_ITALIC, 0.5, 1, &area_base_line);
      cv::Size conf_label_size = getTextSize(conf_label, cv::FONT_ITALIC, 0.5, 1, &conf_base_line);

      top = (top > area_label_size.height) ? top : area_label_size.height;
      top = (top > conf_label_size.height) ? top : conf_label_size.height;

      cv::rectangle(image_undistorted, cv::Point(left, top - round(1.5*area_label_size.height)), cv::Point(left + round(1.5*area_label_size.width), top + area_base_line), cv::Scalar(255, 255, 255), cv::FILLED);
      cv::putText(image_undistorted, area_label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);

      cv::rectangle(image_undistorted, cv::Point(left, (top+height) - round(1.5*conf_label_size.height)), cv::Point(left + round(1.5*conf_label_size.width), (top+height) + conf_base_line), cv::Scalar(255, 255, 255), cv::FILLED);
      cv::putText(image_undistorted, conf_label, cv::Point(left, top+height), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
    }

    // if confidence < confidence_threshold --> erase the bounding box
    else {
      boxes_it = boxes.erase(boxes_it);
    }
  }

  // draw the LIDAR FOV on the image to be displayed (image_undistorted)
  cv::polylines(image_undistorted, FOV_vertices, true, cv::Scalar(255, 0, 0), 2);

  return;
}

// ----------------------- getClusters -----------------------

void getClusters(TransferParameters &params,
                std::vector<Results_box> &boxes,
                PointClouds &point_clouds)
{
  // pair bounding box (YOLO detection) index with the 3D points corresponding to that bounding box
  std::unordered_map<int, std::vector<pcl::PointXYZ>> box_index_points;

  // pair bounding box index with 2D box coordinates
  std::unordered_map<int, std::vector<int>> box_index_coords2d;

  for(int box_index = 0; box_index < boxes.size(); ++box_index){
    auto box = boxes[box_index];
    std::vector<int> coords2d {int(box.getLeftX()), int(box.getTopY()), int(box.getRightX()), int(box.getBotY())};
    box_index_coords2d[box_index] = coords2d;
  }

  // std::cout << "------------------------------" << '\n';
  // iterate through all points and corresponding projections (pixels)
  for(auto const &point_pixel : point_clouds.points_pixels_projections) {

    pcl::PointXYZ point;
    point.x = point_pixel.first.x;
    point.y = point_pixel.first.y;
    point.z = point_pixel.first.z;

    cv::Point projected_pixel = point_pixel.second;

    pcl::PointXYZRGB cluster_rgb_point;
    cluster_rgb_point.x = point.x;
    cluster_rgb_point.y = point.y;
    cluster_rgb_point.z = point.z;

    // default point color: white (white background)
    cv::Vec3b cluster_rgb_point_color = cv::Vec3b(255, 255, 255);

    // iterate through bounding boxes (YOLO detections)
    // *friendly reminder: the bounding boxes with low confidence or low intersection area were erased in "getBoxesIntersectionAreas" function
    for(int box_index = 0; box_index < boxes.size(); ++box_index){
      auto box = boxes[box_index];
      int cluster_box_margin = 0; // If we don't want more context per each bounding box
      // int cluster_box_margin = (box.getHeight()/174)*20; // Spain average height 1.74m; 20cm of margin (bigger bounding box)

      // if the point projection (pixel) is inside a bounding box (could be inside more than one)
      if((projected_pixel.x >= box.getLeftX() - cluster_box_margin) &&
         (projected_pixel.y >= box.getTopY() - cluster_box_margin) &&
         (projected_pixel.x <= box.getLeftX() + box.getWidth() + cluster_box_margin) &&
         (projected_pixel.y <= box.getTopY() + box.getHeight() + cluster_box_margin)) {

           // point color: black
           cluster_rgb_point_color = cv::Vec3b(0, 0, 0);

           // pair the corresponding point with the bounding box using the box_index as the identifier ('save' purposes)
           box_index_points[box_index].push_back(point);
      }
    }

    // store the point in "point_clouds.pcd_clusters" ('display' purposes)
    cluster_rgb_point.rgb = *reinterpret_cast<float*>(&cluster_rgb_point_color);
    point_clouds.pcd_clusters->points.push_back(cluster_rgb_point);
  }

  // identifier to save each cluster to a different file
  int cluster_id = 0;

  // iterate through the bounding boxes indices
  for(auto const & it : box_index_points) {
    auto box_index = it.first;
    auto points = it.second;

    // store the number of points that each cluster has
    params.number_of_points.push_back(points.size());

    // 'save' mode
    if (params.option == "save") {

      // if the cluster has more than 1024 points --> save it
      if(points.size() >= 1024) {

        // point cloud that contains the points of one cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr aux_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        aux_point_cloud->insert(aux_point_cloud->begin(), points.begin(), points.end());

        pcl::io::savePLYFile(params.clusters_path + params.dataset + "/" + params.scene + "/" + params.image_filename + "_" + std::to_string(cluster_id) + ".ply", *aux_point_cloud);

        auto box_coord2d = box_index_coords2d[box_index];
        std::ofstream ofs (params.coords2d_path + params.dataset + "/" + params.scene + "/" + params.image_filename + "_" + std::to_string(cluster_id) + ".txt", std::ofstream::out);
        ofs << box_coord2d[0] << " " << box_coord2d[1] << " " << box_coord2d[2] << " " << box_coord2d[3];

        ++cluster_id;
      }
    }
  }

  return;
}

// ----------------------- getProjectedPixel -----------------------

cv::Point getProjectedPixel(const ProjectionMatrices &matrices,
                            pcl::PointXYZI point)
{
  int posx = -1;
  int posy = -1;

  boost::numeric::ublas::vector<float> position_vector(4);
  boost::numeric::ublas::vector<float> world_point(3);

  position_vector(0) = point.x;
  position_vector(1) = point.y;
  position_vector(2) = point.z;
  position_vector(3) = 1.0;

  world_point = boost::numeric::ublas::prec_prod(position_vector, matrices.m_rgb_fusion_matrix);

  posx = world_point(0)/world_point(2);
  posy = world_point(1)/world_point(2);

  return cv::Point(posx, posy);
}

// ----------------------- getProjectedPointCloud -----------------------

void getProjectedPointCloud(TransferParameters &params,
                            const ProjectionMatrices &matrices,
                            PointClouds &point_clouds,
                            cv::Mat &image_undistorted)
{
  for(auto const &point : point_clouds.pcd->points) {
    pcl::PointXYZRGB projected_point;
    projected_point.x = point.x;
    projected_point.y = point.y;
    projected_point.z = point.z;

    uint32_t rgb;
    cv::Vec3b color_rgb;

    cv::Point projected_pixel = getProjectedPixel(matrices, point);

    if(projected_pixel.x < image_undistorted.cols &&
       projected_pixel.y < image_undistorted.rows &&
       projected_pixel.x >= 0 &&
       projected_pixel.y >= 0) {

      // pair each point to its corresponding projection (pixel)
      point_clouds.points_pixels_projections.push_back(std::make_pair(point, projected_pixel));

      color_rgb.val[0] = image_undistorted.at<cv::Vec3b>(projected_pixel.y,projected_pixel.x)[0];
      color_rgb.val[1] = image_undistorted.at<cv::Vec3b>(projected_pixel.y,projected_pixel.x+1)[1];
      color_rgb.val[2] = image_undistorted.at<cv::Vec3b>(projected_pixel.y,projected_pixel.x+2)[2];
      rgb = (static_cast<uint32_t>(color_rgb.val[2])<<16 | static_cast<uint32_t>(color_rgb.val[1])<<8 | static_cast<uint32_t>(color_rgb.val[0]));

      // to get the LIDAR FOV vertices
      if((projected_pixel.x < params.FOV_top_left.x)  && (projected_pixel.y < params.FOV_top_left.y)) {
        params.FOV_top_left = projected_pixel;
      }
      if((projected_pixel.x > params.FOV_top_right.x) && (projected_pixel.y < params.FOV_top_right.y)) {
        params.FOV_top_right = projected_pixel;
      }
      if((projected_pixel.x > params.FOV_bot_right.x) && (projected_pixel.y > params.FOV_bot_right.y)) {
        params.FOV_bot_right = projected_pixel;
      }
      if((projected_pixel.x < params.FOV_bot_left.x)  && (projected_pixel.y > params.FOV_bot_left.y)) {
        params.FOV_bot_left = projected_pixel;
      }
    }

    projected_point.rgb = *reinterpret_cast<float*>(&rgb);
    point_clouds.pcd_projected->push_back(projected_point);
  }
  return;
}

// ----------------------- transferImage -----------------------

void transferImage(TransferParameters &params,
                   const ProjectionMatrices &matrices,
                   std::vector<Results_box> &boxes,
                   pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  cv::Mat image = cv::imread(params.image_path);
  cv::Mat image_undistorted;
  cv::undistort(image,  image_undistorted , matrices.m_rgb_camera_matrix_cv, matrices.m_rgb_distortion_coef_cv);

  PointClouds point_clouds;

  // load the point cloud that corresponds to this image
  loadPointCloud(params.point_cloud_path, point_clouds.pcd);

  // project the point cloud onto the image to get the pixels corresponding to each point of the cloud
  getProjectedPointCloud(params, matrices, point_clouds, image_undistorted);

  // compute the intersection area between each bounding box and the LIDAR FOV
  // it the intersection area < box_intersection_area_threshold --> erase the bounding box
  getBoxesIntersectionAreas(params, boxes, image_undistorted);

  // get the points which projections (pixels) are inside a bounding box --> 3D clusters (if 'save' mode --> save them)
  getClusters(params, boxes, point_clouds);

  // 'display' mode
  if(params.display_scene && params.display_dataset && params.display_all) {

    // if 'space bar' key is pressed, displayImage returns True and the point clouds (original, projection and 3D clusters) are displayed
    if(displayImage(params, image_undistorted)) {
      displayPointClouds(params, point_clouds, viewer);
    }
  }
  return;
}

// ----------------------- transferScene -----------------------

void transferScene(TransferParameters &params,
                   const ProjectionMatrices &matrices,
                   pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  std::cout << "  · Scene " << params.scene << '\n';

  // check if results (pedestrian or non-pedestrian bounding boxes) map of a specific dataset and scene exists
  if (!resultsSceneMapExists(params)) {
    std::cerr << "ERROR: No such " << params.dataset << "/" << params.scene << " results_map" << std::endl;
    return;
  }

  // load the map that matches the images with their corresponding point clouds
  std::unordered_map<std::string, std::string> matching_map = loadMatchingMap(params.matching_map_path);

  // deserialize the map
  std::unordered_map<std::string, std::vector<Results_box>> results_map;
  std::ifstream results_map_file(params.results_scene_map_path);
  boost::archive::text_iarchive ia{results_map_file};
  ia >> results_map;

  for (auto const &results_map_it : results_map) {

    // bbp map example: "20200110_140011_122.txt"
    if (params.type == "bbp") {
      params.image_filename = results_map_it.first.substr(0, results_map_it.first.find("."));
    }

    // bbnp map example: "scene1/20200110_135858_596.txt"
    else if(params.type == "bbnp") {
      std::string s = results_map_it.first.substr(results_map_it.first.find("/") + 1); // 20200110_135858_596.txt
      params.image_filename = s.substr(0, s.find(".")); // 20200110_135858_596
    }

    // discard image 20200110_135503_101 from Outdoor - scene1 (calibration issues)
    if(params.image_filename != "20200110_135503_101") {
      std::vector<Results_box> boxes = results_map_it.second;

      // path to the specific image
      params.image_path = params.media_path + params.dataset + "/" + params.scene + "/" + params.image_filename + ".png";

      // check if the path to the corresponding (matched) point cloud exists
      params.point_cloud_path = matching_map[params.image_path];
      if(boost::filesystem::exists(params.point_cloud_path)) {
        if(params.point_cloud_path == "/home/oscar/media/pointcloud/datasets/outdoor/scene1/20200110_135506_107.bin"){
            transferImage(params, matrices, boxes, viewer);
        }
      }
      else {
        std::cerr << "ERROR: No such point cloud " << params.point_cloud_path << std::endl;
      }
    }

    // if you are in 'display mode' and pressed 'ESC', 'd' or 's' while displaying the transfering results --> return
    if(params.option == "display" && (!params.display_all || !params.display_dataset || !params.display_scene)) {
      return;
    }
  }

  return;
}

// ----------------------- transferDataset -----------------------

void transferDataset(TransferParameters &params,
                     const ProjectionMatrices &matrices,
                     pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  try{

    std::cout << '\n' << "- Dataset " << params.dataset << '\n';

    // to store the path to each scene map (of one specific dataset)
    std::vector<boost::filesystem::path> matching_maps_paths;

    std::copy(boost::filesystem::directory_iterator(params.matching_maps_path + params.dataset),
      boost::filesystem::directory_iterator(), std::back_inserter(matching_maps_paths));

    // sort paths alphabetically (1, 10, 2, 3, 4..)
    std::sort(matching_maps_paths.begin(), matching_maps_paths.end());

    for(auto const &matching_map_path : matching_maps_paths) {

      params.matching_map_path = matching_map_path.string();

      // scene# (1, 2, ...)
      params.scene = matching_map_path.filename().string().substr(0, matching_map_path.filename().string().find("_"));

      // need this 'if' to display the next dataset when you press 'd' while displaying the results
      if(params.option == "display") {
        params.display_scene = 1;
      }

      std::cout << params.scene << '\n';

      if(boost::filesystem::exists(params.media_path + params.dataset + "/" + params.scene)) {
        transferScene(params, matrices, viewer);
      }

      // if you are in 'display mode' and pressed 'ESC' or 'd' while displaying the transfering results --> return
      if(params.option == "display" && (!params.display_all || !params.display_dataset)) {
        return;
      }
    }
    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}

// ----------------------- transferLabels -----------------------

void transferLabels(TransferParameters &params,
                    pcl::visualization::PCLVisualizer::Ptr &viewer)
{
  try {

    std::cout << '\n' << "Transfering labels..." << '\n';

    // load fusion matrices
    ProjectionMatrices matrices;
    readRgbFusionMatrix(params.projection_matrices_path, matrices);

    // to store the path to each dataset folder (highway, indoor and outdoor)
    std::vector<boost::filesystem::path> datasets_paths;

    std::copy(boost::filesystem::directory_iterator(params.matching_maps_path),
      boost::filesystem::directory_iterator(), std::back_inserter(datasets_paths));

    // sort paths alphabetically
    std::sort(datasets_paths.begin(), datasets_paths.end());

    // iterate through the datasets folders
    for(auto const &dataset_path : datasets_paths) {

      //highway, indoor or outdoor
      params.dataset = dataset_path.filename().string();

      // need this 'if' to display the next dataset when you press 'd' while displaying the results
      if(params.option == "display") {
        params.display_dataset = 1;
      }

      // the outdoor dataset is the only one without transfering issues atm
      if(params.dataset == "outdoor") {
        transferDataset(params, matrices, viewer);
      }

      // if you are in 'display mode' and pressed 'ESC' while displaying the transfering results --> stop
      if(params.option == "display" && !params.display_all) {
        return;
      }
    }
    return;
  }
  catch (const boost::filesystem::filesystem_error &e)
  {
    std::cerr << "ERROR: " << e.what() << '\n';
  }
}
