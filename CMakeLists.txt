cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(pedestrian_detector)

find_package(PCL 1.3 REQUIRED)
find_package(OpenCV 4.1 REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${OpenCV_DEFINITIONS})

set(MAPS_SOURCES src/utils/maps.cpp src/utils/boxes.cpp include/utils/maps.hpp include/utils/boxes.hpp)
add_executable(generate_maps ${MAPS_SOURCES} src/utils/generate_maps.cpp)
target_link_libraries(generate_maps ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

set(EVALUATE_RESULTS_SOURCES ${MAPS_SOURCES} src/image_pedestrian_detector/evaluation.cpp include/image_pedestrian_detector/evaluation.hpp)
add_executable(evaluate_results ${EVALUATE_RESULTS_SOURCES} src/image_pedestrian_detector/evaluate_results.cpp)
target_link_libraries(evaluate_results ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable(get_precision_recall ${EVALUATE_RESULTS_SOURCES} src/image_pedestrian_detector/get_precision_recall.cpp)
target_link_libraries(get_precision_recall ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

set(IMAGE_PEDESTRIAN_DETECTOR_SOURCES ${MAPS_SOURCES} src/image_pedestrian_detector/image_pedestrian_detection.cpp include/image_pedestrian_detector/image_pedestrian_detection.hpp)
add_executable (image_detect_pedestrians ${IMAGE_PEDESTRIAN_DETECTOR_SOURCES} src/image_pedestrian_detector/image_detect_pedestrians.cpp)
target_link_libraries (image_detect_pedestrians ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

set(TRANSFER_SOURCES src/labeling_transfer/labeling_transfer.cpp src/point_cloud_pedestrian_detector/point_cloud_utils.cpp src/utils/histogram.cpp include/labeling_transfer/labeling_transfer.hpp include/point_cloud_pedestrian_detector/point_cloud_utils.hpp include/utils/histogram.hpp)
add_executable (transfer_labels ${TRANSFER_SOURCES} ${MAPS_SOURCES} src/labeling_transfer/transfer_labels.cpp)
target_link_libraries (transfer_labels ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable (display_point_cloud src/point_cloud_pedestrian_detector/display_point_cloud.cpp src/point_cloud_pedestrian_detector/point_cloud_utils.cpp include/point_cloud_pedestrian_detector/point_cloud_utils.hpp)
target_link_libraries (display_point_cloud ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable (get_boxes_stats src/utils/get_boxes_stats.cpp src/utils/boxes.cpp src/utils/maps.cpp src/utils/histogram.cpp include/utils/boxes.hpp include/utils/maps.hpp include/utils/histogram.hpp)
target_link_libraries (get_boxes_stats ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable (generate_non_pedestrian_boxes src/image_pedestrian_detector/generate_non_pedestrian_boxes.cpp src/utils/boxes.cpp src/utils/maps.cpp src/utils/histogram.cpp include/utils/boxes.hpp include/utils/maps.hpp include/utils/histogram.hpp)
target_link_libraries (generate_non_pedestrian_boxes ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable(preprocess_bbp_point_clouds src/point_cloud_pedestrian_detector/preprocess_bbp_point_clouds.cpp src/utils/maps.cpp src/utils/boxes.cpp include/utils/maps.hpp include/utils/boxes.hpp)
target_link_libraries (preprocess_bbp_point_clouds ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})

add_executable(organize_ply_hdf5 src/point_cloud_pedestrian_detector/organize_ply_hdf5.cpp)
target_link_libraries (organize_ply_hdf5 ${Boost_LIBRARIES})

add_executable(check_normalized_point_clouds src/point_cloud_pedestrian_detector/check_normalized_point_clouds.cpp)
target_link_libraries (check_normalized_point_clouds ${PCL_LIBRARIES} ${Boost_LIBRARIES})


add_executable(bin_to_ply src/point_cloud_pedestrian_detector/bin_to_ply.cpp)
target_link_libraries (bin_to_ply ${PCL_LIBRARIES} ${Boost_LIBRARIES})
