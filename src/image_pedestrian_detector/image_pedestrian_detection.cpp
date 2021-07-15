/*
Auxiliar functions to detect pedestrians in RGB images using YOLOv3.

Main: image_detect_pedestrians.cpp
*/

#include "image_pedestrian_detector/image_pedestrian_detection.hpp"

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/dnn.hpp>

// DetectorParameters constructor

DetectorParameters::DetectorParameters(const std::string &opt,
                                       const std::string &d_path,
                                       const std::string &r_path) :
    option(opt),
    datasets_path(d_path),
    results_path(r_path),
    display_scene(0),
    display_dataset(0),
    display_all(0)
{
}

struct Box
{
  int top, left, width, height;
  float confidence;

  Box(int t,
      int l,
      int w,
      int h,
      float c) :
    top(t),
    left(l),
    width(w),
    height(h),
    confidence(c)
  {
  }
};

//--------------------------------- Free functions ---------------------------------

// ----------------------- saveResults -----------------------

void saveResults(const cv::Mat &image,
                const DetectorParameters &params,
                Box &box,
                std::ofstream &output_file)
{
  // [classId (person: 0)] [center_x] [center_y] [width] [height] [confidence]
  output_file << "0 " << (float)(box.left + box.width/2)/image.cols << " " << (float)(box.top + box.height/2)/image.rows << " " <<
    (float)box.width/image.cols << " " << (float)box.height/image.rows << " " << box.confidence << std::endl;

  return;
}

// ----------------------- drawBoxes -----------------------

void drawBoxes(cv::Mat &image,
               DetectorParameters &params,
               int box_class,
               Box &box,
               bool last_box)
{
  // if(box.confidence >= 0.8) { // uncomment if you only want to display bounding boxes with confidence > some threshold (e.g. 0.8)

    // draw the bounding box
    cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.left+box.width, box.top+box.height),cv::Scalar(0, 255, 0), 2);

    // display label at the top of the bounding box showing class (person) and YOLO confidence
    std::string label = params.classes[box_class] + ": " + cv::format("%.2f", box.confidence);

    boost::to_upper(label);

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);

    (box.top > labelSize.height) ? box.top = box.top : box.top = labelSize.height;

    cv::rectangle(image, cv::Point(box.left, box.top - round(1.5*labelSize.height)),
      cv::Point(box.left + round(1.5*labelSize.width), box.top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);

    cv::putText(image, label, cv::Point(box.left, box.top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
  // }

  // if it is the last box of the image --> display the image
  if(last_box) {
    cv::destroyAllWindows();
    cv::namedWindow(params.image_path,  cv::WINDOW_NORMAL);
    cv::resizeWindow(params.image_path, 1920, 1080);
    cv::imshow(params.image_path, image);
    // cv::imwrite("example.png", image); // uncomment to save the image with YOLO detections

    // wait until 'ESC', 'd', 's' or 'space bar' key is pressed
    int key_pressed;
    do {
      key_pressed = cv::waitKey();
    }
    while(key_pressed != 32 && key_pressed != 27 && key_pressed != 100 && key_pressed != 115);
    cv::destroyAllWindows();

    if (key_pressed == 27) { // ESC --> stop
      params.display_all = 0;
    }
    else if (key_pressed == 100) { // d --> change dataset
      params.display_dataset = 0;
    }
    else if (key_pressed == 115) { // s --> change scene
      params.display_scene = 0;
    }
  }
  return;
}

// ----------------------- outputResults -----------------------

void outputResults(cv::Mat &image,
                   DetectorParameters &params,
                   std::vector<int> final_class,
                   const std::vector<cv::Rect> &final_boxes,
                   const std::vector<float> &final_confidences)
{
  std::ofstream output_file;

  // 'save' mode
  if(params.option == "save") {

    // e.g. "/home/oscar/media/image/results/outdoor/scene1/"
    std::string save_results_path = params.results_path + params.dataset + "/" + params.scene + "/";

    // if the folder doesn't exist -> create it (create_directories will create the full path)
    if(!boost::filesystem::is_directory(save_results_path)) {
      boost::filesystem::create_directories(save_results_path);
    }

    // e.g. "/home/oscar/media/image/results/outdoor/scene1/20200110_135735_406.txt"
    output_file.open(save_results_path + params.image_filename.substr(0, params.image_filename.find(".")) + ".txt");
  }

  bool last_box = false;

  for(int i=0; i<final_boxes.size(); ++i){
    int top = final_boxes[i].y;
    int left = final_boxes[i].x;
    int width = final_boxes[i].width;
    int height = final_boxes[i].height;
    float confidence = final_confidences[i];

    Box box(top, left, width, height, confidence);

    // 'save' mode
    if(params.option == "save") {
      saveResults(image, params, box, output_file);
    }

    // 'display' mode
    else if (params.display_scene && params.display_dataset && params.display_all) {
      if(i == final_boxes.size()-1) {
        last_box = true;
      }
      drawBoxes(image, params, final_class[i], box, last_box);
    }
  }
}

// ----------------------- processImage -----------------------

void processImage(cv::Mat &image,
                  DetectorParameters &params,
                  cv::dnn::Net &yolo_net)
{
  // generate 4D blob from input image
  cv::Mat blob;
  std::vector<cv::Mat> net_output;
  double scalefactor = 1/255.0;
  cv::Size size = cv::Size(416, 416);
  cv::Scalar mean = cv::Scalar(0,0,0);
  bool swapRB = false;
  bool crop = false;
  cv::dnn::blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop);

  // get names of output layers
  std::vector<cv::String> names;
  std::vector<int> outLayers = yolo_net.getUnconnectedOutLayers(); // get indices of output layers, i.e. layers with unconnected outputs
  std::vector<cv::String> layersNames = yolo_net.getLayerNames(); // get names of all layers in the network

  names.resize(outLayers.size());
  for (size_t i = 0; i < outLayers.size(); ++i) { // get the names of the output layers in names
    names[i] = layersNames[outLayers[i] - 1];
  }

  // invoke forward propagation through network
  yolo_net.setInput(blob);
  yolo_net.forward(net_output, names);

  // scan through all bounding boxes and keep only the ones with high confidence
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (size_t i = 0; i < net_output.size(); ++i)
  {
    float* data = (float*)net_output[i].data;
    for (int j = 0; j < net_output[i].rows; ++j, data += net_output[i].cols)
    {
      cv::Mat scores = net_output[i].row(j).colRange(5, net_output[i].cols);
      cv::Point classId;
      double confidence;

      // get the value and location of the maximum score
      cv::minMaxLoc(scores, 0, &confidence, 0, &classId);

      // if it's a person
      if(classId.x == 0 && classId.y == 0)
      {
        // if YOLO confidence > defined threshold
        if (confidence > params.conf_threshold)
        {
          cv::Rect box; int cx, cy;
          cx = (int)(data[0] * image.cols);
          cy = (int)(data[1] * image.rows);
          box.width = (int)(data[2] * image.cols);
          box.height = (int)(data[3] * image.rows);
          box.x = cx - box.width/2; // left
          box.y = cy - box.height/2; // top

          if(boxes.size() > 0) {
            bool outside = true;
            for(int k=0; k<boxes.size(); ++k){
                if( (box.x > boxes[k].x) && (box.y > boxes[k].y) &&
                        (box.y < (boxes[k].y + boxes[k].height)) &&
                        (box.x < (boxes[k].x + boxes[k].width))  &&
                        ((box.x + box.width) < (boxes[k].x + boxes[k].width)) &&
                        ((box.y + box.height) < (boxes[k].y + boxes[k].height))){

                    if(classIds[k] == classId.x){
                        outside = false;
                        break;
                    }
                }
            }
            if(outside){
              boxes.push_back(box);
              classIds.push_back(classId.x);
              confidences.push_back((float)confidence);
            }

          }
          else {
            boxes.push_back(box);
            classIds.push_back(classId.x);
            confidences.push_back((float)confidence);
          }
        }
      }
    }
  }

  // perform non-maxima suppression
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, params.conf_threshold, params.nms_threshold, indices);

  std::vector<cv::Rect> final_boxes;
  std::vector<int> final_class;
  std::vector<float> final_confidences;

  for(auto it=indices.begin(); it!=indices.end(); ++it) {
    final_boxes.push_back(boxes[*it]);
    final_class.push_back(classIds[*it]);
    final_confidences.push_back(confidences[*it]);
  }

  // save or display results
  outputResults(image, params, final_class, final_boxes, final_confidences);

  return;
}

//--------------------------------- image_pedestrian_detection.hpp ---------------------------------

// ----------------------- loadYoloData -----------------------

void loadYoloData(DetectorParameters &params,
                  cv::dnn::Net &yolo_net)
{
  params.yolo_base_path = "../yolo/";
  params.yolo_classes_file = params.yolo_base_path + "coco.names";
  params.yolo_model_configuration = params.yolo_base_path + "yolov3.cfg";
  params.yolo_model_weights = params.yolo_base_path + "yolov3.weights";
  params.conf_threshold = 0.3;
  params.nms_threshold = 0.3;
  params.classes.clear();

  std::ifstream ifs(params.yolo_classes_file.c_str());
  std::string line;
  while (getline(ifs, line)){
      params.classes.push_back(line);
  }

  // load neural network
  yolo_net = cv::dnn::readNetFromDarknet(params.yolo_model_configuration, params.yolo_model_weights);
  yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

// ----------------------- processScene -----------------------

void processScene(DetectorParameters &params,
                  cv::dnn::Net &yolo_net)
{
  // to store the path to each image (of one specific scene and dataset)
  std::vector<boost::filesystem::path> images_paths;

  std::copy(boost::filesystem::directory_iterator(params.datasets_path + params.dataset + "/" + params.scene + "/"),
    boost::filesystem::directory_iterator(), std::back_inserter(images_paths));

  // sort paths alphabetically
  std::sort(images_paths.begin(), images_paths.end());

  // iterate the images
  for(auto const &image_path : images_paths) {

    // full path to image
    params.image_path = image_path.string();

    // image filename
    params.image_filename = image_path.filename().string();

    // load image
    cv::Mat image = cv::imread(params.image_path);

    // process specific image
    processImage(image, params, yolo_net);

    // if you are in 'display mode' and pressed 'ESC', 'd' or 's' while displaying the results --> return
    if(params.option == "display" && (!params.display_all || !params.display_dataset || !params.display_scene)) {
      return;
    }
  }
  return;
}

// ----------------------- processDataset -----------------------

void processDataset(DetectorParameters &params,
                    cv::dnn::Net &yolo_net)
{
  // to store the path to each scene folder (of one specific dataset)
  std::vector<boost::filesystem::path> scenes_paths;

  std::copy(boost::filesystem::directory_iterator(params.datasets_path + params.dataset),
    boost::filesystem::directory_iterator(), std::back_inserter(scenes_paths));

  // sort paths alphabetically (1, 10, 2, 3, 4..)
  std::sort(scenes_paths.begin(), scenes_paths.end());

  // iterate through the scenes folders
  for(auto const &scene_path : scenes_paths) {

    // scene# (1, 2, ...)
    params.scene = scene_path.filename().string();

    // need this 'if' to display the next scene when you press 's' while displaying the results
    if(params.option == "display") {
      params.display_scene = 1;
    }

    std::cout << "  · Scene " << params.scene << '\n';

    // process specific scene
    processScene(params, yolo_net);

    // if you are in 'display mode' and pressed 'ESC' or 'd' while displaying the results --> return
    if(params.option == "display" && (!params.display_all || !params.display_dataset)) {
      return;
    }
  }
  return;
}

// ----------------------- processAll -----------------------

void processAll(DetectorParameters &params,
                cv::dnn::Net &yolo_net)
{
  std::cout << '\n' << "Detecting pedestrians..." << '\n';

  // to store the path to each dataset folder (highway, indoor and outdoor)
  std::vector<boost::filesystem::path> datasets_paths;

  std::copy(boost::filesystem::directory_iterator(params.datasets_path),
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

    std::cout <<  '\n' << "- Dataset " << params.dataset << '\n';

    // process specific dataset
    processDataset(params, yolo_net);

    // if you are in 'display mode' and pressed 'ESC' while displaying the results --> stop
    if(params.option == "display" && !params.display_all) {
      return;
    }
  }
  return;
}
