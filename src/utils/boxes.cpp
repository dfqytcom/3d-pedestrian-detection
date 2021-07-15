/*
Functions to manage the ground truth (Gt) and YOLO detections (Results) bounding boxes.
*/

#include "utils/boxes.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const int image_cols = 1224;
const int image_rows = 1024;

//-----------------------------------Box-----------------------------------

// Box constructors

Box::Box()
{
}

Box::Box(int id,
         float x,
         float y,
         float w,
         float h) :
    classId(id),
    center_x(x),
    center_y(y),
    width(w),
    height(h)
{
}


float Box::getClassId()
{
  return classId;
}

float Box::getLeftX()
{
  return (center_x-width/2)*image_cols;
}

float Box::getRightX()
{
  return Box::getLeftX() + Box::getWidth();
}

float Box::getTopY()
{
  return (center_y-height/2)*image_rows;
}

float Box::getBotY()
{
  return Box::getTopY() + Box::getHeight();
}

float Box::getWidth()
{
  return width*image_cols;
}

float Box::getHeight()
{
  return height*image_rows;
}

float Box::getArea()
{
  return (width*image_cols+1)*(height*image_rows+1);
}

void Box::drawBox(cv::Mat &image, bool match)
{
}

//----------------------------------Gt_box-----------------------------------

// Gt_box constructors

Gt_box::Gt_box() :
    Box::Box()
{
}

Gt_box::Gt_box(int id,
               float x,
               float y,
               float w,
               float h) :
    Box::Box(id,
             x,
             y,
             w,
             h)
{
}

// Gt_box drawBox

void Gt_box::drawBox(cv::Mat &image,
                     bool match)
{
  int top = getTopY();
  int left = getLeftX();
  int width = getWidth();
  int height = getHeight();

  // if the ground truth bounding box was correctly detected by YOLO --> true positive (blue)
  if(match) {
    cv::rectangle(image, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(255, 0, 0), 2);
  }

  // if not --> false negative (yellow)
  else {
    cv::rectangle(image, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 255, 255), 2);

    // Display label at the top of the bounding box
    std::string label = "FN";

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);

    top = (top > labelSize.height) ? top : labelSize.height;

    cv::rectangle(image, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(image, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
  }
}

//--------------------------------Results_box--------------------------------

// Results_box constructors

Results_box::Results_box() : Box::Box(){}

Results_box::Results_box(int id, float x, float y, float w, float h, float c)
  : Box::Box(id, x, y, w, h),
    confidence(c)
  {
  }

float Results_box::getConfidence() {
  return confidence;
}


// Results_box drawBox

void Results_box::drawBox(cv::Mat &image, bool match) {
  int top = getTopY();
  int left = getLeftX();
  int width = getWidth();
  int height = getHeight();

  // if the YOLO detection is correct --> true positive (green)
  if(match) {
    cv::rectangle(image, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 255, 0), 2);
    std::string label = "TP";

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);

    top = (top > labelSize.height) ? top : labelSize.height;

    cv::rectangle(image, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(image, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
  }

  // if not --> false positive (red)
  else {
    cv::rectangle(image, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 0, 255), 2);

    // label showing the confidence of the bounding box
    std::string label = "FP, " + cv::format("%.2f", getConfidence());

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);

    top = (top > labelSize.height) ? top : labelSize.height;

    cv::rectangle(image, cv::Point(left, (top+height) - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), (top+height) + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(image, label, cv::Point(left, top+height), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);
  }
}
