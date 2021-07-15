#pragma once

#include <boost/serialization/base_object.hpp>
#include <opencv2/core/types.hpp>

class Box
{
  public:
    Box();
    Box(int id,
        float x,
        float y,
        float w,
        float h);
    ~Box() = default;

    float getClassId();
    float getLeftX();
    float getRightX();
    float getTopY();
    float getBotY();
    float getWidth();
    float getHeight();
    float getArea();

    virtual void drawBox(cv::Mat &image,
                         bool match);

  private:

    // serialization

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar,
                   const unsigned int version)
    {
      ar & classId;
      ar & center_x;
      ar & center_y;
      ar & width;
      ar & height;
    }

    int classId;
    float center_x;
    float center_y;
    float width;
    float height;
};


class Gt_box : public Box
{
  public:
    Gt_box();
    Gt_box(int id,
           float x,
           float y,
           float w,
           float h);
    ~Gt_box() = default;

    void drawBox(cv::Mat &image,
                 bool match);
};


class Results_box : public Box
{
  public:
    Results_box();
    Results_box(int id,
                float x,
                float y,
                float w,
                float h,
                float c);
    ~Results_box() = default;

    float getConfidence();

    void drawBox(cv::Mat &image,
                bool match);

  private:

    // serialization (results boxes also have a confidence variable)

    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive &ar,
                   const unsigned int version)
    {
      ar & boost::serialization::base_object<Box>(*this);
      ar & confidence;
    }

    float confidence;
};
