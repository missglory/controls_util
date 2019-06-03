#include <iostream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;

struct UserData{
    cv::Mat image;
    cv::Mat orig;
    std::vector<cv::Point2d> points;
};



void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
    static int activeCP = -1;
    cv::Point2d currentP(x, y);
    UserData* ud = (UserData*)userdata;

    cv::Mat& image		= ud->image;
    cv::Mat& orig		= ud->orig;
    float w			= image.cols;
    float h			= image.rows;

    if (event == cv::EVENT_LBUTTONUP) {
        activeCP = -1;
    }
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        for (int i = 0; i < 16; i++) {
            if (cv::norm(currentP - ud->points[i]) < 5) {
                activeCP = i;
                break;
            }
        }
    }

    if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
        if (activeCP > -1)
        {
            ud->points[activeCP] = currentP;
            orig.copyTo(image);
            for (auto pt : ud->points)
            {
                cv::circle(image, pt, 2, cv::Scalar(0, 255, 0), 5);
            }
            cv::imshow("show", image);
        }
    }

}



int main()
{
    dlib::frontal_face_detector dlib_detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    try {
        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;
    }
    catch (...) {
        return false;
    }

    cv::namedWindow("show", cv::WINDOW_FREERATIO);
    cv::Mat img = cv::imread("../boy.jpg");
    cv::Mat imggray;
    int ch = img.channels();
    int tp = img.type();
    cv::cvtColor(img, imggray, cv::COLOR_RGB2GRAY);
    ch = imggray.channels();
    tp = imggray.type();

    dlib::cv_image<uchar> dlibimg(imggray);
    std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlibimg);

    std::vector<cv::Point2d> shape;
    if(!faces_dlib.empty()) {
        dlib::full_object_detection shape_dlib = pose_model(dlibimg, faces_dlib[0]);
        for (unsigned j = 0; j < shape_dlib.num_parts(); j++) shape.push_back(cv::Point(shape_dlib.part(j).x(), shape_dlib.part(j).y()));
        for (auto pt : shape)
        {
            cv::circle(img, pt, 2, cv::Scalar(255));
        }
    }
    auto& face = faces_dlib[0];
//    cv::rectangle(img, cv::Point{(int)face.left(), (int)face.top()},
//                  cv::Point{(int)face.right(), (int)face.bottom()}, cv::Scalar{255,0,0});



    cv::imshow("show", img);
    UserData ud;
    img.copyTo(ud.orig);
    ud.image = img;
    int ptSize = 16;
    ud.points.resize(16);
    for (int i = 0; i < 16; i++)
    {
        ud.points[i] = shape[i];
    }
    cv::setMouseCallback("show", mouse_callback, &ud);
    cv::waitKey();
    return 0;
}
