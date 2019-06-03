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
    std::vector<cv::Point2f> points;
    std::vector<double> ratio;
    std::vector<cv::Point2f> shape;
    std::vector<double> dist;
    std::vector<float> knots;
    std::vector<cv::Point2f> controls;
    cv::Mat Nx;
    std::vector<cv::Point2f> pts;
    int quantize;
};


float N(const std::vector<float> &knot, float t, int k, int q)
{
    if (q == 1) return (t >= knot[k] && t < knot[k + 1]) ? 1.f : 0.f;

    float div1 = knot[k + q - 1] - knot[k];
    float div2 = knot[k + q] - knot[k + 1];
    float val1 = (div1 != 0) ? (t - knot[k]) * N(knot, t, k, q - 1) / div1 : 0;
    float val2 = (div2 != 0) ? (knot[k + q] - t) * N(knot, t, k + 1, q - 1) / div2 : 0;

    return val1 + val2;
}

cv::Point2f ComputePoint(const cv::Mat& Nx, const std::vector<cv::Point2f>& controls, int x)
{
    cv::Point2f ret;
    for (int i = 0; i < Nx.rows; i++) {
        ret += Nx.at<float>(i, x) * controls[i];
    }
    return ret;
}


void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
    static int activeCP = -1;
    cv::Point2f currentP(x, y);
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
            ud->controls[activeCP] = currentP;
            ud->controls[(activeCP + ud->points.size())%ud->controls.size()] = currentP;
            ud->ratio[activeCP] = cv::norm(currentP - ud->shape[33]) / ud->dist[activeCP];
            orig.copyTo(image);
            for (auto pt : ud->points)
            {
                cv::circle(image, pt, 2, cv::Scalar(0, 255, 0), 3);
            }
            for (int i = 0; i < 16; i++)
            {
                double val = ud->ratio[i];
                cv::line(image, cv::Point2d(w/1.5 + i / 16.0 * w/3.0, val * 100), cv::Point2d(w/1.5 + i / 16.0 * w/3.0, 0), cv::Scalar(255), 5);
            }

            for(int i = ud->knots[1] * ud->quantize; i < ud->knots[ud->knots.size() - 3] * ud->quantize; i++)
            {
                ud->pts[i] = ComputePoint(ud->Nx, ud->controls, i);
                cv::circle(image, ud->pts[i], 1, cv::Scalar(255), 2);
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

    UserData ud;

    if(!faces_dlib.empty()) {
        dlib::full_object_detection shape_dlib = pose_model(dlibimg, faces_dlib[0]);
        for (unsigned j = 0; j < shape_dlib.num_parts(); j++) ud.shape.push_back(cv::Point(shape_dlib.part(j).x(), shape_dlib.part(j).y()));
    }
    auto& face = faces_dlib[0];
//    cv::rectangle(img, cv::Point{(int)face.left(), (int)face.top()},
//                  cv::Point{(int)face.right(), (int)face.bottom()}, cv::Scalar{255,0,0});



    ud.image = img;


    int ptSize = 16;
    ud.points.resize(ptSize);
    ud.dist.resize(ptSize);
    for (int i = 0; i < ptSize; i++)
    {
        auto pt = ud.shape[i];
        ud.points[i] = pt;
        ud.dist[i] = cv::norm(pt - ud.shape[33]);
    }

    cv::RotatedRect elps = fitEllipse( cv::Mat(ud.points) );
    cv::ellipse(img, elps, cv::Scalar(50,50,50), 2, 8);



    img.copyTo(ud.orig);
    for (auto pt : ud.points)
    {
        cv::circle(img, pt, 2, cv::Scalar(0, 255, 0), 3);
    }

    ud.ratio = std::vector<double>(ptSize, 1.0);

    ud.controls = ud.points;
    int n_u_add = 3;
    int n_u = ud.controls.size() + n_u_add, n_v = 4, n_u_active = n_u - n_u_add;
    ud.controls.resize(n_u);
    for (int i = 0; i < n_u_add; i++)
    {
        ud.controls[n_u - i - 1] = ud.controls[n_u_add - 1 - i];
    }
    const int q = 4;
    int kx_size = n_u + q;
    ud.knots.resize(kx_size, 0.f);
    for (int i = 0; i < kx_size; i++) {
        ud.knots[i] = i / (kx_size - 1.f);
    }
    ud.quantize = 1024;
    ud.Nx = cv::Mat(n_u, ud.quantize, CV_32FC1);
    for (int i = 0; i < n_u; i++) {
        for (int t = 0; t < ud.quantize; t++) {
            ud.Nx.at<float>(i, t) = N(ud.knots, t / (ud.quantize - 1.f), i, q);
        }
    }
    ud.pts.resize(ud.quantize);
    for(int i = ud.knots[2] * ud.quantize; i < ud.knots[ud.knots.size() - 1] * ud.quantize; i++)
    {
        ud.pts[i] = ComputePoint(ud.Nx, ud.controls, i);
        cv::circle(img, ud.pts[i], 1, cv::Scalar(255), 2);
    }
    cv::imshow("show", img);


    cv::setMouseCallback("show", mouse_callback, &ud);
    cv::waitKey();
    return 0;
}
