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
#include <math.h>
using namespace std;


struct UserData{
    cv::Mat image;
    cv::Mat orig;
    std::vector<cv::Point2f> points;
    std::vector<double> ratio;
    std::vector<cv::Point2f> shape;
    std::vector<double> dist;
    std::vector< std::vector<cv::Point> > controls;
    int quantize;
    cv::Mat Nx;
    std::vector<float> knots;
    std::vector<cv::Point2f> pts;
    cv::RotatedRect ellipse;
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

cv::Point2f ComputePoint(const cv::Mat& Nx, const std::vector<cv::Point>& controls, int x)
{
    float norm;
    for (int i = 0; i < Nx.rows; i++) {
        cv::Point2f control = {(float)controls[i].x, (float)controls[i].y};
        norm += Nx.at<float>(i, x);
    }
    if (norm < 0.99999)
        return cv::Point2f{-1.f, -1.f};

    cv::Point2f ret;
    for (int i = 0; i < Nx.rows; i++) {
        cv::Point2f control = {(float)controls[i].x, (float)controls[i].y};
        ret += Nx.at<float>(i, x) * control;
    }
    return ret;
}


void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
    static int activeCP = -1;
    cv::Point currentP(x, y);
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
        for (int i = 0; i < ud->controls[0].size() - 8; i++) {
            if (cv::norm(currentP - ud->controls[0][i]) < 5) {
                activeCP = i;
                break;
            }
        }
    }

    if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
        if (activeCP > -1)
        {
            ud->controls[0][activeCP] = currentP;
            if (activeCP < 8 || activeCP >= ud->controls[0].size() - 8)
            {
                int secondInd = (activeCP + ud->controls[0].size() - 8)%ud->controls[0].size();
                ud->controls[0][secondInd] = currentP;
            }
            ud->ratio[activeCP] = cv::norm(currentP - cv::Point(ud->shape[33])) / ud->dist[activeCP];
            orig.copyTo(image);


            for(int i = ud->knots[4] * (ud->quantize-1); i < ud->knots[ud->knots.size() - 4] * (ud->quantize-1); i++)
            {
                ud->pts[i] = ComputePoint(ud->Nx, ud->controls[0], i);

                if (ud->pts[i].x > 0.f)
                {
                    float t = 3.14f * i / ud->quantize * 2.f + ud->ellipse.angle / 180.f * 3.14;
                    cv::Point2f pt(ud->ellipse.size.width/2.f * std::cos(t),
                                   ud->ellipse.size.height/2.f * std::sin(t));
                    ud->ratio[i] = cv::norm(ud->pts[i] - ud->ellipse.center) / cv::norm(pt);
                    double val = ud->ratio[i];
                    cv::line(image, cv::Point2d(1 + w/1.1f * i / ud->quantize, val * 60),
                                    cv::Point2d(1 + w/1.1f * i / ud->quantize, 0), cv::Scalar(255), 2);

//                     cv::circle(image, ud->pts[i], 1, cv::Scalar(255), 2);
                }

            }

            std::vector<std::vector<cv::Point> > pnts(1);
            pnts.reserve(ud->quantize / 2);
            for (int i = 0; i < ud->pts.size(); i++)
            {
                if (ud->pts[i].x > 0.f) pnts[0].push_back(cv::Point(ud->pts[i]));
            }
            cv::drawContours(image, pnts, 0, cv::Scalar(0,255,0), 2);

            for (int i = 0; i < ud->controls[0].size(); i++)
            {
                auto pt = ud->controls[0][i];
                cv::circle(image, pt, 2, cv::Scalar(0, 0, 255), 3);
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

    ud.ellipse = fitEllipse( cv::Mat(ud.points) );
    cv::ellipse(img, ud.ellipse, cv::Scalar(50,50,50), 2, 8);

    img.copyTo(ud.orig);
    for (int i = 0; i < ud.points.size(); i++)
    {
        auto pt = ud.points[i];
//        cv::circle(img, pt, 2, cv::Scalar(255 - (float)i/ud.points.size() * 255, (float)i/ud.points.size() * 255, 0), 3);
    }


    int controlsSize = 50;
    ud.controls.resize(1);
    ud.controls[0].resize(controlsSize);
    for (int i = 0; i < controlsSize; i++)
    {
        float t = 3.14f * i / controlsSize * 2.f + ud.ellipse.angle / 180.f * 3.14;
        ud.controls[0][i] = ud.ellipse.center + cv::Point2f(ud.ellipse.size.width/2.f * std::cos(t),
                                                    ud.ellipse.size.height/2.f * std::sin(t)) * 1.2;
        cv::circle(img, ud.controls[0][i], 2, cv::Scalar(255, 0, 0), 3);
    }
//    cv::drawContours(img, ud.controls, 0, cv::Scalar(0,255,0), 2);

    int n_u_add = 8;
    int n_u = ud.controls[0].size() + n_u_add;
    ud.controls[0].resize(n_u);
    for (int i = 0; i < n_u_add; i++)
    {
        ud.controls[0][n_u - i - 1] = ud.controls[0][n_u_add - 1 - i];
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
    for(int i = ud.knots[4] * (ud.quantize-1); i < ud.knots[ud.knots.size() - 4] * (ud.quantize-1); i++)
    {
        ud.pts[i] = ComputePoint(ud.Nx, ud.controls[0], i);
        if (ud.pts[i].x > 0.f)
            cv::circle(img, ud.pts[i], 1, cv::Scalar(255), 2);
    }
    ud.ratio = std::vector<double>(ud.quantize, 1.0);
    for (int i = 0; i < ud.quantize; i++)
    {
        float t = 3.14f * i / ud.quantize * 2.f + ud.ellipse.angle / 180.f * 3.14;
        cv::Point2f pt(ud.ellipse.size.width/2.f * std::cos(t),
                       ud.ellipse.size.height/2.f * std::sin(t));
        ud.ratio[i] = cv::norm(ud.pts[i] - ud.ellipse.center) / cv::norm(pt);
    }

    cv::imshow("show", img);

    cv::setMouseCallback("show", mouse_callback, &ud);
    cv::waitKey();
    return 0;
}
