#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;


Mat getMask(Mat img){

    cv::Mat gray, blur, mask;

    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0, 0);
    cv::threshold(blur, mask, 0, 255, cv::THRESH_OTSU);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    return mask;
}

Rect getMaxContourCoordinates(Mat img){

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> maxContour;
    cv::Rect coordinates;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() > 0){
        maxContour = *std::max_element(contours.begin(), contours.end(),
                        [](std::vector<cv::Point> const &lhs, std::vector<cv::Point> const &rhs)
                        { return contourArea(lhs, false) < contourArea(rhs, false) ;}
                        );
        coordinates = boundingRect(maxContour);
    }
    return coordinates;
}

Mat centerCrop(Mat img, int w, int h){
    
    int x, y;

    x = (int)( img.size().width/2 - (w/2) );
    y = (int)( img.size().height/2 - (h/2) );
    Rect roi(x, y, w, h);
    
    return img(roi);
}


int main(int argc, char **argv){

    string path, dataset, label, filename;
    Mat img, imgResize, mask, imgROI;
    Rect coordinates;

    path = "";
    dataset = "";
    label = "";
    filename = "";

    img = imread(path + dataset + label + filename);
    // resize(img, imgResize, cv::Size(), 0.5, 0.5);
    cout << "Height: " << img.size().height << endl;
    cout << "Width: " << img.size().width << endl;

    mask = getMask(img);
    coordinates = getMaxContourCoordinates(mask);
    imgROI = img(coordinates);

    imshow("ROI", imgROI);
    waitKey(0);

    return 0;
}
