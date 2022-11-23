#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;

void plotImage(Mat image) {
    std::string display_name = "Image";
    int window_dimension = 700;

    namedWindow(display_name, WINDOW_NORMAL);
    resizeWindow(display_name, window_dimension, window_dimension);
    imshow(display_name, image);
    waitKey(0);
}


class ImageProcessing {

    public:
        Mat image;

        ImageProcessing(Mat image) {
            image = image;
        }
};

int main(int argc, char** argv) {

    std::string path = "D:/DOMI/University/Thesis/Coding/Dataset/TestSet/Novara_good/3Fpt2.png";
    Mat image, gray, mask, final_image;

    image = imread(path);

    if (!image.data) {
        std::cout << "No image data \n";
        return -1;
    }
    else {

        cvtColor(image, gray, COLOR_BGR2GRAY);
        threshold(gray, mask, 0, 255, THRESH_BINARY + THRESH_OTSU);
        // morphologyEx(mask, final_image, MORPH_CLOSE, getStructuringElement(5));
        plotImage(image);
    }

    return 0;
}