#include <iostream>
#include <stdio.h>
using namespace std;

int HIGHRES = 1024;
int BORDER_THICK = 5;
int DARK_THRESH = 15;
int THRESH_RATIO = 1000;
int COMPRESSION = 6;

class ImageProcessing
{

}

float
thresholding(Mat img, int denoiseKernel, int areaCloseKernel)
{
    grayImg = cvtColor(img, );
    blurImg = GaussianBlur(gray, (denoiseKernel, denoiseKernel), 0);
    thImg = thresholding(blurImg, 0, 255, );
    areaCloseImg = morphologyEx(thImg, );
    return areaCloseImg;
}

int main(int argc, int *argv)
{
    cout << "hello world";
    return 0;
}