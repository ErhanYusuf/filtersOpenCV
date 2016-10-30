//Proyecto 1 Vision para robots

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class ImageFilter{
public:
	ImageFilter();
	ImageFilter(int iLowHIn, int iHighHIn, int iLowSIn, int iHighSIn, int iLowVIn, int iHighVIn);
	Mat YIQColorFilter(Mat originalImage, int init, int end);
	Mat HSVcolorFilter(Mat originalImage);
	//Mat HSVFilterCalibration();
	Mat YIQColorFilter(Mat originalImage);
	Mat bilateralFilterBlur(Mat originalImage);
	Mat LPFMeanBlur(Mat originalImage);
	Mat LPFGaussianBlur(Mat originalImage);
	Mat LPFMedianBlur(Mat originalImage);
	Mat HPFLaplacian(Mat originalImage);
	Mat HPFEdgeDetector(Mat originalImage);
	Mat HPFEnhancement(Mat originalImage);
	Mat MBinDilate(Mat originalImage,int dilation_type);
	Mat MBinErode(Mat originalImage, int erosion_type);
	Mat MBinOpening(Mat originalImage);
	Mat MBinClosing(Mat originalImage);
	Mat grayscaleFilter(Mat originalImage);
	Mat sobelDerivative(Mat originalImage);
	Mat blobColoring(Mat originalImage);

	Mat waterShed(Mat originalImage);
	Mat MorphClosing(Mat original);
	Mat MorphOpening(Mat original);

	Mat FindContours(Mat originalImage);
	Mat FindFigures(Mat originalImage);
	//void WaterShed2(Mat originalImage);
	//void segmentation_and_characterization(Mat& image2segment,Mat& segmentedImage,std::vector<region_info>& regions_table);


private:
	int DELAY_CAPTION;
	int DELAY_BLUR;
	int MAX_KERNEL_LENGTH;

	int EDGETHRESH;
	int LOWTHRESHOLD;
	int MAX_LOWTHRESHOLD;
	int RATIO;
	int YIQValues[3][3];

	int iLowH;
	int iHighH;
	int iLowS;
	int iHighS;
	int iLowV;
	int iHighV;
};