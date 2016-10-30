//Proyecto 1 Vision para robots

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageFilter.h"

#include <string>

using namespace std;
using namespace cv;

void filterSelection(string , Mat);
void fillHSVParams(struct HSVParams *p);
void HSVCalibration(struct HSVParams *p);

int iLowH = 40;
int iHighH = 75;
int iLowS = 44;
int iHighS = 255;
int iLowV = 104;
int iHighV = 255;

/** Filters List
	"hsv" or "HSV"
	"yiq" or "YIQ"
	"grayscaleFilter"
	"bilateralFilterBlur"
	"MeanBlur"
	"GaussianBlur"
	"MedianBlur"
	"Laplacian"
	"EdgeDetector"
	"Enhancement"
	"Dilate"
	"Erode"
	"Opening"
	"Closing"
	"sobelDerivative"
**/

void HSVcolorFilter(Mat);

struct HSVParams{
	int iLowH;
	int iHighH;
	int iLowS;
	int iHighS;
	int iLowV;
	int iHighV;
};


void fillHSVParams(struct HSVParams *p){
	p->iLowH = 160;
	p->iHighH = 179;
	p->iLowS = 100;
	p->iHighS = 255;
	p->iLowV = 100;
	p->iHighV = 255;
}

void printHSVParams(struct HSVParams *p){
	cout << p->iLowH << endl;
	cout << p->iHighH << endl;
	cout << p->iLowS << endl;
	cout << p->iHighS << endl;
	cout << p->iLowV << endl;
	cout << p->iHighV << endl;
}

int main(int argc, char **argv){
	char key = 'c';
	string filter;
	VideoCapture cap(0);
	HSVParams params;
	//filter = "HSV" ;
	ImageFilter filtro;
	if(!cap.open(0)){
		return 0;
	}

	if(filter == "HSV" || filter == "hsv"){
		HSVCalibration(&params);
		printHSVParams(&params);
	}
	
	Mat image = imread("green.jpg",1);
	//Mat filteredImg;
	//image = imread(argv[1],1);
	//filteredImg = filtro.HPFEdgeDetector(image);
	//filteredImg = filtro.HPFLaplacian(filteredImg);
	
	//imwrite("CowsFilteredImage.jpg", filteredImg);
	/*
	while(key != 27){
		HSVcolorFilter(image);
		key = waitKey(10);
	}
	*/
	
	while(key != 27){
		Mat frame = image.clone();
		//resize(image,frame,cvSize(640,480));
		//Mat frame;
		Mat img;
		//Mat img = frame.clone();
		//cap >> frame;
		//Mat img = frame.clone();

		cvtColor( frame, img, CV_BGR2GRAY );

		ImageFilter filters;
		Mat filteredImg;
		
		Mat LPMedian, LPGaussian, LPAvg;
		Mat HPLaplacian, HPEdge;

		Mat sobDeriv;
		Mat Morph, Dilation, Erotion, Op, Clos;

		// Add it if while
		//if(frame.empty())
		//	break;


		LPMedian = img.clone();
		LPGaussian = img.clone();
		LPAvg = img.clone();
		HPLaplacian = img.clone();
		HPEdge = frame.clone();
		sobDeriv = img.clone();
		Morph = frame.clone();
		Dilation = frame.clone();
		Erotion = frame.clone();
		Op = frame.clone();
		Clos = frame.clone();
		Mat waterShed = frame.clone();

		//LPMedian = filters.LPFMedianBlur(LPMedian);
		
		//imshow("LPMedian ", LPMedian);

		//LPGaussian = filters.LPFGaussianBlur(LPGaussian);

		//imshow("LPGaussian ", LPGaussian);
		
		//LPAvg = filters.LPFMeanBlur(LPAvg);

		//imshow("LPMean", LPAvg);
		
		//HPLaplacian = filters.HPFLaplacian(HPLaplacian);

		//imshow("HPLaplacian", HPLaplacian);

		//HPEdge = filters.HPFEdgeDetector(HPEdge);

		//imshow("HPEdge", HPEdge);

		//sobDeriv = filters.sobelDerivative(sobDeriv);

		//imshow("YDeriv",sobDeriv);
		
		////////////////////////////////////////////////////
		/////////REVIAR A PARTIR DE AQUI ///////////////////
		////////////////////////////////////////////////////
		
		//sobDeriv = filters.HPFEnhancement(sobDeriv);

		//imshow("Enhancement", sobDeriv);
		
		//Dilation = filters.MBinDilate(Dilation,0);

		//imshow("Dilation", Dilation);

		//Erotion = filters.MBinErode(Erotion,0);

		//imshow("Erotion",Erotion);

		//Op = filters.MBinOpening(Op);

		//imshow("Opening",Op);
				
		//Clos = filters.MBinClosing(Clos);

		//imshow("Closing",Clos);
		
		//
		//
		//
		//
		//Op = filters.blobColoring(img);
		//imshow("BColoring", Op);
		//Clos = filters.FindContours(frame);
		//imshow("Contours", Clos);

		//waterShed = filters.waterShed(waterShed);
		HSVcolorFilter(waterShed);
		//imshow("waterShed", waterShed);
		imshow("Image ", frame);

		//imshow("Filtered", filteredImg);
		//imwrite("image.jpg",filteredImg);
	
		key = waitKey(1);
	}
	
	return 0;
}

void HSVcolorFilter(Mat originalImage) {
	Mat orig = originalImage.clone();
    Mat src_gray;
    Mat imgHSV;
    Mat imgThresholded;
	ImageFilter filtro;
	Mat BGRHSVimg;

    cvtColor(orig, imgHSV, COLOR_BGR2HSV);       // Convert the captured frame from BGR to HSV.
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), 
    Scalar(iHighH, iHighS, iHighV), imgThresholded); // Threshold the image.
    
    Mat other_filter;
    bitwise_and(imgHSV,imgHSV,other_filter, imgThresholded = imgThresholded);
    cvtColor(other_filter, BGRHSVimg, CV_HSV2BGR);
    //imshow("BGR", BGRHSVimg);

    // Morphological opening (remove small objects from the foreground).
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    // Morphological closing (fill small holes in the foreground).
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    // Get the moments.
    Moments oMoments = moments(imgThresholded);

    //imshow("BGR", BGRHSVimg);
    Mat myW = filtro.waterShed(BGRHSVimg);

    //Mat segmentedImage = Mat::zeros(imageBinarizada.rows,imageBinarizada.cols,CV_8UC3);
	//segmentation_and_characterization(imageBinarizada,segmentedImage,regions_table);

    // Receive the centroid area.
    
    // Cloned the modified image to calculate the points.
    src_gray = imgThresholded.clone();
    blur(src_gray, src_gray, Size(3,3));

    namedWindow( "HSV_Filter", WINDOW_NORMAL );
    cvCreateTrackbar("LowH" , "HSV_Filter", &iLowH, 179);   // Hue (0 - 179).
    cvCreateTrackbar("HighH", "HSV_Filter", &iHighH, 179);
    cvCreateTrackbar("LowS" , "HSV_Filter", &iLowS, 255);   // Saturation (0 - 255).
    cvCreateTrackbar("HighS", "HSV_Filter", &iHighS, 255);
    cvCreateTrackbar("LowV" , "HSV_Filter", &iLowV, 255);   // Value (0 - 255).
    cvCreateTrackbar("HighV", "HSV_Filter", &iHighV, 255);
 
    //return src_gray;
    imshow("WaterS", myW);
    imshow("HSV Filter", src_gray);
}

void HSVCalibration(struct HSVParams *p){
	char key = 'c';
	VideoCapture cap2;
	bool loop = true; 
 	while(key == 27){
		Mat originalImage;
		ImageFilter filters;
		Mat filteredImg;
		cap2 >> originalImage;
		if(originalImage.empty())
			break;
		
	    Mat src_gray;
	    Mat imgHSV;
	    Mat imgThresholded;

	    cvtColor(originalImage, imgHSV, COLOR_BGR2HSV);       // Convert the captured frame from BGR to HSV.
	    inRange(imgHSV, Scalar(p->iLowH, p->iLowS, p->iLowV), 
	    Scalar(p->iHighH, p->iHighS, p->iHighV), imgThresholded); // Threshold the image.

	    // Morphological opening (remove small objects from the foreground).
	    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	    // Morphological closing (fill small holes in the foreground).
	    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	    // Get the moments.
	    Moments oMoments = moments(imgThresholded);

	    // Receive the centroid area.
	    //double dArea = oMoments.m00;
	    //areaColorDetection = dArea / 100000;

	    // Cloned the modified image to calculate the points.
	    src_gray = imgThresholded.clone();

	    // Blur to soften the image points.
	    blur(src_gray, src_gray, Size(3,3));
	    namedWindow( "HSV_Filter", WINDOW_NORMAL );
	    cvCreateTrackbar("LowH" , "HSV_Filter", &p->iLowH, 10);   // Hue (0 - 179).
	    cvCreateTrackbar("HighH", "HSV_Filter", &p->iHighH, 179);
	    cvCreateTrackbar("LowS" , "HSV_Filter", &p->iLowS, 10);   // Saturation (0 - 255).
	    cvCreateTrackbar("HighS", "HSV_Filter", &p->iHighS, 255);
	    cvCreateTrackbar("LowV" , "HSV_Filter", &p->iLowV, 255);   // Value (0 - 255).
	    cvCreateTrackbar("HighV", "HSV_Filter", &p->iHighV, 255);

	    //imshow("HSV_Filter", src_gray);
	    key = waitKey(0);
	}
}

void filterSelection(string filter, Mat frame){
	ImageFilter filters;
	Mat filteredImg;

	if(filter == "grayscaleFilter"){
		Mat img = filters.grayscaleFilter(frame);
		imshow("src_gray" , img);
	}
	else if(filter == "YIQ" || filter == "yiq"){
		filteredImg = filters.YIQColorFilter(frame);
		imshow("YIQ Filter", filteredImg);

	}
	else if(filter == "HSV" || filter == "hsv"){
		filteredImg = filters.HSVcolorFilter(frame);
		imshow("HSV Filter", filteredImg);
	}
	else if(filter == "bilateralFilterBlur"){		
	 	filteredImg = filters.bilateralFilterBlur(frame);
	 	imshow("bilateralFilterBlur Filter", filteredImg);
	}
	else if(filter == "MeanBlur"){
		filteredImg =  filters.LPFMeanBlur(frame);
		imshow("MeanBlur ", filteredImg);
	}
	else if(filter == "GaussianBlur"){		
		filteredImg = filters.LPFGaussianBlur(frame);
		imshow("GaussianBlur Filter", filteredImg);
	}
	else if(filter == "MedianBlur"){
		filteredImg = filters.LPFMedianBlur(frame);
		imshow("MedianBlur Filter", filteredImg);
	}
	else if(filter == "Laplacian"){
		filteredImg = filters.HPFLaplacian(frame);
		imshow("Laplacian Filter", filteredImg);
	}
	else if(filter == "EdgeDetector"){
		filteredImg = filters.HPFEdgeDetector(frame);
		imshow("Edge Filter", filteredImg);		
	}
	else if(filter == "Enhancement"){

	}
	else if(filter == "Dilate"){
		filteredImg = filters.MBinDilate(frame,0);
		imshow("Dilate Filter", filteredImg);
	}
	else if(filter == "Erode"){
		filteredImg = filters.MBinErode(frame,0);
		imshow("Erode Filter", filteredImg);
	}
	else if(filter == "Opening"){
		filteredImg = filters.MBinOpening(frame);
		imshow("Opening Filter", filteredImg);
	}
	else if(filter == "Closing"){
		filteredImg = filters.MBinClosing(frame);
		imshow("Closing Filter", filteredImg);
	}
	else if(filter == "sobelDerivative"){
		filteredImg = filters.sobelDerivative(frame);
		imshow("sobelDerivative Filter", filteredImg);
	}
}
