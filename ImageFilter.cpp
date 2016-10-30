#include "ImageFilter.h"

ImageFilter::ImageFilter(){
	DELAY_CAPTION = 1500;
	DELAY_BLUR = 100;
	MAX_KERNEL_LENGTH = 8;

	EDGETHRESH = 1;
	LOWTHRESHOLD;
	MAX_LOWTHRESHOLD = 100;
	RATIO = 3;

	//YIQValues[][3] = {{0.299,0.587,0.114},{0.596,-0.274,-0.322},{0.211,-0.253,0.312}};
}

ImageFilter::ImageFilter(int iLowHIn, int iHighHIn, int iLowSIn, int iHighSIn, int iLowVIn, int iHighVIn){
	DELAY_CAPTION = 1500;
	DELAY_BLUR = 100;
	MAX_KERNEL_LENGTH = 8;

	EDGETHRESH = 1;
	LOWTHRESHOLD;
	MAX_LOWTHRESHOLD = 100;
	RATIO = 3;


	iLowH = iLowHIn;
	iHighH = iHighHIn;
	iLowS = iLowSIn;
	iHighS = iHighSIn;
	iLowV =iLowVIn ;
	iHighV = iHighVIn;
}

Mat ImageFilter::YIQColorFilter(Mat original){
	Mat originalImage = original.clone();
	int RGB[3];
	double YIQRes[3];;
	RGB[2] = originalImage.at<cv::Vec3b>(1,1)[0];
	RGB[1] = originalImage.at<cv::Vec3b>(1,1)[1];
	RGB[0] = originalImage.at<cv::Vec3b>(1,1)[2];

	for(int i = 0 ; i < originalImage.rows; i++){
		YIQRes[0] = RGB[0]*0.299 + RGB[1]* 0.587 + RGB[2]* 0.114;
		YIQRes[1] = RGB[0]*0.596 + RGB[1]*-0.274 + RGB[2]*-0.322;
		YIQRes[2] = RGB[0]*0.211 + RGB[1]*-0.523 + RGB[2]* 0.312;

		originalImage.at<cv::Vec3b>(1,1)[0] = YIQRes[0];
		originalImage.at<cv::Vec3b>(1,1)[1] = YIQRes[1];
		originalImage.at<cv::Vec3b>(1,1)[2] = YIQRes[2];
	}
	//imshow("YIQ Image", originalImage);      // Show the thresholded image.
	return originalImage;
}

Mat ImageFilter::HSVcolorFilter(Mat originalImage) {
    Mat src_gray;
    Mat imgHSV;
    Mat imgThresholded;

    cvtColor(originalImage, imgHSV, COLOR_BGR2HSV);       // Convert the captured frame from BGR to HSV.
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), 
    Scalar(iHighH, iHighS, iHighV), imgThresholded); // Threshold the image.

    // Morphological opening (remove small objects from the foreground).
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    // Morphological closing (fill small holes in the foreground).
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    // Get the moments.
    Moments oMoments = moments(imgThresholded);

    // Receive the centroid area.
    
    // Cloned the modified image to calculate the points.
    src_gray = imgThresholded.clone();
    blur(src_gray, src_gray, Size(3,3));
    return src_gray;
    //imshow("HSV Filter", src_gray);
}

Mat ImageFilter::LPFMeanBlur(Mat originalImage){
	Mat meanImg = originalImage.clone();
	for(int i = 1; i < MAX_KERNEL_LENGTH; i = i+2)
		blur(originalImage,meanImg,Size(i,i), Point(-1,-1));

	//imshow("LPFMean Image", meanImg);      // Show the thresholded image.
	return meanImg;

}

Mat ImageFilter::LPFGaussianBlur(Mat originalImage){
	Mat gaussianImg = originalImage.clone();

	for(int i = 1; i < MAX_KERNEL_LENGTH; i = i+2)
		GaussianBlur(originalImage,gaussianImg,Size(i,i),3,3);

	//imshow("GaussianBlur Image", gaussianImg);      // Show the thresholded image.
	return gaussianImg;

}

//Really Slow Check what can be done
Mat ImageFilter::LPFMedianBlur(Mat originalImage){
	Mat medianImg = originalImage.clone();

	for(int i = 1; i < MAX_KERNEL_LENGTH; i = i+2)
		medianBlur(originalImage,medianImg, i );

	//imshow("Median Filter Image", medianImg);      // Show the thresholded image.
	return medianImg;
}

Mat ImageFilter::bilateralFilterBlur(Mat originalImage){
	Mat bilateralImg = originalImage.clone();

	for(int i = 1; i < MAX_KERNEL_LENGTH; i = i+2)
		bilateralFilter(originalImage,bilateralImg, i , i*2 ,i/2 );

	//imshow("Bilateral Filter Image", bilateralImg);      // Show the thresholded image.	
	return bilateralImg;
}

Mat ImageFilter::HPFLaplacian(Mat originalImage){
	int SCALE = 3;
	int DELTA = 75;
	int DDEPTH = CV_16S;
	int KENREL_SIZE = 3;
	Mat laplacianImg = originalImage.clone();
	Mat abs_dst;

	Laplacian(originalImage,laplacianImg,DDEPTH, KENREL_SIZE, DELTA, BORDER_DEFAULT );
	convertScaleAbs(laplacianImg, abs_dst);
	//imshow("Laplacian abs_dst Filter Image", abs_dst);      // Show the thresholded image.	
	//imshow("Laplacian laplacianImg Image", laplacianImg);      // Show the thresholded image.	
	return laplacianImg;
}

Mat ImageFilter::HPFEdgeDetector(Mat originalImage){
	int KENREL_SIZE = 3;
	Mat src_gray, detected_edges;
	LOWTHRESHOLD = 75;
	/// Convert the image to grayscale
	Mat edgeDetectorImg = originalImage.clone();
	cvtColor( originalImage, src_gray, CV_BGR2GRAY );

	// Create a Trackbar for user to enter threshold
  	//createTrackbar( "Min Threshold:", "Edge Detector Image" , &LOWTHRESHOLD, MAX_LOWTHRESHOLD );

	/// Reduce noise with a kernel 3x3
  	blur( src_gray, detected_edges, Size(3,3) );

  	/// Canny detector
  	Canny( detected_edges, detected_edges, LOWTHRESHOLD, LOWTHRESHOLD*RATIO, KENREL_SIZE );

  	/// Using Canny's output as a mask, we display our result
  	edgeDetectorImg = Scalar::all(0);

  	originalImage.copyTo( edgeDetectorImg, detected_edges);
  	//imshow( "Edge Detector Image", edgeDetectorImg );
  	//imshow( "detected_edges Image", detected_edges );
  	return detected_edges;

}

Mat ImageFilter::HPFEnhancement(Mat originalImage){
	Mat enhancementImg = originalImage.clone();
	/*
	Mat lapl = originalImage.clone();
	lapl = HPFLaplacian(enhancementImg);
	substract(lapl,enhancementImg,enhancementImg);
	*/
	GaussianBlur(enhancementImg, enhancementImg, cv::Size(0, 0), 0.1);
	addWeighted(enhancementImg, 1.5, enhancementImg, -0.5, 0, enhancementImg);

	return enhancementImg;
}

// Values threshold_type 
  /* 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
Mat ImageFilter::MBinDilate(Mat originalImage,int dilation_type){
	int dilation_elem = 0;
	int dilation_size = 3;
	int threshold_value = 200;
	int threshold_type = 0;
	
	Mat dilationImg = originalImage.clone();
	//Image Binarization
	Mat threasholdImg ;
	cvtColor( originalImage, threasholdImg, CV_BGR2GRAY );
	//threasholdImg = originalImage.clone();
	imshow("B/W", threasholdImg);
	threshold( threasholdImg, threasholdImg, threshold_value, 255 ,threshold_type );
	//imshow("Threahold Img", threasholdImg);
	
	//int dilation_type;
  	if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  	else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  	else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
	
	Mat element = getStructuringElement(dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ));
	
	dilate(threasholdImg,dilationImg,element);
	//imshow("Dilation Demo", dilationImg);
	return dilationImg;

}

Mat ImageFilter::MBinErode(Mat originalImage, int erosion_type){
	int erosion_elem = 0;
	int erosion_size = 3;
	int threshold_value = 200;
	int threshold_type = 0;
	Mat erosionImg = originalImage.clone();
	
	Mat threasholdImg ;
	cvtColor( originalImage, threasholdImg, CV_BGR2GRAY );
	//imshow("B/W", threasholdImg);
	threshold( threasholdImg, threasholdImg, threshold_value, 255 ,threshold_type );
	
	if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  	else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  	else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ));

	//int erosion_type;
	
    
	erode(threasholdImg,erosionImg,element);
	//imshow("Erotion Demo", erosionImg);
	return erosionImg;
}

Mat ImageFilter::MBinOpening(Mat originalImage){
	Mat openingImg = originalImage.clone();
	Mat erosionImg,dilationImg;
	Mat threasholdImg ;
	
	int threshold_value = 127;
	int threshold_type = 0;
	int erosion_elem = 0;
	int erosion_size = 2;
	int erosion_type = 0;
	int dilation_type = 0;
	int dilation_elem = 0;
	int dilation_size = 2;
	
	cvtColor( originalImage, threasholdImg, CV_BGR2GRAY );
	//imshow("B/W", threasholdImg);
	threshold( threasholdImg, threasholdImg, threshold_value, 255 ,threshold_type );
	Mat element = getStructuringElement(erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ));

	//int erosion_type;
	if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  	else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  	else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  	erode(threasholdImg,erosionImg,element);
  	dilate(erosionImg,dilationImg,element);
  	//imshow("Opening Image", dilationImg);
  	return dilationImg;
}

Mat ImageFilter::MBinClosing(Mat originalImage){
	Mat closingImg = originalImage.clone();
	Mat erosionImg,dilationImg;
	Mat threasholdImg ;

	int threshold_value = 127;
	int threshold_type = 0;
	int erosion_elem = 0;
	int erosion_size = 2;
	int erosion_type = 0;
	int dilation_type = 0;
	int dilation_elem = 0;
	int dilation_size = 2;
	
	cvtColor( originalImage, threasholdImg, CV_BGR2GRAY );
	//imshow("B/W", threasholdImg);
	threshold( threasholdImg, threasholdImg, threshold_value, 255 ,threshold_type );
	Mat element = getStructuringElement(erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ));

	//int erosion_type;
	if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  	else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  	else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  	dilate(threasholdImg,dilationImg,element);
  	erode(dilationImg,erosionImg,element);
  	//imshow("Closing Image", erosionImg);
  	return erosionImg;
}

Mat ImageFilter::grayscaleFilter(Mat originalImage){
	Mat src_gray = originalImage.clone();
	cvtColor( originalImage, src_gray, CV_BGR2GRAY );
	//imshow("src_gray",src_gray);
	return src_gray;
}

Mat ImageFilter::sobelDerivative(Mat originalImage){
	Mat src_gray = originalImage.clone();
	Mat grad;
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

  	GaussianBlur( src_gray, src_gray, Size(3,3), 0, 0, BORDER_DEFAULT );

  	/// Convert it to gray
  	cvtColor( src_gray, src_gray, CV_BGR2GRAY );
  	/// Create window
  	//namedWindow( "Sobel Filter", CV_WINDOW_AUTOSIZE );

  	/// Generate grad_x and grad_y
  	Mat grad_x, grad_y;
  	Mat abs_grad_x, abs_grad_y;

  	/// Gradient X
  	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_x, abs_grad_x );

  	/// Gradient Y
  	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_y, abs_grad_y );

  	/// Total Gradient (approximate)
  	addWeighted( src_gray, 0, abs_grad_y, 0.5, 0, grad );

  	//imshow( "Sobel Filter ", grad );
  	return grad;
}

Mat ImageFilter::blobColoring(Mat originalImage){

	// Setup SimpleBlobDetector parameters.
	Mat im = originalImage.clone();
	//im = MBinErode(originalImage,0);
	//imshow("Erode", im);
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 1500;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// Storage for blobs
	vector<KeyPoint> keypoints;
#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

	// Set up detector with params
	SimpleBlobDetector detector(params);

	// Detect blobs
	detector.detect( im, keypoints);
#else 

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

	// Detect blobs
	detector->detect( im, keypoints);
#endif 

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;
	drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

	// Show blobs
	//imshow("keypoints", im_with_keypoints );
	return im_with_keypoints;
}

Mat ImageFilter::FindContours(Mat originalImage){
  	
  	RNG rng(12345);
	
	vector<vector<Point> > contours;
  	vector<Vec4i> hierarchy;
	Mat canny_output;
  	Mat src_gray = originalImage.clone();
  	int thresh = 100;

  	//cvtColor( originalImage, src_gray, CV_BGR2GRAY );
	blur( src_gray, src_gray, Size(3,3) );

  	/// Detect edges using canny
  	Canny( src_gray, canny_output, thresh, thresh*2, 3 );
  	/// Find contours
  	findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  	/// Draw contours
  	Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  	for( int i = 0; i< contours.size(); i++ ){
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }

  	/// Show in a window
  	//namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  	//imshow( "Contours", drawing );
  	return drawing;

}

Mat ImageFilter::MorphClosing(Mat original){
    // Morphological closing (fill small holes in the foreground).
    dilate( original, original, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(original, original, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    return original;
}

Mat ImageFilter::MorphOpening(Mat original){
	// Morphological opening (remove small objects from the foreground).
    erode(original, original, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( original, original, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    return original;
}

Mat ImageFilter::waterShed(Mat originalImage){
    
    imshow("Input Image WaterShed", originalImage);
    Mat src = originalImage.clone();

    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("Imagen Binaria", bw);
	
	// Morphological opening (remove small objects from the foreground).
    bw = MorphOpening(bw);
    // Morphological closing (fill small holes in the foreground).
    bw = MorphClosing(bw);

	// Morphological opening (remove small objects from the foreground).
    bw = MorphOpening(bw);
    // Morphological closing (fill small holes in the foreground).
    bw = MorphClosing(bw);


    Mat dist = originalImage.clone();
    distanceTransform(bw, dist, CV_DIST_L2, 3);
    normalize(dist, dist, 0, 1., NORM_MINMAX);
	imshow("Dist", dist);

    // Aplicamos otro Threshold
    // Esto marcara los objetos de atras
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
   	//imshow("Dist,Thresh", dist);

   	//Mat dist = binOriginalImage.clone();
    Mat erodeElement = getStructuringElement(MORPH_RECT,Size(3,3));
    Mat dilateElement = getStructuringElement(MORPH_RECT,Size(8,8));
    //erode(dist,dist,erodeElement);
    //erode(dist,dist,erodeElement);
    //dilate(dist,dist,dilateElement);
    //dilate(dist,dist,dilateElement);

    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    vector< vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, i, Scalar::all(i+1), -1);
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    //imshow("Marcadores", markers*10000);

    // Aplicamos el efecto de gota de aceite
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
    //imshow("Markers_v2", mark);

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++){
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++){
        for (int j = 0; j < markers.cols; j++){
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    //imshow("Resultado", dst);
    return dst;
}


Mat ImageFilter::FindFigures(Mat originalImage){


}