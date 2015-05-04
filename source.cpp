#include "opencv2/opencv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int nrows = 480, ncolumns = 640;
enum { STEREO_BM = 0, STEREO_SGBM = 1, BM_STATE = 2 };
int typeofbm = BM_STATE;
Mat leftImagerBM, rightImagerBM;
CvMat* img1r = cvCreateMat(nrows, ncolumns, CV_8U);
CvMat* img2r = cvCreateMat(nrows, ncolumns, CV_8U);
Mat disparityBM(leftImagerBM.rows, leftImagerBM.cols, CV_16S);
Mat disparityBMn(leftImagerBM.rows, leftImagerBM.cols, CV_8U);

int minDisparity = 64;
int numberOfDisparities = 128;
int preFilterSize = 41;
int preFilterCap = 31;
int SADWindowSize = 41;
int textureThreshold = 10;
int uniquenessRatio = 15;

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

void ComputeDisparityMap(int typeofbm)
{
	//cout << "BM Computation Started" << endl;
	if (typeofbm == STEREO_BM)
	{
		//cout << "Stereo BM" << endl;
		/*STEREO BM | C++*/
		StereoBM stereoBM(CV_STEREO_BM_BASIC);
		stereoBM.state->minDisparity = minDisparity*-1;
		stereoBM.state->preFilterSize = preFilterSize;
		stereoBM.state->preFilterCap = preFilterCap;
		stereoBM.state->SADWindowSize = SADWindowSize;
		stereoBM.state->numberOfDisparities = numberOfDisparities;
		stereoBM.state->textureThreshold = textureThreshold;
		stereoBM.state->uniquenessRatio = uniquenessRatio;
		stereoBM(leftImagerBM, rightImagerBM, disparityBM);
		normalize(disparityBM, disparityBMn, 0, 256, CV_MINMAX);
		imshow("Disparity Map Normalized", disparityBMn);
		imshow("Disparity Map", disparityBM);
	}
	else if (typeofbm == STEREO_SGBM)
	{
		//cout << "Stereo SGBM" << endl;
		/*STEREO SGBM | C++*/
		StereoSGBM stereoSGBM(minDisparity*-1, numberOfDisparities, SADWindowSize, 8 * SADWindowSize * SADWindowSize,
			32 * SADWindowSize * SADWindowSize, 0, preFilterCap, uniquenessRatio, 0, 0);
		stereoSGBM(leftImagerBM, rightImagerBM, disparityBM);
		normalize(disparityBM, disparityBMn, 0, 256, CV_MINMAX);
		imshow("Disparity Map Normalized", disparityBMn);
		imshow("Disparity Map", disparityBM);
	}
	else
	{
		//cout << "BM State" << endl;
		/*BM STATE | C*/ ///*
		CvStereoBMState *BMState = cvCreateStereoBMState();
		BMState->preFilterSize = preFilterSize;
		BMState->preFilterCap = preFilterCap;
		BMState->SADWindowSize = SADWindowSize;
		BMState->minDisparity = minDisparity*-1;
		BMState->numberOfDisparities = numberOfDisparities;
		BMState->textureThreshold = textureThreshold;
		BMState->uniquenessRatio = uniquenessRatio;
		
		CvMat* disp = cvCreateMat(nrows, ncolumns, CV_16S);
		CvMat* vdisp = cvCreateMat(nrows, ncolumns, CV_8U);
		cvFindStereoCorrespondenceBM(img1r, img2r, disp, BMState);
		cvNormalize(disp, vdisp, 0, 256, CV_MINMAX);
		cvShowImage("Disparity Map", disp);
		cvShowImage("Disparity Map Normalized", vdisp);
		//Mat mat3d(leftImagerBM.rows, leftImagerBM.cols, CV_32FC3);
		//Mat Qmat = (CvMat*)cvLoad("images//Q.xml");
		//reprojectImageTo3D((Mat)vdisp, mat3d, Qmat);
		//saveXYZ("xyz_point.txt", mat3d);
		cvReleaseStereoBMState(&BMState);
	}
	cout << "BM Computed" << endl;
}

static void min_disparities(int, void*)
{
	if (minDisparity % 16 != 0) //Must be divisible by 16
	{
		minDisparity += (16 - minDisparity % 16);
		setTrackbarPos("min_disparities", "Track Bars", minDisparity);
		return;
	}
	cout << "Min Disparities = " << minDisparity*-1 << endl;
	ComputeDisparityMap(typeofbm);
}
static void num_of_disparities(int, void*)
{
	if (numberOfDisparities < 16)
	{
		numberOfDisparities = 16;
		setTrackbarPos("num_of_disparities", "Track Bars", numberOfDisparities);
		return;
	}
	if (numberOfDisparities % 16 != 0) //Must be divisible by 16
	{
		numberOfDisparities += (16 - numberOfDisparities % 16);
		setTrackbarPos("num_of_disparities", "Track Bars", numberOfDisparities);
		return;
	}
	cout << "Num of Disparities = " << numberOfDisparities << endl;
	ComputeDisparityMap(typeofbm);
}
static void pre_filter_size(int, void*)
{
	if (preFilterSize < 5)
	{
		preFilterSize = 5;
		setTrackbarPos("pre_filter_size", "Track Bars", preFilterSize);
		return;
	}
	if (preFilterSize > 255)
	{
		preFilterSize = 255;
		setTrackbarPos("pre_filter_size", "Track Bars", preFilterSize);
		return;
	}
	if (preFilterSize % 2 == 0) //Must be an odd number
	{
		preFilterSize += 1;
		setTrackbarPos("pre_filter_size", "Track Bars", preFilterSize);
		return;
	}
	cout << "Pre Filter Size = " << preFilterSize << endl;
	ComputeDisparityMap(typeofbm);
}
static void pre_filter_cap(int, void*)
{
	if (preFilterCap < 1)
	{
		preFilterCap = 1;
		setTrackbarPos("pre_filter_cap", "Track Bars", preFilterCap);
		return;
	}
	if (preFilterCap > 63)
	{
		preFilterCap = 63;
		setTrackbarPos("pre_filter_cap", "Track Bars", preFilterCap);
		return;
	}
	cout << "Pre Filter Cap = " << preFilterCap << endl;
	ComputeDisparityMap(typeofbm);
}
static void SAD_window_size(int, void*)
{
	if (SADWindowSize < 5) //Should be greater than 5
	{
		SADWindowSize = 5;
		setTrackbarPos("SAD_window_size", "Track Bars", SADWindowSize);
		return;
	}
	if (SADWindowSize % 2 == 0) //Must be an odd number
	{
		SADWindowSize += 1;
		setTrackbarPos("SAD_window_size", "Track Bars", SADWindowSize);
		return;
	}
	if (typeofbm == BM_STATE) //SAD Window Size should be smaller than the image size
	{
		if (SADWindowSize >= img1r->width || SADWindowSize >= img1r->height)
		{
			cout << "SAD window size should be smaller" << endl;
			return;
		}
	}
	else
	{
		if (SADWindowSize >= leftImagerBM.rows || SADWindowSize >= leftImagerBM.cols)
		{
			cout << "SAD window size should be smaller" << endl;
			return;
		}
	}
	cout << "SAD Window Size = " << SADWindowSize<< endl;
	ComputeDisparityMap(typeofbm);
}
static void texture_threshold(int, void*)
{
	cout << "Texture Threshold= " << textureThreshold<< endl;
	ComputeDisparityMap(typeofbm);
}
static void uniqueness_ratio(int, void*)
{
	cout << "Uniqueness Ratio = " << uniquenessRatio << endl;
	ComputeDisparityMap(typeofbm);
}

int main(int, char**)
{
	std::string leftfilename = "images//left1.jpg";
	std::string rightfilename = "images//left1.jpg";

	Mat leftImage, rightImage;
	IplImage* img1 = cvLoadImage("images//left1.jpg", 0);
	IplImage* img2 = cvLoadImage("images//right1.jpg", 0);
	leftImage = imread(leftfilename, 0);
	rightImage = imread(rightfilename, 0);

	if (leftImage.empty() || rightImage.empty())
		return -1;

	imshow("Left", leftImage);
	imshow("Right", rightImage);

	CvMat* mx1c = (CvMat*)cvLoad("images//mx1.xml");
	CvMat* mx2c = (CvMat*)cvLoad("images//mx2.xml");
	CvMat* my1c = (CvMat*)cvLoad("images//my1.xml");
	CvMat* my2c = (CvMat*)cvLoad("images//my2.xml");
	Mat mx1bm = mx1c;
	Mat mx2bm = mx2c;
	Mat my1bm = my1c;
	Mat my2bm = my2c;
	if (typeofbm == BM_STATE)
	{
		cvRemap(img1, img1r, mx1c, my1c);
		cvRemap(img2, img2r, mx2c, my2c);
		cvNamedWindow("Left Rectified");
		cvShowImage("Left Rectified", img1r);
		cvNamedWindow("Right Rectified");
		cvShowImage("Right Rectified", img2r);
	}
	else
	{
		remap(leftImage, leftImagerBM, mx1bm, my1bm, INTER_LINEAR);
		remap(rightImage, rightImagerBM, mx2bm, my2bm, INTER_LINEAR);
		imshow("Left Rectified", leftImagerBM);
		imshow("Right Rectified", rightImagerBM);
	}

	cvNamedWindow("Disparity Map");
	cvNamedWindow("Disparity Map Normalized");
	namedWindow("Track Bars");
	ComputeDisparityMap(typeofbm);

	createTrackbar("min_disparities", "Track Bars", &minDisparity, 64, min_disparities);
	createTrackbar("num_of_disparities", "Track Bars", &numberOfDisparities, 300, num_of_disparities);
	createTrackbar("pre_filter_size", "Track Bars", &preFilterSize, 300, pre_filter_size);
	createTrackbar("pre_filter_cap", "Track Bars", &preFilterCap, 300, pre_filter_cap);
	createTrackbar("SAD_window_size", "Track Bars", &SADWindowSize, 480, SAD_window_size);
	createTrackbar("texture_threshold", "Track Bars", &textureThreshold, 300, texture_threshold);
	createTrackbar("uniqueness_ratio", "Track Bars", &uniquenessRatio, 300, uniqueness_ratio);

	for (;;) { if (waitKey(30) == 'x') break; }
	cvReleaseMat(&img1r);
	cvReleaseMat(&img2r);
	return 0;
}
