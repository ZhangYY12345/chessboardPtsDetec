#include "ours_method.h"
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include <iostream>


using namespace std;
using namespace cv;

int main()
{
	double start, stop, duration;

	start = static_cast<double>(getTickCount());
	//ours
	cv::Mat img4 = imread("D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/20191231/patternsImgL/1_pattern4.jpg");
	cv::Mat img5 = img4.clone();
	cvtColor(img4, img4, COLOR_BGR2GRAY);
	{
		vector<cv::Mat> oneImgs;
		cv::Mat img0 = imread("D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/20191231/patternsImgL/1_pattern0.jpg");
		cvtColor(img0, img0, COLOR_BGR2GRAY);
		cv::Mat img1 = imread("D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/20191231/patternsImgL/1_pattern1.jpg");
		cvtColor(img1, img1, COLOR_BGR2GRAY);
		cv::Mat img2 = imread("D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/20191231/patternsImgL/1_pattern2.jpg");
		cvtColor(img2, img2, COLOR_BGR2GRAY);
		cv::Mat img3 = imread("D:/studying/stereo vision/research code/fisheye-stereo-calibrate/fisheyeStereoCalib/fisheyeStereoCalib/fisheyeStereoCalib/20191231/patternsImgL/1_pattern3.jpg");
		cvtColor(img3, img3, COLOR_BGR2GRAY);
		oneImgs.push_back(img0);
		oneImgs.push_back(img1);
		oneImgs.push_back(img2);
		oneImgs.push_back(img3);
		oneImgs.push_back(img4);
	
		double gridSize = 16.5;
		int hNum = 17;
		int vNum = 31;
		vector<cv::Point2f> oneImgPts;
		vector<cv::Point3f> oneObjPts;
		//cv::Mat mask_R;
		//createMask_lines2(mask_R);
		detectPts(oneImgs, oneImgPts, oneObjPts, gridSize, hNum, vNum, TOP_LEFT);//TOP_LEFT, 
	
		//save pts
		std::ofstream myfile;
		myfile.setf(ios::fixed, ios::floatfield);
		myfile.precision(6);
		string txtName = "ours_pts.txt";
		myfile.open(txtName, ios::out);
		for (int i = 0; i < oneImgPts.size(); i++)
		{
			myfile << oneImgPts.at(i).x << "\t " << oneImgPts.at(i).y << "\t\t " 
			<< oneObjPts.at(i).x << "\t " << oneObjPts.at(i).y << "\t " << oneObjPts.at(i).z << std::endl;
		}
		myfile.close();
	
	
		// draw corners
		if(!oneImgPts.empty())
		{
			cv::Mat img6 = img5.clone();
			int type = img6.type();
			int cn = CV_MAT_CN(type);
			CV_CheckType(type, cn == 1 || cn == 3 || cn == 4,
				"Number of channels must be 1, 3 or 4");
	
			int depth = CV_MAT_DEPTH(type);
			CV_CheckType(type, depth == CV_8U || depth == CV_16U || depth == CV_32F,
				"Only 8-bit, 16-bit or floating-point 32-bit images are supported");
	
			const int shift = 0;
			const int radius = 4;
			const int r = radius * (1 << shift);
	
			double scale = 1;
			switch (depth)
			{
			case CV_8U:
				scale = 1;
				break;
			case CV_16U:
				scale = 256;
				break;
			case CV_32F:
				scale = 1. / 255;
				break;
			}
			int line_type = (type == CV_8UC1 || type == CV_8UC3) ? LINE_AA : LINE_8;
	
			Scalar color(0, 0, 255, 0);
			if (cn == 1)
				color = Scalar::all(200);
			color *= scale;
	
			for(auto itor = oneImgPts.begin(); itor != oneImgPts.end(); itor++)
			{
				cv::Point2i pt(
					cvRound(itor->x*(1 << shift)),
					cvRound(itor->y*(1 << shift))
				);
	
				circle(img6, pt, r + (1 << shift), color, 1, line_type, shift);
			}
			imwrite("dst_ours.jpg", img6);
		}
		stop = static_cast<double>(getTickCount());
		duration = ((double)(stop - start)) / getTickFrequency(); //计算时间，以秒为单位
		std::cout << duration << "s" << endl;
	}
	
	//other common methods
	//{
	//	//opencv findChessboardCorners
	//	{
	//		cv::Mat img6 = img5.clone();
	//		int hNum = 17;
	//		int vNum = 31;

	//		vector<Point2f> cvPts;
	//		bool found = false;
	//		found = findChessboardCornersSB(img6, Size(vNum, hNum), cvPts, CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY);//
	//		if (found)
	//		{
	//			drawChessboardCorners(img6, Size(vNum, hNum), Mat(cvPts), found);
	//			std::ofstream myfile;
	//			myfile.setf(ios::fixed, ios::floatfield);
	//			myfile.precision(6);
	//			string txtName = "corner_12.txt";
	//			myfile.open(txtName, ios::out);
	//			for (int i = 0; i < cvPts.size(); i++)
	//			{
	//				float x_pos = cvPts.at(i).x;
	//				float y_pos = cvPts.at(i).y;

	//				myfile << x_pos << "\t" << y_pos << std::endl;
	//			}
	//			myfile.close();
	//			imwrite("dst_OpenCV_SB.jpg", img6);
	//		}
	//	}

	//	//libcdetect    // matlab code in http://www.cvlibs.net/software/libcbdetect/

	//	//
	//}
	return 0;
}