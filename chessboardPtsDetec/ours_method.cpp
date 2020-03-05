#include "ours_method.h"
#include <vector>
#include <stack>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "tinyxml2.h"

using namespace std;

cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg)
{
	if (firstImg.size != secondImg.size)
		return cv::Mat();

	if (firstImg.depth() == CV_32F && secondImg.depth() == CV_32F
		&& ((firstImg.channels() == 3 && secondImg.channels() == 3)
			|| (firstImg.channels() == 6 && secondImg.channels() == 6)))
	{
		int width = firstImg.cols;
		int height = firstImg.rows;

		cv::Mat res(height, width, CV_32FC1);

		if (firstImg.channels() == 3)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<cv::Vec3f>(j, i).dot(secondImg.at<cv::Vec3f>(j, i));
				}
			}
		}
		else if (firstImg.channels() == 6)
		{
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res.at<float>(j, i) = firstImg.at<cv::Vec6f>(j, i).dot(secondImg.at<cv::Vec6f>(j, i));
				}
			}
		}
		return res;
	}
	return firstImg.mul(secondImg);
}

cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps)
{
	if (guidedImg.size() != inputP.size())
		return cv::Mat();

	int width = guidedImg.cols;
	int height = guidedImg.rows;

	normalize(guidedImg, guidedImg, 0, 1, cv::NORM_MINMAX, CV_32F);
	normalize(inputP, inputP, 0, 1, cv::NORM_MINMAX, CV_32F);

	cv::Mat meanGuid;
	boxFilter(guidedImg, meanGuid, CV_32F, cv::Size(r, r));
	cv::Mat meanP;
	boxFilter(inputP, meanP, CV_32F, cv::Size(r, r));

	std::vector<cv::Mat> guidedImg_split;
	cv::split(guidedImg, guidedImg_split);

	std::vector<cv::Mat> corrGuid_split;
	cv::Mat corrGuidP;
	for (int i = 0; i < guidedImg_split.size(); i++)
	{
		cv::Mat corrGuid_channel;
		cv::boxFilter(guidedImg_split[i].mul(inputP), corrGuid_channel, CV_32F, cv::Size(r, r));
		corrGuid_split.push_back(corrGuid_channel);
	}
	merge(corrGuid_split, corrGuidP);

	cv::Mat corrGuid;
	boxFilter(guidedImg.mul(guidedImg), corrGuid, CV_32F, cv::Size(r, r));

	cv::Mat varGuid;
	varGuid = corrGuid - meanGuid.mul(meanGuid);

	std::vector<cv::Mat> meanGrid_split;
	cv::split(meanGuid, meanGrid_split);

	std::vector<cv::Mat> guidmul_split;
	cv::Mat meanGuidmulP;
	for (int i = 0; i < meanGrid_split.size(); i++)
	{
		cv::Mat guidmul_channel;
		guidmul_channel = meanGrid_split[i].mul(meanP);
		guidmul_split.push_back(guidmul_channel);
	}
	merge(guidmul_split, meanGuidmulP);

	cv::Mat covGuidP;
	covGuidP = corrGuidP - meanGuidmulP;

	//create image mask for matrix adding integer
	cv::Mat onesMat = cv::Mat::ones(varGuid.size(), varGuid.depth());
	cv::Mat mergeOnes;
	if (varGuid.channels() == 1)
	{
		mergeOnes = onesMat;
	}
	else if (varGuid.channels() == 3)
	{
		std::vector<cv::Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}
	else if (varGuid.channels() == 6)
	{
		std::vector<cv::Mat> oneChannel;
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);
		oneChannel.push_back(onesMat);

		merge(oneChannel, mergeOnes);
	}

	cv::Mat a = covGuidP / (varGuid + mergeOnes * eps);
	cv::Mat b = meanP - multiChl_to_oneChl_mul(a, meanGuid);

	cv::boxFilter(a, a, CV_32F, cv::Size(r, r));
	cv::boxFilter(b, b, CV_32F, cv::Size(r, r));

	cv::Mat filteredImg = multiChl_to_oneChl_mul(a, guidedImg) + b;
	return filteredImg;
}

void createMask_lines(cv::Mat& dst)
{
	vector<vector<cv::Point2i> > contours;
	{
		vector<cv::Point2i> oneContour;

		cv::Point2i p1(2559, 0);
		cv::Point2i p2(2061, 0);
		cv::Point2i p3(2070, 12);
		cv::Point2i p4(2086, 31);
		cv::Point2i p5(2116, 61);
		cv::Point2i p6(2141, 88);
		cv::Point2i p7(2179, 130);
		cv::Point2i p8(2246, 217);
		cv::Point2i p9(2314, 324);
		cv::Point2i p10(2329, 345);
		cv::Point2i p11(2346, 362);
		cv::Point2i p12(2397, 391);
		cv::Point2i p13(2425, 499);
		cv::Point2i p14(2418, 510);
		cv::Point2i p15(2424, 577);
		cv::Point2i p16(2414, 599);
		cv::Point2i p17(2400, 666);
		cv::Point2i p18(2399, 712);
		cv::Point2i p19(2400, 742);
		cv::Point2i p20(2401, 768);
		cv::Point2i p21(2402, 779);
		cv::Point2i p22(2404, 800);
		cv::Point2i p23(2407, 815);
		cv::Point2i p24(2409, 834);
		cv::Point2i p25(2416, 862);
		cv::Point2i p26(2425, 885);
		cv::Point2i p27(2425, 891);
		cv::Point2i p28(2418, 935);
		cv::Point2i p29(2424, 962);
		cv::Point2i p30(2403, 1050);
		cv::Point2i p31(2302, 1303);
		cv::Point2i p32(2221, 1439);
		cv::Point2i p33(2559, 1439);

		oneContour.push_back(p1);
		oneContour.push_back(p2);
		oneContour.push_back(p3);
		oneContour.push_back(p4);
		oneContour.push_back(p5);
		oneContour.push_back(p6);
		oneContour.push_back(p7);
		oneContour.push_back(p8);
		oneContour.push_back(p9);
		oneContour.push_back(p10);
		oneContour.push_back(p11);
		oneContour.push_back(p12);
		oneContour.push_back(p13);
		oneContour.push_back(p14);
		oneContour.push_back(p15);
		oneContour.push_back(p16);
		oneContour.push_back(p17);
		oneContour.push_back(p18);
		oneContour.push_back(p19);
		oneContour.push_back(p20);
		oneContour.push_back(p21);
		oneContour.push_back(p22);
		oneContour.push_back(p23);
		oneContour.push_back(p24);
		oneContour.push_back(p25);
		oneContour.push_back(p26);
		oneContour.push_back(p27);
		oneContour.push_back(p28);
		oneContour.push_back(p29);
		oneContour.push_back(p30);
		oneContour.push_back(p31);
		oneContour.push_back(p32);
		oneContour.push_back(p33);

		contours.push_back(oneContour);
	}
	//
	{
		vector<cv::Point2i> oneContour;

		cv::Point2i p1(0, 0);
		cv::Point2i p2(357, 0);
		cv::Point2i p3(274, 116);
		cv::Point2i p4(241, 179);
		cv::Point2i p5(191, 284);
		cv::Point2i p6(158, 372);
		cv::Point2i p7(130, 480);
		cv::Point2i p8(113, 590);
		cv::Point2i p9(107, 685);
		cv::Point2i p10(110, 801);
		cv::Point2i p11(122, 903);
		cv::Point2i p12(149, 1024);
		cv::Point2i p13(188, 1129);
		cv::Point2i p14(285, 1356);
		cv::Point2i p15(350, 1439);
		cv::Point2i p16(0, 1439);

		oneContour.push_back(p1);
		oneContour.push_back(p2);
		oneContour.push_back(p3);
		oneContour.push_back(p4);
		oneContour.push_back(p5);
		oneContour.push_back(p6);
		oneContour.push_back(p7);
		oneContour.push_back(p8);
		oneContour.push_back(p9);
		oneContour.push_back(p10);
		oneContour.push_back(p11);
		oneContour.push_back(p12);
		oneContour.push_back(p13);
		oneContour.push_back(p14);
		oneContour.push_back(p15);
		oneContour.push_back(p16);

		contours.push_back(oneContour);
	}

	int width = 2560;
	int height = 1440;
	cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);

	drawContours(img, contours, -1, 255, cv::FILLED);
	//imwrite("img_.jpg", img);
	bitwise_not(img, dst);
}

void createMask_lines2(cv::Mat& dst)
{
	vector<vector<cv::Point2i> > contours;
	{
		vector<cv::Point2i> oneContour;

		cv::Point2i p1(300, 0);
		cv::Point2i p2(118, 394);
		cv::Point2i p3(240, 464);
		cv::Point2i p4(149, 576);
		cv::Point2i p5(128, 603);
		cv::Point2i p6(116, 810);
		cv::Point2i p7(137, 887);
		cv::Point2i p8(144, 942);
		cv::Point2i p9(147, 985);
		cv::Point2i p10(144, 1028);
		cv::Point2i p11(202, 1226);
		cv::Point2i p12(236, 1298);
		cv::Point2i p13(241, 1328);
		cv::Point2i p14(310, 1435);
		cv::Point2i p15(0, 1439);
		cv::Point2i p16(0, 0);

		oneContour.push_back(p1);
		oneContour.push_back(p2);
		oneContour.push_back(p3);
		oneContour.push_back(p4);
		oneContour.push_back(p5);
		oneContour.push_back(p6);
		oneContour.push_back(p7);
		oneContour.push_back(p8);
		oneContour.push_back(p9);
		oneContour.push_back(p10);
		oneContour.push_back(p11);
		oneContour.push_back(p12);
		oneContour.push_back(p13);
		oneContour.push_back(p14);
		oneContour.push_back(p15);
		oneContour.push_back(p16);

		contours.push_back(oneContour);
	}
	//
	int width = 2560;
	int height = 1440;
	cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);

	cv::drawContours(img, contours, -1, 255, cv::FILLED);
	//imwrite("img_.jpg", img);
	bitwise_not(img, dst);
}

cv::Mat detectLines_(cv::Mat& src1, cv::Mat& src2, bool isHorizon)
{
	// Check type of img1 and img2
	if (src1.type() != CV_64FC1) {
		cv::Mat tmp;
		src1.convertTo(tmp, CV_64FC1);
		src1 = tmp;
	}
	if (src2.type() != CV_64FC1) {
		cv::Mat tmp;
		src2.convertTo(tmp, CV_64FC1);
		src2 = tmp;
	}
	cv::Mat diff = src1 - src2;
	cv::Mat cross = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	cv::Mat cross_inv = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	double thresh = 100;//185 for 20191017
	bool positive; // Whether previous found cross point was positive
	bool search; // Whether serching
	bool found_first;
	int val_now, val_prev;

	if (isHorizon)
	{
		// search for y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(0, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = 1; y < diff.rows; ++y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross.at<uchar>(y, x) != 255) {
							cross.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross.at<uchar>(y - 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(diff.rows - 1, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = diff.rows - 2; y > 0; --y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross_inv.at<uchar>(y, x) != 255) {
							cross_inv.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross_inv.at<uchar>(y + 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}
	}
	else
	{
		// search for x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, 0);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = 1; x < diff.cols; ++x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross.at<uchar>(y, x) = 255;
					}
					else {
						cross.at<uchar>(y, x - 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, diff.cols - 1);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = diff.cols - 2; x > 0; --x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross_inv.at<uchar>(y, x) = 255;
					}
					else {
						cross_inv.at<uchar>(y, x + 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}
	}

	cv::Mat dst_cross;
	cv::bitwise_and(cross, cross_inv, dst_cross);
	return dst_cross;
}

/**
 * \brief detecting lines for further work///methods following the function detectLine() in LineDetection.cpp
 * \param src1
 * \param src2
 * \param dst :image type is CV_8UC1
 * \param isHorizon
 */
void detectLines_(cv::Mat src1, cv::Mat src2, cv::Mat& dst, cv::Mat& dst_inv, bool isHorizon)
{
	// Check type of img1 and img2
	if (src1.type() != CV_64FC1) {
		cv::Mat tmp;
		src1.convertTo(tmp, CV_64FC1);
		src1 = tmp;
	}
	if (src2.type() != CV_64FC1) {
		cv::Mat tmp;
		src2.convertTo(tmp, CV_64FC1);
		src2 = tmp;
	}
	cv::Mat diff = src1 - src2;
	cv::Mat cross = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	cv::Mat cross_inv = cv::Mat::zeros(diff.rows, diff.cols, CV_8UC1);
	double thresh = 50;//185 for 20191017
	bool positive; // Whether previous found cross point was positive
	bool search; // Whether serching
	bool found_first;
	int val_now, val_prev;

	if (isHorizon)
	{
		// search for y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(0, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = 1; y < diff.rows; ++y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross.at<uchar>(y, x) != 255) {
							cross.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross.at<uchar>(y - 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed y direction
		for (int x = 0; x < diff.cols; x++) {
			val_prev = diff.at<double>(diff.rows - 1, x);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int y = diff.rows - 2; y > 0; --y) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						if (cross_inv.at<uchar>(y, x) != 255) {
							cross_inv.at<uchar>(y, x) = 255;
						}
					}
					else {
						cross_inv.at<uchar>(y + 1, x) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}
	}
	else
	{
		// search for x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, 0);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = 1; x < diff.cols; ++x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross.at<uchar>(y, x) = 255;
					}
					else {
						cross.at<uchar>(y, x - 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

		// search for inversed x direction
		for (int y = 0; y < diff.rows; y++) {
			val_prev = diff.at<double>(y, diff.cols - 1);
			positive = (val_prev > 0);
			search = false;
			found_first = false;
			for (int x = diff.cols - 2; x > 0; --x) {
				val_now = diff.at<double>(y, x);
				if (search && (
					((val_now <= 0) && positive) || ((val_now >= 0) && !positive))) {// found crossed point
					if (abs(val_now) < abs(val_prev)) {
						cross_inv.at<uchar>(y, x) = 255;
					}
					else {
						cross_inv.at<uchar>(y, x + 1) = 255;
					}
					positive = !positive;
					search = false;
				}
				if (!search && abs(val_now) > thresh) {
					search = true;
					if (!found_first) {
						found_first = true;
						positive = (val_now > 0);
					}
				}
				val_prev = val_now;
			}
		}

	}

	//cv::bitwise_and(cross, cross_inv, dst);
	dst = cross.clone();
	dst_inv = cross_inv.clone();
}

void connectEdge(cv::Mat& src, int winSize_thres, bool isHorizon)
{
	int width = src.cols;
	int height = src.rows;

	int half_winsize_thres = winSize_thres;

	if (isHorizon)
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y - 1, x) == 255 || src.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					int offset_x1[2] = { -1, 1 };
					//
					int starty = 1;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[0]--;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[0] >= 0 && x + offset_x1[0] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[0] / 2) = 255;
								if (offset_y1 / 2 <= 0 && offset_x1[0] / 2 <= 0 && starty > 2)
								{
									x = x + offset_x1[0] / 2 - 1;
									y = y + offset_y1 / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					starty = 1;
					num_8 = 0;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[1]++;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[1] >= 0 && x + offset_x1[1] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[1] / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

	}
	else
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y, x - 1) == 255 || src.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					int num_8 = 0;
					int offset_y1[2] = { -1, 1 };
					//
					int startx = 1;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[0]--;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[0] >= 0 && y + offset_y1[0] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[0] / 2, x + offset_x1 / 2) = 255;
								if (offset_x1 / 2 <= 0 && offset_y1[0] / 2 <= 0 && startx > 2)
								{
									x = x + offset_x1 / 2 - 1;
									y = y + offset_y1[0] / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					startx = 1;
					num_8 = 0;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[1]++;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[1] >= 0 && y + offset_y1[1] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[1] / 2, x + offset_x1 / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

	}
}

void connectEdge_(cv::Mat& src, int winSize_thres, bool isHorizon)
{
	int width = src.cols;
	int height = src.rows;

	int half_winsize_thres = winSize_thres;

	if (isHorizon)
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y - 1, x) == 255 || src.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					int offset_x1[2] = { -1, 1 };
					//
					int starty = 1;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							num_8++;
					}
					starty++;
					while (num_8 == 0 && -offset_x1[0] < half_winsize_thres)
					{
						offset_x1[0]--;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[0] >= 0 && x + offset_x1[0] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[0] / 2) = 255;
								if (offset_y1 / 2 <= 0 && offset_x1[0] / 2 <= 0 && starty > 2)
								{
									x = x + offset_x1[0] / 2 - 1;
									y = y + offset_y1 / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					starty = 1;
					num_8 = 0;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							num_8++;
					}
					starty++;
					while (num_8 == 0 && offset_x1[1] < half_winsize_thres)
					{
						offset_x1[1]++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[1] >= 0 && x + offset_x1[1] < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							{
								src.at<uchar>(y + offset_y1 / 2, x + offset_x1[1] / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

	}
	else
	{
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (src.at<uchar>(y, x) == 255)
				{
					if (src.at<uchar>(y, x - 1) == 255 || src.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					int num_8 = 0;
					int offset_y1[2] = { -1, 1 };
					//
					int startx = 1;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							num_8++;
					}
					startx++;
					while (num_8 == 0 && -offset_y1[0] < half_winsize_thres)
					{
						offset_y1[0]--;
						//startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[0] >= 0 && y + offset_y1[0] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[0] / 2, x + offset_x1 / 2) = 255;
								if (offset_x1 / 2 <= 0 && offset_y1[0] / 2 <= 0 && startx > 2)
								{
									x = x + offset_x1 / 2 - 1;
									y = y + offset_y1[0] / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
					//
					startx = 1;
					num_8 = 0;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							num_8++;
					}
					startx++;
					while (num_8 == 0 && offset_y1[1] < half_winsize_thres)
					{
						offset_y1[1]++;
						//startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[1] >= 0 && y + offset_y1[1] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (src.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							{
								src.at<uchar>(y + offset_y1[1] / 2, x + offset_x1 / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
	}
}

void connectEdge2(cv::Mat& src, int winSize_thres, bool isHorizon)
{
	int width = src.cols;
	int height = src.rows;

	int half_winsize_thres = winSize_thres;

	if (isHorizon)
	{
		cv::Mat tmp1, tmp2;
		tmp1 = src.clone();
		tmp2 = src.clone();

		int offset_x1[2];
		for (int y = height - 3; y <= 2; y--)
		{
			for (int x = width - 3; x <= 2; x--)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp1.at<uchar>(y, x) == 255)
				{
					if (tmp1.at<uchar>(y - 1, x) == 255 || tmp1.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					offset_x1[0] = -1;
					//
					int starty = 1;
					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (tmp1.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[0]--;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[0] >= 0 && x + offset_x1[0] < width))
							{
								continue;
							}
							if (tmp1.at<uchar>(y + offset_y1, x + offset_x1[0]) == 255)
							{
								tmp1.at<uchar>(y + offset_y1 / 2, x + offset_x1[0] / 2) = 255;
								if (offset_y1 / 2 <= 0 && offset_x1[0] / 2 <= 0 && starty > 2)
								{
									x = x + offset_x1[0] / 2 - 1;
									y = y + offset_y1 / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp2.at<uchar>(y, x) == 255)
				{
					if (tmp2.at<uchar>(y - 1, x) == 255 || tmp2.at<uchar>(y + 1, x) == 255)
					{
						continue;
					}
					//检查8邻域
					int num_8 = 0;
					int starty = 1;
					offset_x1[1] = 1;

					for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
					{
						if (tmp2.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							num_8++;
					}
					while (num_8 == 0 && starty < half_winsize_thres)
					{
						offset_x1[1]++;
						starty++;
						for (int offset_y1 = -starty; offset_y1 <= starty; offset_y1++)
						{
							if (!(y + offset_y1 >= 0 && y + offset_y1 < height && x + offset_x1[1] >= 0 && x + offset_x1[1] < width))
							{
								continue;
							}
							if (tmp2.at<uchar>(y + offset_y1, x + offset_x1[1]) == 255)
							{
								tmp2.at<uchar>(y + offset_y1 / 2, x + offset_x1[1] / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}

		bitwise_and(tmp1, tmp2, src);

	}
	else
	{

		cv::Mat tmp1, tmp2;
		tmp1 = src.clone();
		tmp2 = src.clone();

		int offset_y1[2];
		for (int y = height - 3; y <= 2; y--)
		{
			for (int x = width - 3; x <= 2; x--)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp1.at<uchar>(y, x) == 255)
				{
					if (tmp1.at<uchar>(y, x - 1) == 255 || tmp1.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					int num_8 = 0;
					offset_y1[0] = -1;
					//
					int startx = 1;
					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (tmp1.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[0]--;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[0] >= 0 && y + offset_y1[0] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (tmp1.at<uchar>(y + offset_y1[0], x + offset_x1) == 255)
							{
								tmp1.at<uchar>(y + offset_y1[0] / 2, x + offset_x1 / 2) = 255;
								if (offset_x1 / 2 <= 0 && offset_y1[0] / 2 <= 0 && startx > 2)
								{
									x = x + offset_x1 / 2 - 1;
									y = y + offset_y1[0] / 2 - 1;
								}
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
		//
		for (int y = 2; y < height - 2; y++)
		{
			for (int x = 2; x < width - 2; x++)
			{
				//如果该中心点为255,则考虑它的八邻域
				if (tmp2.at<uchar>(y, x) == 255)
				{
					if (tmp2.at<uchar>(y, x - 1) == 255 || tmp2.at<uchar>(y, x + 1) == 255)
					{
						continue;
					}

					//检查8邻域
					offset_y1[1] = 1;
					int num_8 = 0;
					int startx = 1;

					for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
					{
						if (tmp2.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							num_8++;
					}
					while (num_8 == 0 && startx < half_winsize_thres)
					{
						offset_y1[1]++;
						startx++;
						for (int offset_x1 = -startx; offset_x1 <= startx; offset_x1++)
						{
							if (!(y + offset_y1[1] >= 0 && y + offset_y1[1] < height && x + offset_x1 >= 0 && x + offset_x1 < width))
							{
								continue;
							}
							if (tmp2.at<uchar>(y + offset_y1[1], x + offset_x1) == 255)
							{
								tmp2.at<uchar>(y + offset_y1[1] / 2, x + offset_x1 / 2) = 255;
								num_8++;
								break;
							}
						}
					}
				}
			}
		}
		bitwise_and(tmp1, tmp2, src);
	}
}

void myGetLines(cv::Mat& src, cv::Mat& tmp, cv::Point2i startPt, std::vector<cv::Point2i>& oneLine, int lenThres, bool isHorizon)
{
	if (!oneLine.empty())
	{
		oneLine.clear();
	}
	if (isHorizon)
	{
		int max_x = startPt.x;
		int min_x = startPt.x;
		stack<int> p_x, p_y;
		p_x.push(startPt.x);
		p_y.push(startPt.y);
		while (!p_x.empty())
		{
			cv::Point2i p(p_x.top(), p_y.top());
			if (p.x > max_x)
			{
				max_x = p.x;
			}
			if (p.x < min_x)
			{
				min_x = p.x;
			}
			oneLine.push_back(p);
			p_x.pop(); p_y.pop();
			if ((p.y != 0 && p.x != 0) && (tmp.at<uchar>(p.y - 1, p.x - 1) == 255)) { // Top left
				tmp.at<uchar>(p.y - 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y - 1);
			}
			if ((p.y != 0) && (tmp.at<uchar>(p.y - 1, p.x) == 255)) { // Top
				tmp.at<uchar>(p.y - 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y - 1);
			}
			if ((p.y != 0 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y - 1, p.x + 1) == 255)) { // Top right
				tmp.at<uchar>(p.y - 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y - 1);
			}
			if ((p.x != 0) && (tmp.at<uchar>(p.y, p.x - 1) == 255)) { // left
				tmp.at<uchar>(p.y, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y);
			}
			if ((p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y, p.x + 1) == 255)) { // Right
				tmp.at<uchar>(p.y, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y);
			}
			if ((p.y != tmp.rows - 1 && p.x != 0) && (tmp.at<uchar>(p.y + 1, p.x - 1) == 255)) { // Down left
				tmp.at<uchar>(p.y + 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1) && (tmp.at<uchar>(p.y + 1, p.x) == 255)) { // Down
				tmp.at<uchar>(p.y + 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y + 1, p.x + 1) == 255)) { // Down right
				tmp.at<uchar>(p.y + 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y + 1);
			}
		}

		if ((max_x - min_x) < lenThres)
		{
			for (std::vector<cv::Point2i>::iterator it = oneLine.begin(); it != oneLine.end(); it++)
			{
				cv::Point2i pt = *it;
				src.at<uchar>(pt.y, pt.x) = 0;
			}
			oneLine.clear();
		}
	}
	else
	{
		int max_y = startPt.y;
		int min_y = startPt.y;
		stack<int> p_x, p_y;
		p_x.push(startPt.x);
		p_y.push(startPt.y);
		while (!p_x.empty())
		{
			cv::Point2i p(p_x.top(), p_y.top());
			if (p.y > max_y)
			{
				max_y = p.y;
			}
			if (p.y < min_y)
			{
				min_y = p.y;
			}
			oneLine.push_back(p);
			p_x.pop(); p_y.pop();
			if ((p.y != 0 && p.x != 0) && (tmp.at<uchar>(p.y - 1, p.x - 1) == 255)) { // Top left
				tmp.at<uchar>(p.y - 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y - 1);
			}
			if ((p.y != 0) && (tmp.at<uchar>(p.y - 1, p.x) == 255)) { // Top
				tmp.at<uchar>(p.y - 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y - 1);
			}
			if ((p.y != 0 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y - 1, p.x + 1) == 255)) { // Top right
				tmp.at<uchar>(p.y - 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y - 1);
			}
			if ((p.x != 0) && (tmp.at<uchar>(p.y, p.x - 1) == 255)) { // left
				tmp.at<uchar>(p.y, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y);
			}
			if ((p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y, p.x + 1) == 255)) { // Right
				tmp.at<uchar>(p.y, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y);
			}
			if ((p.y != tmp.rows - 1 && p.x != 0) && (tmp.at<uchar>(p.y + 1, p.x - 1) == 255)) { // Down left
				tmp.at<uchar>(p.y + 1, p.x - 1) = 0;
				p_x.push(p.x - 1); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1) && (tmp.at<uchar>(p.y + 1, p.x) == 255)) { // Down
				tmp.at<uchar>(p.y + 1, p.x) = 0;
				p_x.push(p.x); p_y.push(p.y + 1);
			}
			if ((p.y != tmp.rows - 1 && p.x != tmp.cols - 1) && (tmp.at<uchar>(p.y + 1, p.x + 1) == 255)) { // Down right
				tmp.at<uchar>(p.y + 1, p.x + 1) = 0;
				p_x.push(p.x + 1); p_y.push(p.y + 1);
			}
		}

		if ((max_y - min_y) < lenThres)
		{
			for (std::vector<cv::Point2i>::iterator it = oneLine.begin(); it != oneLine.end(); it++)
			{
				cv::Point2i pt = *it;
				src.at<uchar>(pt.y, pt.x) = 0;
			}
			oneLine.clear();
		}
	}
}

void removeShortEdges(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon, RIGHT_COUNT_SIDE mode)
{
	int width = src.cols;
	int height = src.rows;

	int count = 0;
	if (!lines.empty())
	{
		lines.clear();
	}

	cv::Mat tmp = src.clone();

	if (isHorizon)
	{
		if (mode == TOP_LEFT || mode == BOTTOM_LEFT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = 2; x < width - 2; x++)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
						}
					}
				}
			}
		}
		else if (mode == TOP_RIGHT || mode == BOTTOM_RIGHT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = width - 3; x > 1; x--)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
						}
					}
				}
			}
		}
	}
	else
	{
		for (int x = 2; x < width - 2; x++)
		{
			for (int y = 2; y < height - 2; y++)
			{
				if (tmp.at<uchar>(y, x) == 255)
				{
					std::vector<cv::Point2i> line;
					myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
					if (!line.empty())
					{
						lines[count] = line;
						count++;
					}
				}
			}
		}
	}
}

int removeShortEdges2(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon,
	RIGHT_COUNT_SIDE mode)
{
	int width = src.cols;
	int height = src.rows;

	int count = 0;
	if (!lines.empty())
	{
		lines.clear();
	}

	cv::Mat tmp = src.clone();

	int maxLen = 0;
	if (isHorizon)
	{
		if (mode == TOP_LEFT || mode == BOTTOM_LEFT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = 2; x < width - 2; x++)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
							if (maxLen < line.size())
							{
								maxLen = line.size();
							}
						}
					}
				}
			}
		}
		else if (mode == TOP_RIGHT || mode == BOTTOM_RIGHT)
		{
			for (int y = 2; y < height - 2; y++)
			{
				for (int x = width - 3; x > 1; x--)
				{
					if (tmp.at<uchar>(y, x) == 255)
					{
						std::vector<cv::Point2i> line;
						myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
						if (!line.empty())
						{
							lines[count] = line;
							count++;
							if (maxLen < line.size())
							{
								maxLen = line.size();
							}
						}
					}
				}
			}
		}
	}
	else
	{
		for (int x = 2; x < width - 2; x++)
		{
			for (int y = 2; y < height - 2; y++)
			{
				if (tmp.at<uchar>(y, x) == 255)
				{
					std::vector<cv::Point2i> line;
					myGetLines(src, tmp, cv::Point2i(x, y), line, lenThres, isHorizon);
					if (!line.empty())
					{
						lines[count] = line;
						count++;
						if (maxLen < line.size())
						{
							maxLen = line.size();
						}
					}
				}
			}
		}
	}
	return maxLen;
}

void post_removeShortEdges2(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon, RIGHT_COUNT_SIDE mode)
{
	int width = src.cols;
	int height = src.rows;

	if (isHorizon)
	{
		for (auto it = lines.begin(); it != lines.end(); it++)
		{
			if (it->second.size() < lenThres)
			{
				for (std::vector<cv::Point2i>::iterator it_ = it->second.begin(); it_ != it->second.end(); it_++)
				{
					cv::Point2i pt = *it_;
					src.at<uchar>(pt.y, pt.x) = 0;
				}
			}
		}
	}
	else
	{
		for (auto it = lines.begin(); it != lines.end(); it++)
		{
			if (it->second.size() < lenThres)
			{
				for (std::vector<cv::Point2i>::iterator it_ = it->second.begin(); it_ != it->second.end(); it_++)
				{
					cv::Point2i pt = *it_;
					src.at<uchar>(pt.y, pt.x) = 0;
				}
			}
		}
	}

	removeShortEdges2(src, lines, lenThres, isHorizon, mode);
}

void post_process(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, bool isHorizon, RIGHT_COUNT_SIDE mode)
{
	connectEdge(src, 5, isHorizon);
	connectEdge_(src, 10, isHorizon);
	connectEdge_(src, 10, isHorizon);
	connectEdge_(src, 10, isHorizon);
	//int maxLen = removeShortEdges2(src, lines, 100, isHorizon, mode);
	//post_removeShortEdges2(src, lines, maxLen / 2, isHorizon, mode);
	int maxLen = removeShortEdges2(src, lines, 100, isHorizon, mode);
	connectEdge_(src, 20, isHorizon);
	post_removeShortEdges2(src, lines, maxLen / 2, isHorizon, mode);
}

/**
 * \brief detect chess corners based on line detection
 * \param src
 * \param pts
 * \param ptsReal
 */
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal, double grid_size)
{
	cv::Mat lineV, lineV_inv;
	cv::Mat lineH, lineH_inv;

	cv::Mat src0 = getGuidedFilter(src[0], src[0], 7, 1e-6);
	cv::Mat src1 = getGuidedFilter(src[1], src[1], 7, 1e-6);
	cv::Mat src2 = getGuidedFilter(src[2], src[2], 7, 1e-6);
	cv::Mat src3 = getGuidedFilter(src[3], src[3], 7, 1e-6);
	detectLines_(src0, src1, lineV, lineV_inv, false);
	detectLines_(src2, src3, lineH, lineH_inv, true);

	//detectLines_(src[0], src[1], lineV, lineV_inv, false);
	//detectLines_(src[2], src[3], lineH, lineH_inv, true);

	bitwise_and(lineV, lineV_inv, lineV);
	bitwise_and(lineH, lineH_inv, lineH);

	std::map<int, std::vector<cv::Point2i> > lines_H, lines_V;
	post_process(lineH, lines_H, true);
	post_process(lineV, lines_V, false);

	std::map<cv::Point2i, cv::Point2d, myCmp_map>  pts_H;

	cv::Mat ptsImg;
	bitwise_and(lineH, lineV, ptsImg);
	//cv::Mat ptsImg_2;
	//bitwise_not(ptsImg, ptsImg_2);

	int height = ptsImg.rows;
	int width = ptsImg.cols;
	int half_winsize = 3;
	cv::Mat tmp = ptsImg.clone();
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (tmp.at<uchar>(y, x) == 255)
			{
				int h_key = -1;
				int v_key = -1;
				for (auto it = lines_H.begin();
					it != lines_H.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						h_key = it->first;
						break;
					}
				}
				for (auto it = lines_V.begin();
					it != lines_V.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						v_key = it->first;
						break;
					}
				}

				vector<cv::Point2i> originPts;
				int num = 0;
				for (int winy = -half_winsize; winy <= half_winsize; winy++)
				{
					for (int winx = -half_winsize; winx <= half_winsize; winx++)
					{
						if (!(y + winy >= 0 && y + winy < height && x + winx >= 0 && x + winx < width))
						{
							continue;
						}
						if (tmp.at<uchar>(y + winy, x + winx) == 255)
						{
							tmp.at<uchar>(y + winy, x + winx) = 0;
							originPts.push_back(cv::Point2i(x + winx, y + winy));
							num++;
						}
					}
				}
				cv::Point2f cornerPt;
				if (num > 1)
				{
					double sumX = 0.0, sumY = 0.0;
					for (int i = 0; i < originPts.size(); i++)
					{
						sumX += originPts[i].x;
						sumY += originPts[i].y;
					}
					cornerPt.x = sumX / num;
					cornerPt.y = sumY / num;
				}
				else if (num == 1)
				{
					cornerPt.x = x;
					cornerPt.y = y;
				}
				pts_H[cv::Point2i(h_key, v_key)] = cornerPt;
			}
		}
	}

	if (!pts.empty())
	{
		pts.clear();
	}
	for (auto it = pts_H.begin(); it != pts_H.end(); it++)
	{
		pts.push_back(it->second);
		ptsReal.push_back(cv::Point3f(it->first.x * grid_size * 1.0, it->first.y * grid_size * 1.0, 0));
	}
}

/**
 * \brief detecting the corners when the whole view is not avaliable
 * \param src
 * \param pts
 * \param ptsReal
 * \param grid_size
 * \param hNum
 * \param vNum
 * \param mode :indicate the complete side of the view
 */
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask)
{
	cv::Mat lineV, lineV_inv;
	cv::Mat lineH, lineH_inv;
	//cv::Mat src0 = getGuidedFilter(src[0], src[0], 7, 1e-6);
	//cv::Mat src1 = getGuidedFilter(src[1], src[1], 7, 1e-6);
	//cv::Mat src2 = getGuidedFilter(src[2], src[2], 7, 1e-6);
	//cv::Mat src3 = getGuidedFilter(src[3], src[3], 7, 1e-6);
	//detectLines_( src2, src3, lineV, lineV_inv, false);
	//detectLines_(src0, src1, lineH, lineH_inv, true);

	detectLines_(src[2], src[3], lineV, lineV_inv, false);
	detectLines_(src[0], src[1], lineH, lineH_inv, true);

	bitwise_and(lineV, lineV_inv, lineV);
	bitwise_and(lineH, lineH_inv, lineH);

	if (!mask.empty())
	{
		bitwise_and(lineV, mask, lineV);
		bitwise_and(lineH, mask, lineH);
	}
	std::map<int, std::vector<cv::Point2i> > lines_H, lines_V;
	post_process(lineH, lines_H, true, mode);
	post_process(lineV, lines_V, false, mode);

	cout << lines_H.size() << endl;
	cout << lines_V.size() << endl;

	std::map<cv::Point2i, cv::Point2f, myCmp_map>  pts_H;

	cv::Mat ptsImg;
	bitwise_and(lineH, lineV, ptsImg);
	cv::Mat ptsImg_;
	bitwise_or(lineH, lineV, ptsImg_);

	int height = ptsImg.rows;
	int width = ptsImg.cols;

	int max_h = -1;
	int max_v = -1;

	int half_winsize = 3;
	cv::Mat tmp = ptsImg.clone();
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (tmp.at<uchar>(y, x) == 255)
			{
				int h_key = -1;
				int v_key = -1;
				for (auto it = lines_H.begin();
					it != lines_H.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						h_key = it->first;
						break;
					}
				}
				for (auto it = lines_V.begin();
					it != lines_V.end(); it++)
				{
					auto vec_it = find(it->second.begin(), it->second.end(), cv::Point2i(x, y));
					if (vec_it != it->second.end())
					{
						v_key = it->first;
						break;
					}
				}

				if (max_h < h_key)
				{
					max_h = h_key;
				}
				if (max_v < v_key)
				{
					max_v = v_key;
				}

				vector<cv::Point2i> originPts;
				int num = 0;
				for (int winy = -half_winsize; winy <= half_winsize; winy++)
				{
					for (int winx = -half_winsize; winx <= half_winsize; winx++)
					{
						if (!(y + winy >= 0 && y + winy < height && x + winx >= 0 && x + winx < width))
						{
							continue;
						}
						if (tmp.at<uchar>(y + winy, x + winx) == 255)
						{
							tmp.at<uchar>(y + winy, x + winx) = 0;
							originPts.push_back(cv::Point2i(x + winx, y + winy));
							num++;
						}
					}
				}
				cv::Point2f cornerPt;
				if (num > 1)
				{
					double sumX = 0.0, sumY = 0.0;
					for (int i = 0; i < originPts.size(); i++)
					{
						sumX += originPts[i].x;
						sumY += originPts[i].y;
					}
					cornerPt.x = sumX / num;
					cornerPt.y = sumY / num;
				}
				else if (num == 1)
				{
					cornerPt.x = x;
					cornerPt.y = y;
				}
				pts_H[cv::Point2i(h_key, v_key)] = cornerPt;
			}
		}
	}

	if (!pts.empty())
	{
		pts.clear();
	}
	switch (mode)
	{
	case TOP_LEFT:
	{
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, it->first.y * grid_size, 0));
		}
	}
	break;
	case TOP_RIGHT:
	{
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
	break;
	case BOTTOM_LEFT:
	{
		int diff_h = hNum - 1 - max_h;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, it->first.y * grid_size, 0));
		}
	}
	break;
	case BOTTOM_RIGHT:
	{
		int diff_h = hNum - 1 - max_h;
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
	break;
	}

	//cv::Mat src_1, src_2, dst_1;
	//threshold(src[0], src_1, 80, 255, THRESH_BINARY);
	//threshold(src[2], src_2, 80, 255, THRESH_BINARY);
	//bitwise_xor(src_1, src_2, dst_1);

	if (!pts.empty())
	{
		cornerSubPix(src[4], pts, cv::Size(5, 5), cv::Size(-1, -1),
		             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 700, 1e-8));
	}
}

void detectPts2(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask)
{
	cv::Mat lineV, lineV_inv;
	cv::Mat lineH, lineH_inv;
	detectLines_(src[0], src[1], lineV, lineV_inv, false);
	detectLines_(src[2], src[3], lineH, lineH_inv, true);

	bitwise_and(lineV, lineV_inv, lineV);
	bitwise_and(lineH, lineH_inv, lineH);

	if (!mask.empty())
	{
		bitwise_and(lineV, mask, lineV);
		bitwise_and(lineH, mask, lineH);
	}
	std::map<int, std::vector<cv::Point2i> > lines_H, lines_V;
	post_process(lineH, lines_H, true, mode);
	post_process(lineV, lines_V, false, mode);

	std::map<cv::Point2i, cv::Point2f, myCmp_map>  pts_H;

	cv::Mat ptsImg;
	bitwise_and(lineH, lineV, ptsImg);
	cv::Mat ptsImg_;
	bitwise_or(lineH, lineV, ptsImg_);

	int height = ptsImg.rows;
	int width = ptsImg.cols;

	int max_h = lines_H.size() - 1;
	int max_v = lines_V.size() - 1;

	for (int h_i = 0; h_i <= max_h; h_i++)
	{
		cv::Mat h_img = cv::Mat::zeros(height, width, CV_8UC1);
		for (auto it = lines_H[h_i].begin(); it != lines_H[h_i].end(); it++)
		{
			h_img.at<uchar>(it->y, it->x) = 255;
		}
		for (int v_i = 0; v_i <= max_v; v_i++)
		{
			cv::Point key_pt(h_i, v_i);

			cv::Mat tmpImg = h_img.clone();
			for (auto it_ = lines_V[v_i].begin(); it_ != lines_V[v_i].end(); it_++)
			{
				tmpImg.at<uchar>(it_->y, it_->x) = 255;
			}

			cv::Mat cornerDst;
			cornerHarris(tmpImg, cornerDst, 15, 3, 0.04);
			cv::Point maxPt;
			minMaxLoc(cornerDst, NULL, NULL, NULL, &maxPt);

			pts_H[cv::Point(h_i, v_i)] = maxPt;
		}
	}
	if (!pts.empty())
	{
		pts.clear();
	}
	switch (mode)
	{
	case TOP_LEFT:
	{
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, it->first.y * grid_size, 0));
		}
	}
	break;
	case TOP_RIGHT:
	{
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f(it->first.x * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
	break;
	case BOTTOM_LEFT:
	{
		int diff_h = hNum - 1 - max_h;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, it->first.y * grid_size, 0));
		}
	}
	break;
	case BOTTOM_RIGHT:
	{
		int diff_h = hNum - 1 - max_h;
		int diff_v = vNum - 1 - max_v;
		for (auto it = pts_H.begin(); it != pts_H.end(); it++)
		{
			pts.push_back(it->second);
			ptsReal.push_back(cv::Point3f((diff_h + it->first.x) * grid_size, (diff_v + it->first.y) * grid_size, 0));
		}
	}
	break;
	}

	cornerSubPix(ptsImg_, pts, cv::Size(5, 5), cv::Size(-1, -1),
	             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 700, 1e-8));
}

void loadXML_imgPath(std::string xmlPath, cv::Size& imgSize, map<RIGHT_COUNT_SIDE, vector<vector<std::string> > >& path_)
{
	tinyxml2::XMLDocument doc;
	doc.LoadFile(xmlPath.c_str());
	tinyxml2::XMLElement *root = doc.FirstChildElement("all");
	imgSize.width = atoi(root->FirstChildElement("img_width")->GetText());
	imgSize.height = atoi(root->FirstChildElement("img_height")->GetText());


	vector<vector<std::string> > imgPath_tl;
	tinyxml2::XMLElement *root_tl = root->FirstChildElement("images_tl");
	tinyxml2::XMLElement *node_tl = root_tl->FirstChildElement("pair");
	while (node_tl) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_tl->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_tl.push_back(filenames);
		node_tl = node_tl->NextSiblingElement("pair");
	}
	if (!imgPath_tl.empty())
	{
		path_[TOP_LEFT] = imgPath_tl;
	}


	vector<vector<std::string> > imgPath_tr;
	tinyxml2::XMLElement *root_tr = root->FirstChildElement("images_tr");
	tinyxml2::XMLElement *node_tr = root_tr->FirstChildElement("pair");
	while (node_tr) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_tr->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_tr.push_back(filenames);
		node_tr = node_tr->NextSiblingElement("pair");
	}
	if (!imgPath_tr.empty())
	{
		path_[TOP_RIGHT] = imgPath_tr;
	}

	vector<vector<std::string> > imgPath_bl;
	tinyxml2::XMLElement *root_bl = root->FirstChildElement("images_bl");
	tinyxml2::XMLElement *node_bl = root_bl->FirstChildElement("pair");
	while (node_bl) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_bl->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_bl.push_back(filenames);
		node_bl = node_bl->NextSiblingElement("pair");
	}
	if (!imgPath_bl.empty())
	{
		path_[BOTTOM_LEFT] = imgPath_bl;
	}

	vector<vector<std::string> > imgPath_br;
	tinyxml2::XMLElement *root_br = root->FirstChildElement("images_br");
	tinyxml2::XMLElement *node_br = root_br->FirstChildElement("pair");
	while (node_br) {
		vector<string> filenames(5);

		tinyxml2::XMLElement *filename = node_br->FirstChildElement("pattern");
		int count;
		for (count = 0; count < 5; ++count) {
			if (!filename) {
				break;
			}
			filenames[count] = std::string(filename->GetText());
			filename = filename->NextSiblingElement("pattern");
		}
		imgPath_br.push_back(filenames);
		node_br = node_br->NextSiblingElement("pair");
	}
	if (!imgPath_br.empty())
	{
		path_[BOTTOM_RIGHT] = imgPath_br;
	}
}

bool ptsCalib_single2(std::string xmlFilePath, cv::Size& imgSize, douVecPt2f& pts, douVecPt3f& ptsReal, double gridSize,
	int hNum, int vNum, cv::Mat mask)
{
	map<RIGHT_COUNT_SIDE, vector<vector<std::string> > > imgPaths;
	loadXML_imgPath(xmlFilePath, imgSize, imgPaths);

	if (!pts.empty())
	{
		pts.clear();
	}
	if (!ptsReal.empty())
	{
		ptsReal.clear();
	}

	int count = 1;
	for (auto it = imgPaths.begin(); it != imgPaths.end(); it++)
	{
		if (it->first == TOP_LEFT || it->first == TOP_RIGHT || it->first == BOTTOM_LEFT || it->first == BOTTOM_RIGHT)
		{
			for (auto it_1 = it->second.begin(); it_1 != it->second.end(); it_1++)
			{
				std::cout << count << endl;

				vector<cv::Point2f> oneImgPts;
				vector<cv::Point3f> oneObjPts;

				if ((*it_1).size() != 5)
				{
					continue;
				}

				vector<cv::Mat> oneImgs;
				for (int i = 0; i < 5; i++)
				{
					cv::Mat img = cv::imread((*it_1)[i]);
					cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
					oneImgs.push_back(img);
				}

				detectPts(oneImgs, oneImgPts, oneObjPts, gridSize, hNum, vNum, it->first, mask);
				pts.push_back(oneImgPts);
				ptsReal.push_back(oneObjPts);
				count++;
			}
		}

	}

	return(!(pts.empty() || ptsReal.empty() || pts.size() != ptsReal.size()));
}