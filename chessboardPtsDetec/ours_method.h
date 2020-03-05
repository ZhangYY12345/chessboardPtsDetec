#pragma once
#include <opencv2/core/core.hpp>
#include <map>

typedef  std::vector<std::vector<cv::Point2f> >  douVecPt2f;
typedef  std::vector<std::vector<cv::Point3f> >  douVecPt3f;
typedef  std::vector<std::vector<cv::Point2d> >  douVecPt2d;
typedef  std::vector<std::vector<cv::Point3d> >  douVecPt3d;

typedef enum
{
	TOP_LEFT,
	TOP_RIGHT,
	BOTTOM_LEFT,
	BOTTOM_RIGHT,
}RIGHT_COUNT_SIDE;

struct myCmp_map
{
	bool operator() (const cv::Point2i& node1, const cv::Point2i& node2) const
	{
		if (node1.x < node2.x)
		{
			return true;
		}
		if (node1.x == node2.x)
		{
			if (node1.y < node2.y)
				return true;
		}
		return false;
	}
};

//guided filter
cv::Mat multiChl_to_oneChl_mul(cv::Mat firstImg, cv::Mat secondImg);
cv::Mat getGuidedFilter(cv::Mat guidedImg, cv::Mat inputP, int r, double eps);


//
void createMask_lines(cv::Mat& dst);
void createMask_lines2(cv::Mat& dst);

cv::Mat detectLines_(cv::Mat& src1, cv::Mat& src2, bool isHorizon);
void detectLines_(cv::Mat src1, cv::Mat src2, cv::Mat& dst, cv::Mat& dst_inv, bool isHorizon);
void connectEdge(cv::Mat& src, int winSize_thres, bool isHorizon = true);
void connectEdge_(cv::Mat& src, int winSize_thres, bool isHorizon = true);
void connectEdge2(cv::Mat& src, int winSize_thres, bool isHorizon = true);

void myGetLines(cv::Mat& src, cv::Mat& tmp, cv::Point2i startPt, std::vector<cv::Point2i>& oneLine, int lenThres, bool isHorizon = true);
void removeShortEdges(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon = true, RIGHT_COUNT_SIDE mode = TOP_LEFT);
int removeShortEdges2(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon = true, RIGHT_COUNT_SIDE mode = TOP_LEFT);
void post_removeShortEdges2(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, int lenThres, bool isHorizon = true, RIGHT_COUNT_SIDE mode = TOP_LEFT);
void post_process(cv::Mat& src, std::map<int, std::vector<cv::Point2i> >& lines, bool isHorizon = true, RIGHT_COUNT_SIDE mode = TOP_LEFT);
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal, double grid_size);
void detectPts(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask = cv::Mat::Mat());
void detectPts2(std::vector<cv::Mat>& src, std::vector<cv::Point2f>& pts, std::vector<cv::Point3f>& ptsReal,
	double grid_size, int hNum, int vNum, RIGHT_COUNT_SIDE mode, cv::Mat mask = cv::Mat::Mat());

void loadXML_imgPath(std::string xmlPath, cv::Size& imgSize, std::map<RIGHT_COUNT_SIDE, std::vector<std::vector<std::string> > >& path_);
bool ptsCalib_single2(std::string xmlFilePath, cv::Size& imgSize,
	douVecPt2f& pts, douVecPt3f& ptsReal, double gridSize, int hNum, int vNum, cv::Mat mask = cv::Mat::Mat());
