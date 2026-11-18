#pragma once
#include <unordered_set>
#include <map>
#include <opencv.hpp>

namespace vlp {
	// scan id code
	using MultiScanObjCodes = std::map<int, std::map<int, std::vector<cv::Point3f>>>;

	struct Task
	{
		cv::Mat img_;
		int group_;// 帧率
		int cameraIndex_;
		int galvoX_, galvoY_;
		int count_;// 扫描次数

		Task(cv::Mat& img, int group, int cameraIndex) :
			img_(img), cameraIndex_(cameraIndex), group_(group) {}

		Task(cv::Mat& img, int count, int group, int cameraIndex, int x, int y) :
			img_(img), count_(count), cameraIndex_(cameraIndex), group_(group), galvoX_(x), galvoY_(y) {}
	};

	struct ReprojError
	{
		double finalXError_, finalYError_, finalError_;
		int scanCount_;
		int galvoX_, galvoY_;
		cv::Point2f point_;
	};

	struct CGCodeBlock
	{
		int scanCount_;
		int frame_;
		int galvoX_, galvoY_;
		cv::Mat rvec_, tvec_;
		std::map<int, std::vector<cv::Point2f>> imgCodes_;

		CGCodeBlock() = default;

		CGCodeBlock(int count, int frame, int galvoX, int galvoY,
			const std::map<int, std::vector<cv::Point2f>>& codes) :
			scanCount_(count), frame_(frame),
			galvoX_(galvoX), galvoY_(galvoY), imgCodes_(codes) {}

		CGCodeBlock(int frame, int galvoX, int galvoY,
			const std::map<int, std::vector<cv::Point2f>>& codes) :
			frame_(frame), galvoX_(galvoX),
			galvoY_(galvoY), imgCodes_(codes) {}
	};

	struct CameraCodeBlock
	{
		int group_;
		int codeNumber_;
		std::vector<cv::Point2f> imgPoints_;

		CameraCodeBlock(int group, int codeNumber)
			:group_(group), codeNumber_(codeNumber) {}

		CameraCodeBlock(int group, int codeNumber,
			const std::vector<cv::Point2f>& imgPoints) :
			group_(group), codeNumber_(codeNumber), imgPoints_(imgPoints) {}

		bool operator==(const CameraCodeBlock& cameraCodeBlock) const
		{
			return ((this->codeNumber_ == cameraCodeBlock.codeNumber_) &&
				(this->group_ == cameraCodeBlock.group_));
		}
	};

	struct ImagePoint
	{
		int group_;
		int id_;
		cv::Point2f imagePoint_;

		ImagePoint() {};
		ImagePoint(int group, int id) :
			group_(group), id_(id) {}
		ImagePoint(int group, int id, cv::Point2f imagePoint) :
			group_(group), id_(id), imagePoint_(imagePoint) {}

		bool operator==(const ImagePoint& imagePoint) const
		{
			return ((imagePoint.group_ == this->group_) &&
				(imagePoint.id_ == this->id_));
		}
	};

	struct ImagePointHashFunction
	{
		std::size_t operator()(const ImagePoint& p) const
		{
			return (std::hash<int>()(p.group_) +
				std::hash<int>()(p.id_));
		}
	};

	struct ObjPoint
	{
		int group_;
		int id_;
		cv::Point3f objPoint_;
		cv::Vec3d vec_;

		ObjPoint() {};
		ObjPoint(int group, int id) :
			group_(group), id_(id) {};
		ObjPoint(int group, int id, cv::Point3f& objPoint) :
			group_(group), id_(id), objPoint_(objPoint)
		{
			vec_ = { objPoint.x, objPoint.y, objPoint.z };
		}

		bool operator==(const ObjPoint& objPoint) const
		{
			return ((objPoint.group_ == this->group_) &&
				(objPoint.id_ == this->id_));
		}
	};

	struct ObjPointHashFunction
	{
		std::size_t operator()(const ObjPoint& p) const
		{
			return (std::hash<int>()(p.group_) +
				std::hash<int>()(p.id_));
		}
	};

	struct CameraProperty
	{
		cv::Mat K_;	
		cv::Mat distCoeffs_;

		std::map<int, std::vector<CGCodeBlock>> multiScanCodes_;
	};

	struct NormalizationInfo
	{
		cv::Mat max;
		cv::Mat min;
	};

	struct Deviation {
		cv::Point3d point_;
		double deviation_;

		Deviation(const cv::Point3d& point, double dev) :
			point_(point), deviation_(dev) {};
	};

}// namespace vlp
