#pragma once
#include "DefinedDS.h"
#include "Constant.h"
#include "IO.h"
#include "BA.h"
#include "Householder.h"

#include <algorithm>
#include <tuple>
#include <chrono>

namespace vlp {
	class DualCGCalibration
	{
	public:
		DualCGCalibration() = default;
		DualCGCalibration(const DualCGCalibration&) = default;
		DualCGCalibration& operator =(const DualCGCalibration&) = default;
		~DualCGCalibration() = default;

		void SetGalvoParameter(const cv::Mat& G);
		void SetCamParameter(const vlp::CameraProperty& leftCam, const vlp::CameraProperty& rightCam);
		void SetLeftData(const std::map<int, std::vector<CGCodeBlock>>& blocks);
		void SetRightData(const std::map<int, std::vector<CGCodeBlock>>& blocks);
		void SetObjData(const MultiScanObjCodes& allObjCodes);
		bool Run();

	private:
		void SolveIntrinsic();
		void SolvePnP();
		void SolveLeftAXZB();
		void SolveRightAXZB();

		vlp::MultiScanObjCodes allObjCodes_;
		std::map<int, std::tuple<double, double, double>> leftPnpError_, rightPnpError_;// mean max min
		
		cv::Mat G_;
		cg::Householder holder1_, holder2_;
		std::vector<cv::Mat> multiX1_g2w_, multiZ1_g2c_;
		std::vector<cv::Mat> multiX2_g2w_, multiZ2_g2c_;

	public:
		double reprojError_;
		vlp::CameraProperty leftCamera_, rightCamera_;
		cv::Mat optimizedG1_, optimizedG2_;
		cv::Mat T1_g2c_, T2_g2c_;
		cv::Mat T_lg2rg_;
		// key-编码点 value-扫描次数 视口图像
		std::map<int, std::vector<vlp::ReprojError>> leftReprojError_, rightReprojError_;
	};

}// namespace vlp