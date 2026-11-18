#pragma once
#include "DefinedDS.h"
#include "tinyxml2.h"

#include <fstream>
#include <filesystem>

namespace vlp {
	bool CheckFileSuffix(const std::string& filePath,
		const std::string& suffix);

	bool LoadCGNames(const std::string& filePath,
		std::vector<std::string>& cameraNames,
		std::vector<std::string>& galvoNames);

	bool LoadCamParameters(const std::string& filePath,
		vlp::CameraProperty& leftCam, vlp::CameraProperty& rightCam);

	bool LoadCamParametersFromYml(const std::string& filePath,
		vlp::CameraProperty& camera);

	bool LoadGalvoParameters(const std::string& filePath, cv::Mat& G);

	bool WriteImgCodes(const std::string& filePath,
		const std::map<int, std::vector<CGCodeBlock>>& leftData,
		const std::map<int, std::vector<CGCodeBlock>>& rightData);

	bool LoadImgCodes(const std::string& filePath,
		std::map<int, std::vector<CGCodeBlock>>& leftData,
		std::map<int, std::vector<CGCodeBlock>>& rightData);

	bool LoadImgCodesFromXml(const std::string& filePath,
		std::map<int, std::vector<CGCodeBlock>>& data);

	bool LoadObjCodes(const std::string& filePath,
		std::map<int, std::vector<cv::Point3f>>& objCodes);

	bool LoadObjCodesFromXml(const std::string& filePath,
		std::map<int, std::vector<cv::Point3f>>& objCodes);

	bool WriteCGParameters(const std::string& filePath,
		const cv::Mat& G1, const cv::Mat& K1,
		const cv::Mat& distCoeffs1, const cv::Mat& T1_g2c,
		const cv::Mat& G2, const cv::Mat& K2,
		const cv::Mat& distCoeffs2, const cv::Mat& T2_g2c,
		const cv::Mat& T_lg2rg);

	bool LoadCGParameters(const std::string& filePath,
		cv::Mat& G1, cv::Mat& K1,
		cv::Mat& distCoeffs1, cv::Mat& T1_g2c,
		cv::Mat& G2, cv::Mat& K2,
		cv::Mat& distCoeffs2, cv::Mat& T2_g2c,
		cv::Mat& T_lg2rg);

	bool WriteReprojError(const std::string& filePath,
		const std::map<int, std::vector<vlp::ReprojError>>& dev);

	bool WritePnPRt(const std::string& filePath,
		const std::map<int, std::vector<CGCodeBlock>>& multiScanCodes);

	void LoadPattern(const std::string& folderPath,
		std::vector<cv::Mat>& patterns);

	void LoadFeatureImgs(const std::string& folderPath,
		std::vector<cv::Mat>& imgs, std::vector<int>& galvoX,
		std::vector<int>& galvoY);

	void LoadFilePaths(const std::string& folderPath,
		std::vector<std::string>& filePaths,
		std::vector<int>& galvoX, std::vector<int>& galvoY);

	bool WritePointCloud(const std::string& filePath,
		const std::vector<cv::Point3d>& points);

	bool WritePointCloudBinary(const std::string& filePath,
		const std::vector<cv::Point3d>& points);

}// namespace vlp