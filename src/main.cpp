#include "DualCGCalibration.h"

void DualCalibration()
{
	vlp::CameraProperty leftCam, rightCam;
	vlp::LoadCamParameters(vlp::cameraFilePath, leftCam, rightCam);

	cv::Mat G;
	vlp::LoadGalvoParameters(vlp::galvoFilePath, G);

	std::vector<std::string> objCodesFilePaths;
	for (const auto& entry : std::filesystem::directory_iterator(vlp::objCodesFolder)) {
		if (entry.is_regular_file() && entry.path().extension() == ".yaml") {		
			objCodesFilePaths.push_back(entry.path().string());
		}
	}

	// scanCount id code
	std::map<int, std::map<int, std::vector<cv::Point3f>>> allObjCodes;
	for (int i = 0; i < objCodesFilePaths.size(); i++) {
		std::map<int, std::vector<cv::Point3f>> objCodes;
		vlp::LoadObjCodes(objCodesFilePaths[i], objCodes);
		allObjCodes[i + 1] = objCodes;
	}
	
	auto calib = std::make_unique<vlp::DualCGCalibration>();
	calib->SetCamParameter(leftCam, rightCam);
	calib->SetGalvoParameter(G);
	calib->SetObjData(allObjCodes);

	std::map<int, std::vector<vlp::CGCodeBlock>> leftData;
	std::map<int, std::vector<vlp::CGCodeBlock>> rightData;
	vlp::LoadImgCodes(vlp::imgCodesFilePath, leftData, rightData);

	calib->SetLeftData(leftData);
	calib->SetRightData(rightData);

	calib->Run();

	vlp::WriteCGParameters(vlp::CGFilePath,
		calib->optimizedG1_, calib->leftCamera_.K_,
		calib->leftCamera_.distCoeffs_, calib->T1_g2c_,
		calib->optimizedG2_, calib->rightCamera_.K_,
		calib->rightCamera_.distCoeffs_, calib->T2_g2c_,
		calib->T_lg2rg_);

	vlp::WriteReprojError(vlp::leftCGDev, calib->leftReprojError_);
	vlp::WriteReprojError(vlp::rightCGDev, calib->rightReprojError_);

	vlp::WritePnPRt(vlp::leftPNP, calib->leftCamera_.multiScanCodes_);
	vlp::WritePnPRt(vlp::rightPNP, calib->rightCamera_.multiScanCodes_);

	cv::Mat G1_ = calib->optimizedG1_.clone();
	cv::Mat G2_ = calib->optimizedG2_.clone();

	vlp::CameraProperty leftCamera_, rightCamera_;
	leftCamera_.K_ = calib->leftCamera_.K_.clone();
	leftCamera_.distCoeffs_ = calib->leftCamera_.distCoeffs_.clone();
	rightCamera_.K_ = calib->rightCamera_.K_.clone();
	rightCamera_.distCoeffs_ = calib->rightCamera_.distCoeffs_.clone();

	cv::Mat T1_g2c_ = calib->T1_g2c_.clone();
	cv::Mat T2_g2c_ = calib->T2_g2c_.clone();
	cv::Mat T_lg2rg_ = calib->T_lg2rg_.clone();
}

int main()
{
	// calibrate GalvoCam Duo system
	DualCalibration();

	return 0;
}