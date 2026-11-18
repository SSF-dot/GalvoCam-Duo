#pragma once
#include <string>

namespace vlp {
	const std::string settingFilePath = "../configuration/setting.yaml";
	const std::string cameraFilePath = "../configuration/camera_parameters.yaml";
	const std::string galvoFilePath = "../configuration/galvo_parameters.yaml";
	const std::string objCodesFolder = "../obj_points";
	const std::string imgCodesFilePath = "../calibration_results/imgCodes.yaml";
	const std::string CGFilePath = "../calibration_results/CG.yaml";
	const std::string leftCGDev = "../calibration_results/left_cg_dev.txt";
	const std::string rightCGDev = "../calibration_results/right_cg_dev.txt";
	const std::string leftPNP = "../calibration_results/left_pnp.txt";
	const std::string rightPNP = "../calibration_results/right_pnp.txt";
}// namespace vlp
