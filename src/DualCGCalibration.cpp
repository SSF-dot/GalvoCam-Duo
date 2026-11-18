#include "DualCGCalibration.h"

namespace vlp {
	void DualCGCalibration::SetGalvoParameter(const cv::Mat& G)
	{
		G_ = G.clone();

		double alpha = G.at<double>(0, 0);
		double beta = G.at<double>(1, 0);
		double e = G.at<double>(2, 0);

		double ka1 = G.at<double>(3, 0);
		double ka2 = G.at<double>(4, 0);
		double ka3 = G.at<double>(5, 0);

		double kb1 = G.at<double>(6, 0);
		double kb2 = G.at<double>(7, 0);
		double kb3 = G.at<double>(8, 0);

		double ka4 = G.at<double>(9, 0);
		double ka5 = G.at<double>(10, 0);
		double ka6 = G.at<double>(11, 0);
		double ka7 = G.at<double>(12, 0);
		double ka8 = G.at<double>(13, 0);
		double ka9 = G.at<double>(14, 0);

		double kb4 = G.at<double>(15, 0);
		double kb5 = G.at<double>(16, 0);
		double kb6 = G.at<double>(17, 0);
		double kb7 = G.at<double>(18, 0);
		double kb8 = G.at<double>(19, 0);
		double kb9 = G.at<double>(20, 0);

		holder1_.SetData(
			alpha, beta, e,
			ka1, ka2, ka3,
			kb1, kb2, kb3,
			ka4, ka5, ka6,
			ka7, ka8, ka9,
			kb4, kb5, kb6,
			kb7, kb8, kb9);
		holder2_.SetData(
			alpha, beta, e,
			ka1, ka2, ka3,
			kb1, kb2, kb3,
			ka4, ka5, ka6,
			ka7, ka8, ka9,
			kb4, kb5, kb6,
			kb7, kb8, kb9);
	}

	void DualCGCalibration::SetCamParameter(
		const vlp::CameraProperty& leftCam,
		const vlp::CameraProperty& rightCam)
	{
		leftCamera_ = leftCam;
		rightCamera_ = rightCam;
	}
	
	void DualCGCalibration::SetObjData(const MultiScanObjCodes& allObjCodes)
	{
		allObjCodes_ = allObjCodes;
	}

	void DualCGCalibration::SetLeftData(
		const std::map<int, std::vector<CGCodeBlock>>& blocks)
	{
		leftCamera_.multiScanCodes_ = blocks;
	}

	void DualCGCalibration::SetRightData(
		const std::map<int, std::vector<CGCodeBlock>>& blocks)
	{
		rightCamera_.multiScanCodes_ = blocks;
	}

	void DualCGCalibration::SolveIntrinsic()
	{
#pragma omp parallel sections
		{
#pragma omp section
			{
				for (auto& scan : leftCamera_.multiScanCodes_) {
					int scanCount = scan.first;
					auto it = allObjCodes_.find(scanCount);
					if (it == allObjCodes_.end()) {
						continue;
					}

					auto& objCodes = it->second;
					auto& viewPorts = scan.second;
					std::vector<std::vector<cv::Point3f>> allObjPoints;
					std::vector<std::vector<cv::Point2f>> allImgPoints;

					int step = viewPorts.size();
					step /= 20;
					for (size_t i = 0; i < viewPorts.size(); i += step) {
						auto& imgCodes = viewPorts[i].imgCodes_;

						std::vector<cv::Point2f> imgPoints;
						std::vector<cv::Point3f> objPoints;
						for (const auto& code : imgCodes) {
							int key = code.first;
							auto iter = objCodes.find(key);
							if (iter == objCodes.end()) {
								continue;
							}
							imgPoints.insert(imgPoints.end(),
								code.second.begin(), code.second.end());
							objPoints.insert(objPoints.end(),
								iter->second.begin(), iter->second.end());
						}
						allObjPoints.emplace_back(objPoints);
						allImgPoints.emplace_back(imgPoints);
					}

					cv::Mat K = leftCamera_.K_.clone();
					cv::Mat distCoeffs = leftCamera_.distCoeffs_.clone();
					int width = K.at<double>(0, 2) * 2;
					int height = K.at<double>(1, 2) * 2;
					cv::Size size(width, height);
					std::vector<cv::Mat> rvecs;
					std::vector<cv::Mat> tvecs;
								
					double error = cv::calibrateCamera(allObjPoints,
						allImgPoints, size, K, distCoeffs,
						rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS);					
					std::cout << "camera-1: " << error << "\n";

					leftCamera_.K_ = K.clone();
					leftCamera_.distCoeffs_ = distCoeffs.clone();

					break;
				}
			}
#pragma omp section
			{
				for (auto& scan : rightCamera_.multiScanCodes_) {
					int scanCount = scan.first;
					auto it = allObjCodes_.find(scanCount);
					if (it == allObjCodes_.end()) {
						continue;
					}

					auto& objCodes = it->second;
					auto& viewPorts = scan.second;
					std::vector<std::vector<cv::Point3f>> allObjPoints;
					std::vector<std::vector<cv::Point2f>> allImgPoints;

					int step = viewPorts.size();
					step /= 20;
					for (size_t i = 0; i < viewPorts.size(); i += step) {
						auto& imgCodes = viewPorts[i].imgCodes_;

						std::vector<cv::Point2f> imgPoints;
						std::vector<cv::Point3f> objPoints;
						for (const auto& code : imgCodes) {
							int key = code.first;
							auto iter = objCodes.find(key);
							if (iter == objCodes.end()) {
								continue;
							}
							imgPoints.insert(imgPoints.end(),
								code.second.begin(), code.second.end());
							objPoints.insert(objPoints.end(),
								iter->second.begin(), iter->second.end());
						}
						allObjPoints.emplace_back(objPoints);
						allImgPoints.emplace_back(imgPoints);
					}

					cv::Mat K = rightCamera_.K_.clone();
					cv::Mat distCoeffs = rightCamera_.distCoeffs_.clone();
					int width = K.at<double>(0, 2) * 2;
					int height = K.at<double>(1, 2) * 2;
					cv::Size size(width, height);
					std::vector<cv::Mat> rvecs;
					std::vector<cv::Mat> tvecs;
					double error = cv::calibrateCamera(allObjPoints,
						allImgPoints, size, K, distCoeffs,
						rvecs, tvecs, cv::CALIB_USE_INTRINSIC_GUESS);
					std::cout << "camera-2: " << error << "\n";

					rightCamera_.K_ = K.clone();
					rightCamera_.distCoeffs_ = distCoeffs.clone();

					break;
				}
			}
		}	
	}

	void DualCGCalibration::SolvePnP()
	{
#pragma omp parallel sections
		{
#pragma omp section
			{
				for (auto& scan : leftCamera_.multiScanCodes_) {
					int scanCount = scan.first;
					auto it = allObjCodes_.find(scanCount);
					if (it == allObjCodes_.end()) {
						continue;
					}

					auto objCodes = it->second;
					auto& viewPorts = scan.second;
					std::vector<double> errors1;
					for (size_t i = 0; i < viewPorts.size(); i++) {
						auto imgCodes = viewPorts[i].imgCodes_;

						std::vector<cv::Point2f> imgPoints;
						std::vector<cv::Point3f> objPoints;
						for (const auto& code : imgCodes) {
							int key = code.first;
							auto iter = objCodes.find(key);
							if (iter == objCodes.end()) {
								continue;
							}
							imgPoints.insert(imgPoints.end(),
								code.second.begin(), code.second.end());
							objPoints.insert(objPoints.end(),
								iter->second.begin(), iter->second.end());
						}
						cv::Mat rvec, tvec;
						cv::solvePnP(objPoints, imgPoints,
							leftCamera_.K_, leftCamera_.distCoeffs_,
							rvec, tvec, 0, cv::SOLVEPNP_ITERATIVE);
						viewPorts[i].rvec_ = rvec.clone();
						viewPorts[i].tvec_ = tvec.clone();

						std::vector<cv::Point2f> reprojPoints;;
						cv::projectPoints(objPoints, rvec, tvec,
							leftCamera_.K_, leftCamera_.distCoeffs_,
							reprojPoints);
						double e = 0.;
						for (size_t j = 0; j < reprojPoints.size(); j++) {
							e += cv::norm(reprojPoints[j] - imgPoints[j]);
						}
						e /= reprojPoints.size();
						errors1.push_back(e);
					}
					
					double sum = std::accumulate(errors1.begin(), errors1.end(), 0.0);
					double mean = sum / errors1.size();
					double maxVal = *std::max_element(errors1.begin(), errors1.end());
					double minVal = *std::min_element(errors1.begin(), errors1.end());
					leftPnpError_[scan.first] = std::make_tuple(mean, maxVal, minVal);
				}
			}
#pragma omp section
			{
				for (auto& scan : rightCamera_.multiScanCodes_) {
					int scanCount = scan.first;
					auto it = allObjCodes_.find(scanCount);
					if (it == allObjCodes_.end()) {
						continue;
					}

					auto objCodes = it->second;
					auto& viewPorts = scan.second;
					std::vector<double> errors2;
					for (size_t i = 0; i < viewPorts.size(); i++) {
						auto imgCodes = viewPorts[i].imgCodes_;

						std::vector<cv::Point2f> imgPoints;
						std::vector<cv::Point3f> objPoints;
						for (const auto& code : imgCodes) {
							int key = code.first;
							auto iter = objCodes.find(key);
							if (iter == objCodes.end()) {
								continue;
							}
							imgPoints.insert(imgPoints.end(),
								code.second.begin(), code.second.end());
							objPoints.insert(objPoints.end(),
								iter->second.begin(), iter->second.end());
						}
						cv::Mat rvec, tvec;
						cv::solvePnP(objPoints, imgPoints,
							rightCamera_.K_, rightCamera_.distCoeffs_,
							rvec, tvec, 0, cv::SOLVEPNP_ITERATIVE);
						viewPorts[i].rvec_ = rvec.clone();
						viewPorts[i].tvec_ = tvec.clone();

						std::vector<cv::Point2f> reprojPoints;;
						cv::projectPoints(objPoints, rvec, tvec,
							rightCamera_.K_, rightCamera_.distCoeffs_,
							reprojPoints);
						double e = 0.;
						for (size_t j = 0; j < reprojPoints.size(); j++) {
							e += cv::norm(reprojPoints[j] - imgPoints[j]);
						}
						e /= reprojPoints.size();
						errors2.push_back(e);
					}
					double sum = std::accumulate(errors2.begin(), errors2.end(), 0.0);
					double mean = sum / errors2.size();
					double maxVal = *std::max_element(errors2.begin(), errors2.end());
					double minVal = *std::min_element(errors2.begin(), errors2.end());
					rightPnpError_[scan.first] = std::make_tuple(mean, maxVal, minVal);
				}
			}
		}
	}

	void DualCGCalibration::SolveLeftAXZB()
	{
		for (const auto& scan : leftCamera_.multiScanCodes_) {
			//构造 AX=ZB 求解初值
			//A 标定板（world）到相机坐标系的点变换
			std::vector<cv::Mat> R_world2cam;
			std::vector<cv::Mat> t_world2cam;
			//X base到标定板（world）坐标系的点变换
			cv::Mat R_base2world;
			cv::Mat t_base2world;
			//Z gripper到相机坐标系的点变换
			cv::Mat R_gripper2cam;
			cv::Mat t_gripper2cam;
			//B base到gripper坐标系的点变换 (H2*H1).inv()
			std::vector<cv::Mat> R_base2gripper;
			std::vector<cv::Mat> t_base2gripper;

			for (const auto& view : scan.second) {
				cv::Mat R;
				cv::Rodrigues(view.rvec_, R);
				R_world2cam.push_back(R);
				t_world2cam.push_back(view.tvec_);

				cv::Mat H1 = holder1_.CalculateH1(view.galvoX_ / 100., view.galvoY_ / 100.);
				cv::Mat H2 = holder1_.CalculateH2(view.galvoX_ / 100., view.galvoY_ / 100.);
				cv::Mat H = H1 * H2;
				R_base2gripper.push_back(H(cv::Rect(0, 0, 3, 3)));
				t_base2gripper.push_back(H(cv::Rect(3, 0, 1, 3)));
			}
			cv::calibrateRobotWorldHandEye(
				R_world2cam, t_world2cam,
				R_base2gripper, t_base2gripper,
				R_base2world, t_base2world,
				R_gripper2cam, t_gripper2cam,
				cv::CALIB_ROBOT_WORLD_HAND_EYE_SHAH);

			cv::Mat X_g2w = cv::Mat::eye(4, 4, CV_64FC1);
			R_base2world.copyTo(X_g2w(cv::Rect(0, 0, 3, 3)));
			t_base2world.copyTo(X_g2w(cv::Rect(3, 0, 1, 3)));

			cv::Mat Z_g2c = cv::Mat::eye(4, 4, CV_64FC1);
			R_gripper2cam.copyTo(Z_g2c(cv::Rect(0, 0, 3, 3)));
			t_gripper2cam.copyTo(Z_g2c(cv::Rect(3, 0, 1, 3)));

			multiX1_g2w_.push_back(X_g2w);
			multiZ1_g2c_.push_back(Z_g2c);

			//{
			//	for (int i = 0; i < R_world2cam.size(); i++) {
			//		cv::Mat A = cv::Mat::eye(4, 4, CV_64FC1);
			//		cv::Mat B = cv::Mat::eye(4, 4, CV_64FC1);
			//		R_world2cam[i].copyTo(A(cv::Rect(0, 0, 3, 3)));
			//		t_world2cam[i].copyTo(A(cv::Rect(3, 0, 1, 3)));
			//		R_base2gripper[i].copyTo(B(cv::Rect(0, 0, 3, 3)));
			//		t_base2gripper[i].copyTo(B(cv::Rect(3, 0, 1, 3)));

			//		cv::Mat delta = A * X_g2w - Z_g2c * B;
			//	}
			//}
		}
	}

	void DualCGCalibration::SolveRightAXZB()
	{
		for (const auto& scan : rightCamera_.multiScanCodes_) {
			//构造 AX=ZB 求解初值
			//A 标定板（world）到相机坐标系的点变换
			std::vector<cv::Mat> R_world2cam;
			std::vector<cv::Mat> t_world2cam;
			//X base到标定板（world）坐标系的点变换
			cv::Mat R_base2world;
			cv::Mat t_base2world;
			//Z gripper到相机坐标系的点变换
			cv::Mat R_gripper2cam;
			cv::Mat t_gripper2cam;
			//B base到gripper坐标系的点变换 H1*H2
			std::vector<cv::Mat> R_base2gripper;
			std::vector<cv::Mat> t_base2gripper;

			for (const auto& view : scan.second) {
				cv::Mat R;
				cv::Rodrigues(view.rvec_, R);
				R_world2cam.push_back(R);
				t_world2cam.push_back(view.tvec_);

				cv::Mat H1 = holder2_.CalculateH1(view.galvoX_ / 100., view.galvoY_ / 100.);
				cv::Mat H2 = holder2_.CalculateH2(view.galvoX_ / 100., view.galvoY_ / 100.);
				cv::Mat H = H1 * H2;
				R_base2gripper.push_back(H(cv::Rect(0, 0, 3, 3)));
				t_base2gripper.push_back(H(cv::Rect(3, 0, 1, 3)));
			}
			cv::calibrateRobotWorldHandEye(
				R_world2cam, t_world2cam,
				R_base2gripper, t_base2gripper,
				R_base2world, t_base2world,
				R_gripper2cam, t_gripper2cam,
				cv::CALIB_ROBOT_WORLD_HAND_EYE_SHAH);

			cv::Mat X_g2w = cv::Mat::eye(4, 4, CV_64FC1);
			R_base2world.copyTo(X_g2w(cv::Rect(0, 0, 3, 3)));
			t_base2world.copyTo(X_g2w(cv::Rect(3, 0, 1, 3)));

			cv::Mat Z_g2c = cv::Mat::eye(4, 4, CV_64FC1);
			R_gripper2cam.copyTo(Z_g2c(cv::Rect(0, 0, 3, 3)));
			t_gripper2cam.copyTo(Z_g2c(cv::Rect(3, 0, 1, 3)));

			multiX2_g2w_.push_back(X_g2w);
			multiZ2_g2c_.push_back(Z_g2c);

			//{
			//	for (int i = 0; i < R_world2cam.size(); i++) {
			//		cv::Mat A = cv::Mat::eye(4, 4, CV_64FC1);
			//		cv::Mat B = cv::Mat::eye(4, 4, CV_64FC1);
			//		R_world2cam[i].copyTo(A(cv::Rect(0, 0, 3, 3)));
			//		t_world2cam[i].copyTo(A(cv::Rect(3, 0, 1, 3)));
			//		R_base2gripper[i].copyTo(B(cv::Rect(0, 0, 3, 3)));
			//		t_base2gripper[i].copyTo(B(cv::Rect(3, 0, 1, 3)));

			//		cv::Mat delta = A * X_g2w - Z_g2c * B;
			//	}
			//}
		}
	}

	bool DualCGCalibration::Run()
	{
		//auto start0 = std::chrono::high_resolution_clock::now();
		SolveIntrinsic();
		//auto end0 = std::chrono::high_resolution_clock::now();

		//auto start1 = std::chrono::high_resolution_clock::now();
		SolvePnP();
		//auto end1 = std::chrono::high_resolution_clock::now();
		{
			std::cout << "CG-1*******" << "\n";
			for (const auto& err : leftPnpError_) {
				std::cout << "scan:" << err.first << "\t" <<
					std::get<0>(err.second) << "\t" <<
					std::get<1>(err.second) << "\t" <<
					std::get<2>(err.second) << "\n";
			}
			std::cout << "CG-2*******" << "\n";
			for (const auto& err : rightPnpError_) {
				std::cout << "scan:" << err.first << "\t" <<
					std::get<0>(err.second) << "\t" <<
					std::get<1>(err.second) << "\t" <<
					std::get<2>(err.second) << "\n";
			}
		}

		//auto start2 = std::chrono::high_resolution_clock::now();
		SolveLeftAXZB();
		SolveRightAXZB();
		//auto end2 = std::chrono::high_resolution_clock::now();

		cv::Mat G1 = G_.clone();
		cv::Mat G2 = G_.clone();
		// world->g1
		std::vector<cv::Mat> multiX1_w2g;
		for (size_t i = 0; i < multiX1_g2w_.size(); i++) {
			multiX1_w2g.push_back(multiX1_g2w_[i].inv());
		}
		// world->g2
		std::vector<cv::Mat> multiX2_w2g;
		for (size_t i = 0; i < multiX2_g2w_.size(); i++) {
			multiX2_w2g.push_back(multiX2_g2w_[i].inv());
		}

		// g1->g2
		//std::vector<cv::Mat> multiT_lg2rg;
		//for (size_t i = 0; i < multiX1_g2w_.size(); i++) {
		//	multiT_lg2rg.push_back(
		//		multiX2_w2g[i] * multiX1_g2w_[i]);
		//}

		leftReprojError_.clear();
		rightReprojError_.clear();

		//auto start3 = std::chrono::high_resolution_clock::now();
		//bool flag = BA::Optimize(
		//	allObjCodes_,
		//	leftCamera_, rightCamera_,
		//	G1, G2, multiX1_w2g, multiZ1_g2c_,
		//	multiT_lg2rg, multiZ2_g2c_,
		//	leftReprojError_, rightReprojError_);
		//auto end3 = std::chrono::high_resolution_clock::now();

		
		//auto t0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0 - start0).count();
		//auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
		//auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
		//auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count();
		//std::cout << "代码运行时间: " << t0 << " 毫秒" << std::endl;
		//std::cout << "代码运行时间: " << t1 << " 毫秒" << std::endl;
		//std::cout << "代码运行时间: " << t2 << " 毫秒" << std::endl;
		//std::cout << "代码运行时间: " << t3 << " 毫秒" << std::endl;

		//{
		//	cv::Mat R = (cv::Mat_<double>(3, 3) <<
		//		0., 0., 1.,
		//		0., -1., 0.,
		//		1., 0., 0.);
		//	for (int i = 0; i < multiZ1_g2c_.size(); i++) {
		//		R.copyTo(multiZ1_g2c_[i](cv::Rect(0, 0, 3, 3)));
		//		R.copyTo(multiZ2_g2c_[i](cv::Rect(0, 0, 3, 3)));
		//	}
		//}

		(void)BA::Optimize(allObjCodes_, leftCamera_,
			G1, multiX1_w2g, multiZ1_g2c_, leftReprojError_);
		
		(void)BA::Optimize(allObjCodes_, rightCamera_,
			G2, multiX2_w2g, multiZ2_g2c_, rightReprojError_);
		T_lg2rg_ = multiX2_w2g[0] * multiX1_w2g[0].inv();
		


		optimizedG1_ = G1.clone();
		optimizedG2_ = G2.clone();
		//T_lg2rg_ = multiT_lg2rg[0].clone();
		T1_g2c_ = multiZ1_g2c_[0].clone();
		T2_g2c_ = multiZ2_g2c_[0].clone();
		return true;
	}

}// namespace vlp