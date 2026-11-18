#include "BA.h"

namespace vlp {
	bool BA::Optimize(
		const vlp::MultiScanObjCodes& allObjCodes,
		vlp::CameraProperty& camera, cv::Mat& G,
		std::vector<cv::Mat>& multiX_w2g,
		std::vector<cv::Mat>& multiZ_g2c,
		std::map<int, std::vector<vlp::ReprojError>>& dev)
	{
		double camIntr[4] = {
			camera.K_.at<double>(0, 0), // fx
			camera.K_.at<double>(0, 2), // u0
			camera.K_.at<double>(1, 1), // fy
			camera.K_.at<double>(1, 2) }; // v0

		std::vector<cv::Mat> multirvec_w2g(multiX_w2g.size());
		std::vector<cv::Mat> multitvec_w2g(multiX_w2g.size());
		for (int i = 0; i < multiX_w2g.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multiX_w2g[i](cv::Rect(0, 0, 3, 3)), rvec);
			multirvec_w2g[i] = rvec.clone();
			multiX_w2g[i](cv::Rect(3, 0, 1, 3)).copyTo(multitvec_w2g[i]);			
		}

		std::vector<cv::Mat> multirvec_g2c(multiZ_g2c.size());
		std::vector<cv::Mat> multitvec_g2c(multiZ_g2c.size());
		for (int i = 0; i < multiZ_g2c.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multiZ_g2c[i](cv::Rect(0, 0, 3, 3)), rvec);
			multirvec_g2c[i] = rvec.clone();
			multiZ_g2c[i](cv::Rect(3, 0, 1, 3)).copyTo(multitvec_g2c[i]);
		}			

		ceres::Problem problem;		

		std::vector<double> leftErrors;

		for (const auto& scan : camera.multiScanCodes_) {
			int count = scan.first;
			auto it = allObjCodes.find(count);
			if (it == allObjCodes.end()) {
				continue;
			}

			auto objCodes = it->second;
			auto viewports = scan.second;
			for (const auto& view : viewports) {
				double galvoX = view.galvoX_ / 100.;
				double galvoY = view.galvoY_ / 100.;
				auto& imgCodes = view.imgCodes_;
				for (const auto& imgCode : imgCodes) {
					int key = imgCode.first;
					auto iter = objCodes.find(key);
					if (iter == objCodes.end()) {
						continue;
					}
					auto points = imgCode.second;
					for (int i = 0; i < points.size(); i++) {
						cv::Point2f imgPoint = points[i];
						cv::Point3f objPoint = iter->second[i];								

						{
							CGReprojError cgReproj(objPoint, imgPoint, galvoX, galvoY);
							double errors[2] = { 0.,0. };
							cgReproj(
								reinterpret_cast<double*>(G.data),
								camIntr,
								reinterpret_cast<double*>(camera.distCoeffs_.data),
								reinterpret_cast<double*>(multirvec_w2g[count - 1].data),
								reinterpret_cast<double*>(multitvec_w2g[count - 1].data),
								reinterpret_cast<double*>(multirvec_g2c[count - 1].data),
								reinterpret_cast<double*>(multitvec_g2c[count - 1].data),
								errors);
							leftErrors.emplace_back(cv::norm(cv::Vec2d(errors[0], errors[1])));
						}

						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
							CGReprojError, 2, 21, 4, 5, 3, 3, 3, 3>(
								new CGReprojError(objPoint, imgPoint, galvoX, galvoY));
						problem.AddResidualBlock(costFunction, NULL,
							reinterpret_cast<double*>(G.data),
							camIntr,
							reinterpret_cast<double*>(camera.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec_w2g[count - 1].data),					
							reinterpret_cast<double*>(multirvec_g2c[0].data),
							reinterpret_cast<double*>(multitvec_g2c[0].data));
					}
				}
			}
			//problem.SetParameterBlockConstant(
			//	reinterpret_cast<double*>(multirvec_g2c[count - 1].data));
		}
		
		problem.SetParameterBlockConstant(
			reinterpret_cast<double*>(camIntr));
		problem.SetParameterBlockConstant(
			reinterpret_cast<double*>(camera.distCoeffs_.data));

		ceres::Solver::Options options;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.max_num_iterations = 100;
		options.num_threads = 10;
		//options.function_tolerance /= 100;
		//options.gradient_tolerance /= 100;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return false;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		{
			double mean = std::accumulate(leftErrors.begin(), leftErrors.end(), 0.);
			mean /= leftErrors.size();
			double maxVal = *std::max_element(leftErrors.begin(), leftErrors.end());
			double minVal = *std::min_element(leftErrors.begin(), leftErrors.end());
			std::cout << "C-G init: " << mean << "\t" << maxVal << "\t" << minVal << "\n";
		}

		// 相机内参
		camera.K_.at<double>(0, 0) = camIntr[0];
		camera.K_.at<double>(0, 2) = camIntr[1];
		camera.K_.at<double>(1, 1) = camIntr[2];
		camera.K_.at<double>(1, 2) = camIntr[3];

		// 相机畸变已优化
		// 振镜参数已优化

		// 相机-振镜的位姿
		for (int i = 0; i < multirvec_g2c.size(); i++) {			
			cv::Rodrigues(multirvec_g2c[i], multiZ_g2c[i](cv::Rect(0, 0, 3, 3)));
			multitvec_g2c[i].copyTo(multiZ_g2c[i](cv::Rect(3, 0, 1, 3)));
		}
		
		// 世界坐标系到振镜的位姿
		for (int i = 0; i < multirvec_w2g.size(); i++) {
			cv::Rodrigues(multirvec_w2g[i], multiX_w2g[i](cv::Rect(0, 0, 3, 3)));
			multitvec_w2g[i].copyTo(multiX_w2g[i](cv::Rect(3, 0, 1, 3)));
		}

		// 统计优化后的重投影误差
		dev.clear();
		for (const auto& scan : camera.multiScanCodes_) {
			int count = scan.first;
			auto it = allObjCodes.find(count);
			if (it == allObjCodes.end()) {
				continue;
			}

			auto objCodes = it->second;
			auto viewports = scan.second;
			for (const auto& view : viewports) {
				double galvoX = view.galvoX_ / 100.;
				double galvoY = view.galvoY_ / 100.;
				auto& imgCodes = view.imgCodes_;
				for (const auto& imgCode : imgCodes) {
					int key = imgCode.first;
					auto iter = objCodes.find(key);
					if (iter == objCodes.end()) {
						continue;
					}
					auto points = imgCode.second;
					for (int i = 0; i < points.size(); i++) {
						cv::Point2f imgPoint = points[i];
						cv::Point3f objPoint = iter->second[i];

						CGReprojError cgReproj(objPoint, imgPoint, galvoX, galvoY);
						double errors[2] = { 0.,0. };
						cgReproj(
							reinterpret_cast<double*>(G.data),
							camIntr,
							reinterpret_cast<double*>(camera.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec_w2g[count - 1].data),
							reinterpret_cast<double*>(multirvec_g2c[0].data),
							reinterpret_cast<double*>(multitvec_g2c[0].data),
							errors);

						int id = key * 10 + i;
						vlp::ReprojError err;
						err.scanCount_ = count;
						err.galvoX_ = view.galvoX_;
						err.galvoY_ = view.galvoY_;
						err.point_ = imgPoint;
						err.finalXError_ = errors[0];
						err.finalYError_ = errors[1];
						err.finalError_ = cv::norm(cv::Vec2d(errors[0], errors[1]));
						dev[id].emplace_back(err);
					}
				}
			}
		}

		{
			int count = 0;
			double error = 0.;
			for (const auto& ele : dev) {
				for (const auto& dev : ele.second) {
					error += dev.finalError_;
					count++;
				}
			}
			std::cout << "C-G final: " << error / count << "\n";
		}

		return true;
	}

	bool BA::Optimize(
		const vlp::MultiScanObjCodes& allObjCodes,
		vlp::CameraProperty& leftCam,
		vlp::CameraProperty& rightCam,
		cv::Mat& G1, cv::Mat& G2,
		std::vector<cv::Mat>& multiX1_w2g,
		std::vector<cv::Mat>& multiZ1_g2c,
		std::vector<cv::Mat>& multiT_lg2rg,
		std::vector<cv::Mat>& multiZ2_g2c,
		std::map<int, std::vector<vlp::ReprojError>>& leftDev,
		std::map<int, std::vector<vlp::ReprojError>>& rightDev)
	{
		double leftCamIntr[4] = {
			leftCam.K_.at<double>(0, 0), // fx
			leftCam.K_.at<double>(0, 2), // u0
			leftCam.K_.at<double>(1, 1), // fy
			leftCam.K_.at<double>(1, 2) }; // v0

		double rightCamIntr[4] = {
			rightCam.K_.at<double>(0, 0), // fx
			rightCam.K_.at<double>(0, 2), // u0
			rightCam.K_.at<double>(1, 1), // fy
			rightCam.K_.at<double>(1, 2) }; // v0

		std::vector<cv::Mat> multirvec1_w2g(multiX1_w2g.size());
		std::vector<cv::Mat> multitvec1_w2g(multiX1_w2g.size());
		for (int i = 0; i < multiX1_w2g.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multiX1_w2g[i](cv::Rect(0, 0, 3, 3)), rvec);
			multirvec1_w2g[i] = rvec.clone();
			multiX1_w2g[i](cv::Rect(3, 0, 1, 3)).copyTo(multitvec1_w2g[i]);
		}

		std::vector<cv::Mat> multirvec1_g2c(multiZ1_g2c.size());
		std::vector<cv::Mat> multitvec1_g2c(multiZ1_g2c.size());
		for (int i = 0; i < multiZ1_g2c.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multiZ1_g2c[i](cv::Rect(0, 0, 3, 3)), rvec);
			multirvec1_g2c[i] = rvec.clone();
			multiZ1_g2c[i](cv::Rect(3, 0, 1, 3)).copyTo(multitvec1_g2c[i]);
		}

		cv::Mat zerorvec = cv::Mat::zeros(3, 1, CV_64FC1);
		cv::Mat zerotvec = cv::Mat::zeros(3, 1, CV_64FC1);

		std::vector<cv::Mat> multirvec_lg2rg(multiT_lg2rg.size());
		std::vector<cv::Mat> multitvec_lg2rg(multiT_lg2rg.size());
		for (int i = 0; i < multiT_lg2rg.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multiT_lg2rg[i](cv::Rect(0, 0, 3, 3)), rvec);
			multirvec_lg2rg[i] = rvec.clone();
			multiT_lg2rg[i](cv::Rect(3, 0, 1, 3)).copyTo(multitvec_lg2rg[i]);
		}
		
		std::vector<cv::Mat> multirvec2_g2c(multiZ2_g2c.size());
		std::vector<cv::Mat> multitvec2_g2c(multiZ2_g2c.size());
		for (int i = 0; i < multiZ2_g2c.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multiZ2_g2c[i](cv::Rect(0, 0, 3, 3)), rvec);
			multirvec2_g2c[i] = rvec.clone();
			multiZ2_g2c[i](cv::Rect(3, 0, 1, 3)).copyTo(multitvec2_g2c[i]);
		}

		ceres::Problem problem;		

		std::vector<double> leftErrors, rightErrors;

		auto& leftMultiScan = leftCam.multiScanCodes_;
		for (const auto& scan : leftMultiScan) {
			int count = scan.first;
			auto it = allObjCodes.find(count);
			if (it == allObjCodes.end()) {
				continue;
			}

			auto& objCodes = it->second;
			auto& viewports = scan.second;
			for (auto& view : viewports) {
				double galvoX = view.galvoX_ / 100.;
				double galvoY = view.galvoY_ / 100.;
				auto& imgCodes = view.imgCodes_;
				for (const auto& imgCode : imgCodes) {
					int key = imgCode.first;
					auto iter = objCodes.find(key);
					if (iter == objCodes.end()) {
						continue;
					}
					auto& points = imgCode.second;
					for (int i = 0; i < points.size(); i++) {
						cv::Point2f imgPoint = points[i];
						cv::Point3f objPoint = iter->second[i];

						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
							CGReprojError, 2, 21, 4, 5, 3, 3, 3, 3, 3, 3>(
								new CGReprojError(objPoint, imgPoint, galvoX, galvoY));
						problem.AddResidualBlock(costFunction, NULL,
							reinterpret_cast<double*>(G1.data),
							leftCamIntr,
							reinterpret_cast<double*>(leftCam.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(zerorvec.data),
							reinterpret_cast<double*>(zerotvec.data),
							reinterpret_cast<double*>(multirvec1_g2c[0].data),
							reinterpret_cast<double*>(multitvec1_g2c[0].data));

						CGReprojError cgReproj(objPoint, imgPoint, galvoX, galvoY);
						double errors[2] = { 0.,0. };
						cgReproj(
							reinterpret_cast<double*>(G1.data),
							leftCamIntr,
							reinterpret_cast<double*>(leftCam.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(zerorvec.data),
							reinterpret_cast<double*>(zerotvec.data),
							reinterpret_cast<double*>(multirvec1_g2c[count - 1].data),
							reinterpret_cast<double*>(multitvec1_g2c[count - 1].data),
							errors);
						leftErrors.emplace_back(cv::norm(cv::Vec2d(errors[0], errors[1])));
					}
				}
			}
		}
		problem.SetParameterBlockConstant(
			reinterpret_cast<double*>(zerorvec.data));
		problem.SetParameterBlockConstant(
			reinterpret_cast<double*>(zerotvec.data));
		//problem.SetParameterBlockConstant(leftCamIntr);
		//problem.SetParameterBlockConstant(
		//	reinterpret_cast<double*>(leftCam.distCoeffs_.data));

		auto& rightMultiScan = rightCam.multiScanCodes_;
		for (auto& scan : rightMultiScan) {
			int count = scan.first;
			auto it = allObjCodes.find(count);
			if (it == allObjCodes.end()) {
				continue;
			}

			auto& objCodes = it->second;
			auto viewports = scan.second;
			for (auto& view : viewports) {
				double galvoX = view.galvoX_ / 100.;
				double galvoY = view.galvoY_ / 100.;
				auto& imgCodes = view.imgCodes_;
				for (const auto& imgCode : imgCodes) {
					int key = imgCode.first;
					auto iter = objCodes.find(key);
					if (iter == objCodes.end()) {
						continue;
					}
					auto points = imgCode.second;
					for (int i = 0; i < points.size(); i++) {
						cv::Point2f imgPoint = points[i];
						cv::Point3f objPoint = iter->second[i];

						ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<
							CGReprojError, 2, 21, 4, 5, 3, 3, 3, 3, 3, 3>(
								new CGReprojError(objPoint, imgPoint, galvoX, galvoY));
						problem.AddResidualBlock(costFunction, NULL,
							reinterpret_cast<double*>(G2.data),
							rightCamIntr,
							reinterpret_cast<double*>(rightCam.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multirvec_lg2rg[0].data),
							reinterpret_cast<double*>(multitvec_lg2rg[0].data),
							reinterpret_cast<double*>(multirvec2_g2c[0].data),
							reinterpret_cast<double*>(multitvec2_g2c[0].data));

						CGReprojError cgReproj(objPoint, imgPoint, galvoX, galvoY);
						double errors[2] = { 0.,0. };
						cgReproj(
							reinterpret_cast<double*>(G2.data),
							rightCamIntr,
							reinterpret_cast<double*>(rightCam.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multirvec_lg2rg[count - 1].data),
							reinterpret_cast<double*>(multitvec_lg2rg[count - 1].data),
							reinterpret_cast<double*>(multirvec2_g2c[count - 1].data),
							reinterpret_cast<double*>(multitvec2_g2c[count - 1].data),
							errors);
						rightErrors.emplace_back(cv::norm(cv::Vec2d(errors[0], errors[1])));
					}
				}
			}
		}
		//problem.SetParameterBlockConstant(rightCamIntr);
		//problem.SetParameterBlockConstant(
		//	reinterpret_cast<double*>(rightCam.distCoeffs_.data));

		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.num_threads = 4;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		//options.function_tolerance /= 100;
		//options.gradient_tolerance /= 100;
		options.minimizer_progress_to_stdout = true;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";

		if (!(summary.termination_type == ceres::CONVERGENCE)) {
			std::cout << "Bundle Adjustment failed.\n";
			return false;
		}
		std::cout << " Bundle Adjustment statistics (approximated RMSE):\n"
			<< " residuals:   " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost * 2 / summary.num_residuals) << "\n"
			<< " Final RMSE:   " << std::sqrt(summary.final_cost * 2 / summary.num_residuals) << "\n"
			<< " Time (s):     " << summary.total_time_in_seconds << "\n";

		{
			double mean = std::accumulate(leftErrors.begin(), leftErrors.end(), 0.);
			mean /= leftErrors.size();
			double maxVal = *std::max_element(leftErrors.begin(), leftErrors.end());
			double minVal = *std::min_element(leftErrors.begin(), leftErrors.end());
			std::cout << "C-G1 init: " << mean << "\t" << maxVal << "\t" << minVal << "\n";

			mean = std::accumulate(rightErrors.begin(), rightErrors.end(), 0.);
			mean /= rightErrors.size();
			maxVal = *std::max_element(rightErrors.begin(), rightErrors.end());
			minVal = *std::min_element(rightErrors.begin(), rightErrors.end());
			std::cout << "C-G2 init: " << mean << "\t" << maxVal << "\t" << minVal << "\n";
		}
		
		// 相机内参
		leftCam.K_.at<double>(0, 0) = leftCamIntr[0];
		leftCam.K_.at<double>(0, 2) = leftCamIntr[1];
		leftCam.K_.at<double>(1, 1) = leftCamIntr[2];
		leftCam.K_.at<double>(1, 2) = leftCamIntr[3];

		rightCam.K_.at<double>(0, 0) = rightCamIntr[0];
		rightCam.K_.at<double>(0, 2) = rightCamIntr[1];
		rightCam.K_.at<double>(1, 1) = rightCamIntr[2];
		rightCam.K_.at<double>(1, 2) = rightCamIntr[3];

		// 相机畸变已优化
		// 左右振镜参数已优化

		// 左相机-振镜的位姿
		for (int i = 0; i < multirvec1_g2c.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multirvec1_g2c[i], multiZ1_g2c[i](cv::Rect(0, 0, 3, 3)));
			multitvec1_g2c[i].copyTo(multiZ1_g2c[i](cv::Rect(3, 0, 1, 3)));
		}
		
		// 右相机-振镜的位姿
		for (int i = 0; i < multirvec2_g2c.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multirvec2_g2c[i], multiZ2_g2c[i](cv::Rect(0, 0, 3, 3)));
			multitvec2_g2c[i].copyTo(multiZ2_g2c[i](cv::Rect(3, 0, 1, 3)));
		}

		// 左右振镜的位姿
		for (int i = 0; i < multirvec_lg2rg.size(); i++) {
			cv::Mat rvec;
			cv::Rodrigues(multirvec_lg2rg[i], multiT_lg2rg[i](cv::Rect(0, 0, 3, 3)));
			multitvec_lg2rg[i].copyTo(multiT_lg2rg[i](cv::Rect(3, 0, 1, 3)));
		}

		// 世界坐标系到左振镜的位姿
		for (int i = 0; i < multirvec1_w2g.size(); i++) {
			cv::Mat rvec1;
			cv::Rodrigues(multirvec1_w2g[i], multiX1_w2g[i](cv::Rect(0, 0, 3, 3)));
			multitvec1_w2g[i].copyTo(multiX1_w2g[i](cv::Rect(3, 0, 1, 3)));
		}

		// 统计优化后的重投影误差
		leftDev.clear();
		rightDev.clear();
		for (const auto& scan : leftMultiScan) {
			int count = scan.first;
			auto it = allObjCodes.find(count);
			if (it == allObjCodes.end()) {
				continue;
			}

			auto& objCodes = it->second;
			auto& viewports = scan.second;
			for (auto& view : viewports) {
				double galvoX = view.galvoX_ / 100.;
				double galvoY = view.galvoY_ / 100.;
				auto& imgCodes = view.imgCodes_;
				for (const auto& imgCode : imgCodes) {
					int key = imgCode.first;
					auto iter = objCodes.find(key);
					if (iter == objCodes.end()) {
						continue;
					}
					auto& points = imgCode.second;
					for (int i = 0; i < points.size(); i++) {
						cv::Point2f imgPoint = points[i];
						cv::Point3f objPoint = iter->second[i];						

						CGReprojError cgReproj(objPoint, imgPoint, galvoX, galvoY);
						double errors[2] = { 0.,0. };
						cgReproj(
							reinterpret_cast<double*>(G1.data),
							leftCamIntr,
							reinterpret_cast<double*>(leftCam.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(zerorvec.data),
							reinterpret_cast<double*>(zerotvec.data),
							reinterpret_cast<double*>(multirvec1_g2c[0].data),
							reinterpret_cast<double*>(multitvec1_g2c[0].data),
							errors);

						int id = key * 10 + i;
						vlp::ReprojError err;
						err.scanCount_ = count;
						err.galvoX_ = view.galvoX_;
						err.galvoY_ = view.galvoY_;
						err.point_ = imgPoint;
						err.finalXError_ = errors[0];
						err.finalYError_ = errors[1];
						err.finalError_ = cv::norm(cv::Vec2d(errors[0], errors[1]));
						leftDev[id].emplace_back(err);
					}
				}
			}
		}

		for (auto& scan : rightMultiScan) {
			int count = scan.first;
			auto it = allObjCodes.find(count);
			if (it == allObjCodes.end()) {
				continue;
			}

			auto& objCodes = it->second;
			auto viewports = scan.second;
			for (auto& view : viewports) {
				double galvoX = view.galvoX_ / 100.;
				double galvoY = view.galvoY_ / 100.;
				auto& imgCodes = view.imgCodes_;
				for (const auto& imgCode : imgCodes) {
					int key = imgCode.first;
					auto iter = objCodes.find(key);
					if (iter == objCodes.end()) {
						continue;
					}
					auto points = imgCode.second;
					for (int i = 0; i < points.size(); i++) {
						cv::Point2f imgPoint = points[i];
						cv::Point3f objPoint = iter->second[i];			

						CGReprojError cgReproj(objPoint, imgPoint, galvoX, galvoY);
						double errors[2] = { 0.,0. };
						cgReproj(
							reinterpret_cast<double*>(G2.data),
							rightCamIntr,
							reinterpret_cast<double*>(rightCam.distCoeffs_.data),
							reinterpret_cast<double*>(multirvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multitvec1_w2g[count - 1].data),
							reinterpret_cast<double*>(multirvec_lg2rg[0].data),
							reinterpret_cast<double*>(multitvec_lg2rg[0].data),
							reinterpret_cast<double*>(multirvec2_g2c[0].data),
							reinterpret_cast<double*>(multitvec2_g2c[0].data),
							errors);
						
						int id = key * 10 + i;
						vlp::ReprojError err;
						err.scanCount_ = count;
						err.galvoX_ = view.galvoX_;
						err.galvoY_ = view.galvoY_;
						err.point_ = imgPoint;
						err.finalXError_ = errors[0];
						err.finalYError_ = errors[1];
						err.finalError_ = cv::norm(cv::Vec2d(errors[0], errors[1]));
						rightDev[id].emplace_back(err);
					}
				}
			}
		}

		{
			int count = 0;
			double error = 0.;
			for (const auto& ele : leftDev) {
				for (const auto& dev : ele.second) {
					error += dev.finalError_;
					count++;
				}
			}
			std::cout << "C-G1 final: " << error / count << "\n";

			count = 0;
			error = 0.;
			for (const auto& ele : rightDev) {
				for (const auto& dev : ele.second) {
					error += dev.finalError_;
					count++;
				}
			}
			std::cout << "C-G2 final: " << error / count << "\n";
		}

		return true;
	}

}// namespace vlp 