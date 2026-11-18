#pragma once
#include "DefinedDS.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vlp {
	struct CGReprojError
	{
		//待估计的变量：location[2]
		CGReprojError(const cv::Point3f& objPoint,
			const cv::Point2f& observedPoint,
			double galvoX, double galvoY) :
			objPoint_(objPoint),
			observedPoint_(observedPoint),
			galvoX_(galvoX),
			galvoY_(galvoY) {}

		template <typename T>
		bool operator()(const T* G,
			const T* intrParas, const T* distCoeffs,
			const T* rvec_w2g, const T* t_w2g,
			const T* rvec, const T* tvec,
			const T* rvec_g2c, const T* t_g2c, T* residuals) const
		{
			T location[3] = { T(objPoint_.x),T(objPoint_.y),T(objPoint_.z) };

			// world->galvo
			T loc0[3];
			//对空间点进行旋转和平移操作
			ceres::AngleAxisRotatePoint(rvec_w2g, location, loc0);
			loc0[0] += t_w2g[0];
			loc0[1] += t_w2g[1];
			loc0[2] += t_w2g[2];

			T point0[3];
			ceres::AngleAxisRotatePoint(rvec, loc0, point0);
			point0[0] += tvec[0];
			point0[1] += tvec[1];
			point0[2] += tvec[2];

			const T& alpha0 = T(G[0]);
			const T& beta0 = T(G[1]);
			const T& e = T(G[2]);
			const T& ka1 = T(G[3]);
			const T& ka2 = T(G[4]);
			const T& ka3 = T(G[5]);
			const T& kb1 = T(G[6]);
			const T& kb2 = T(G[7]);
			const T& kb3 = T(G[8]);
			const T& ka4 = T(G[9]);
			const T& ka5 = T(G[10]);
			const T& ka6 = T(G[11]);
			const T& ka7 = T(G[12]);
			const T& ka8 = T(G[13]);
			const T& ka9 = T(G[14]);
			const T& kb4 = T(G[15]);
			const T& kb5 = T(G[16]);
			const T& kb6 = T(G[17]);
			const T& kb7 = T(G[18]);
			const T& kb8 = T(G[19]);
			const T& kb9 = T(G[20]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_alpha =
				alpha0 +
				ka1 * a +
				ka2 * a * a +
				ka3 * a * a * a +
				ka4 * b +
				ka5 * b * b +
				ka6 * b * b * b +
				ka7 * a * b +
				ka8 * a * a * b +
				ka9 * a * b * b;
			T alpha = delta_alpha / T(180.0 / CV_PI);

			T delta_beta =
				beta0 +
				kb1 * b +
				kb2 * b * b +
				kb3 * b * b * b +
				kb4 * a +
				kb5 * a * a +
				kb6 * a * a * a +
				kb7 * a * b +
				kb8 * a * b * b +
				kb9 * a * a * b;
			T beta = delta_beta / T(180.0 / CV_PI);

			//mirror-2
			T sinBeta = ceres::sin(beta);
			T cosBeta = ceres::cos(beta);

			T point1[3];
			point1[0] = point0[0];

			point1[1] = point0[1] * (T(1.0) - T(2.0) * sinBeta * sinBeta) +
				point0[2] * (T(-2.0) * cosBeta * sinBeta);

			point1[2] = point0[1] * (T(-2.0) * cosBeta * sinBeta) +
				point0[2] * (T(1.0) - T(2.0) * cosBeta * cosBeta);

			//mirror-1
			T sinAlpha = ceres::sin(alpha);
			T cosAlpha = ceres::cos(alpha);

			T point2[3];
			point2[0] = point1[0] * (T(1.0) - T(2.0) * cosAlpha * cosAlpha) +
				point1[1] * (T(-2.0) * cosAlpha * sinAlpha) +
				T(2.0) * e * cosAlpha * sinAlpha;

			point2[1] = point1[0] * (T(-2.0) * cosAlpha * sinAlpha) +
				point1[1] * (T(1.0) - T(2.0) * sinAlpha * sinAlpha) +
				T(2.0) * e * sinAlpha * sinAlpha;

			point2[2] = point1[2];

			//galvo->camera
			T point3[3];

			ceres::AngleAxisRotatePoint(rvec_g2c, point2, point3);
			point3[0] += t_g2c[0];
			point3[1] += t_g2c[1];
			point3[2] += t_g2c[2];

			//转到归一化平面
			T x = point3[0] / point3[2];
			T y = point3[1] / point3[2];

			//径向和切向畸变系数的输入
			const T& k1 = T(distCoeffs[0]);
			const T& k2 = T(distCoeffs[1]);
			const T& p1 = T(distCoeffs[2]);
			const T& p2 = T(distCoeffs[3]);
			const T& k3 = T(distCoeffs[4]);

			T r2 = x * x + y * y;
			T distortion_x = x * (T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
				T(2.0) * p1 * x * y + p2 * (r2 + T(2.0) * x * x);
			T distortion_y = y * (T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
				p1 * (r2 + T(2.0) * y * y) + T(2.0) * p2 * x * y;

			const T& fx = T(intrParas[0]);
			const T& u0 = T(intrParas[1]);
			const T& fy = T(intrParas[2]);
			const T& v0 = T(intrParas[3]);

			//转换到图像平面
			T predicted_x = fx * distortion_x + u0;
			T predicted_y = fy * distortion_y + v0;

			residuals[0] = predicted_x - T(observedPoint_.x);
			residuals[1] = predicted_y - T(observedPoint_.y);
			return true;
		}

		template <typename T>
		bool operator()(const T* G,
			const T* intrParas, const T* distCoeffs,
			const T* rvec_w2g, const T* t_w2g,
			const T* rvec_g2c, const T* t_g2c, T* residuals) const
		{
			T location[3] = { T(objPoint_.x),T(objPoint_.y),T(objPoint_.z) };

			// world->galvo
			T point0[3];
			//对空间点进行旋转和平移操作
			ceres::AngleAxisRotatePoint(rvec_w2g, location, point0);
			point0[0] += t_w2g[0];
			point0[1] += t_w2g[1];
			point0[2] += t_w2g[2];			

			const T& alpha0 = T(G[0]);
			const T& beta0 = T(G[1]);
			//const T& e = T(G[2]);
			const T& e = T(42.5);
			const T& ka1 = T(G[3]);
			const T& ka2 = T(G[4]);
			const T& ka3 = T(G[5]);
			const T& kb1 = T(G[6]);
			const T& kb2 = T(G[7]);
			const T& kb3 = T(G[8]);
			const T& ka4 = T(G[9]);
			const T& ka5 = T(G[10]);
			const T& ka6 = T(G[11]);
			const T& ka7 = T(G[12]);
			const T& ka8 = T(G[13]);
			const T& ka9 = T(G[14]);
			const T& kb4 = T(G[15]);
			const T& kb5 = T(G[16]);
			const T& kb6 = T(G[17]);
			const T& kb7 = T(G[18]);
			const T& kb8 = T(G[19]);
			const T& kb9 = T(G[20]);

			T a = T(galvoX_);
			T b = T(galvoY_);

			T delta_alpha =				
				ka1 * a +
				ka2 * a * a +
				ka3 * a * a * a +
				ka4 * b +
				ka5 * b * b +
				ka6 * b * b * b /*+
				ka7 * a * b +
				ka8 * a * a * b +
				ka9 * a * b * b*/;
			T alpha = (delta_alpha + alpha0) / T(180.0 / CV_PI);

			T delta_beta =				
				kb1 * b +
				kb2 * b * b +
				kb3 * b * b * b +
				kb4 * a +
				kb5 * a * a +
				kb6 * a * a * a /*+
				kb7 * a * b +
				kb8 * a * b * b +
				kb9 * a * a * b*/;
			T beta = (delta_beta + beta0) / T(180.0 / CV_PI);

			//mirror-2

			//T sinBeta = ceres::sin(beta);
			//T cosBeta = ceres::cos(beta);

			T point1[3];

			//point1[0] = point0[0];

			//point1[1] = point0[1] * (T(1.0) - T(2.0) * sinBeta * sinBeta) +
			//	point0[2] * (T(-2.0) * cosBeta * sinBeta);

			//point1[2] = point0[1] * (T(-2.0) * cosBeta * sinBeta) +
			//	point0[2] * (T(1.0) - T(2.0) * cosBeta * cosBeta);

			T n2[4] = { T(0), ceres::sin(beta), ceres::cos(beta), T(0) };

			point1[0] = point0[0] * (T(1.0) - T(2.0) * n2[0] * n2[0]) +
				point0[1] * (T(-2.0) * n2[0] * n2[1]) +
				point0[2] * (T(-2.0) * n2[0] * n2[2]) +
				(T(-2.0) * n2[0] * n2[3]);

			point1[1] = point0[0] * (T(-2.0) * n2[0] * n2[1]) +
				point0[1] * (T(1.0) - T(2.0) * n2[1] * n2[1]) +
				point0[2] * (T(-2.0) * n2[1] * n2[2]) +
				(T(-2.0) * n2[1] * n2[3]);

			point1[2] = point0[0] * (T(-2.0) * n2[0] * n2[2]) +
				point0[1] * (T(-2.0) * n2[1] * n2[2]) +
				point0[2] * (T(1.0) - T(2.0) * n2[2] * n2[2]) +
				(T(-2.0) * n2[2] * n2[3]);

			//mirror-1

			//T sinAlpha = ceres::sin(alpha);
			//T cosAlpha = ceres::cos(alpha);

			T point2[3];
			//point2[0] = point1[0] * (T(1.0) - T(2.0) * cosAlpha * cosAlpha) +
			//	point1[1] * (T(-2.0) * cosAlpha * sinAlpha) +
			//	T(2.0) * e * cosAlpha * sinAlpha;

			//point2[1] = point1[0] * (T(-2.0) * cosAlpha * sinAlpha) +
			//	point1[1] * (T(1.0) - T(2.0) * sinAlpha * sinAlpha) +
			//	T(2.0) * e * sinAlpha * sinAlpha;

			//point2[2] = point1[2];

			T n1[4] = { ceres::cos(alpha), ceres::sin(alpha), T(0), ceres::sin(alpha) * T(-1.0) * e };

			point2[0] = point1[0] * (T(1.0) - T(2.0) * n1[0] * n1[0]) +
				point1[1] * (T(-2.0) * n1[0] * n1[1]) +
				point1[2] * (T(-2.0) * n1[0] * n1[2]) +
				(T(-2.0) * n1[0] * n1[3]);

			point2[1] = point1[0] * (T(-2.0) * n1[0] * n1[1]) +
				point1[1] * (T(1.0) - T(2.0) * n1[1] * n1[1]) +
				point1[2] * (T(-2.0) * n1[1] * n1[2]) +
				(T(-2.0) * n1[1] * n1[3]);

			point2[2] = point1[0] * (T(-2.0) * n1[0] * n1[2]) +
				point1[1] * (T(-2.0) * n1[1] * n1[2]) +
				point1[2] * (T(1.0) - T(2.0) * n1[2] * n1[2]) +
				(T(-2.0) * n1[2] * n1[3]);

			//galvo->camera
			T point3[3];

			ceres::AngleAxisRotatePoint(rvec_g2c, point2, point3);
			point3[0] += t_g2c[0];
			point3[1] += t_g2c[1];
			point3[2] += t_g2c[2];

			//转到归一化平面
			T x = point3[0] / point3[2];
			T y = point3[1] / point3[2];

			//径向和切向畸变系数的输入
			const T& k1 = T(distCoeffs[0]);
			const T& k2 = T(distCoeffs[1]);
			const T& p1 = T(distCoeffs[2]);
			const T& p2 = T(distCoeffs[3]);
			const T& k3 = T(distCoeffs[4]);

			T r2 = x * x + y * y;
			T distortion_x = x * (T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
				T(2.0) * p1 * x * y + p2 * (r2 + T(2.0) * x * x);
			T distortion_y = y * (T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
				p1 * (r2 + T(2.0) * y * y) + T(2.0) * p2 * x * y;

			const T& fx = T(intrParas[0]);
			const T& u0 = T(intrParas[1]);
			const T& fy = T(intrParas[2]);
			const T& v0 = T(intrParas[3]);

			//转换到图像平面
			T predicted_x = fx * distortion_x + u0;
			T predicted_y = fy * distortion_y + v0;

			residuals[0] = predicted_x - T(observedPoint_.x);
			residuals[1] = predicted_y - T(observedPoint_.y);
			return true;
		}

		double galvoX_, galvoY_;
		cv::Point3f objPoint_;
		cv::Point2f observedPoint_;
	};

	class BA
	{
	public:
		BA() = default;
		~BA() = default;

		static bool Optimize(
			const vlp::MultiScanObjCodes& allObjCodes,
			vlp::CameraProperty& leftCam, vlp::CameraProperty& rightCam,
			cv::Mat& G1, cv::Mat& G2, std::vector<cv::Mat>& multiX1_w2g,
			std::vector<cv::Mat>& multiZ1_g2c,
			std::vector<cv::Mat>& multiT_lg2rg,
			std::vector<cv::Mat>& multiZ2_g2c,
			std::map<int, std::vector<vlp::ReprojError>>& leftDev,
			std::map<int, std::vector<vlp::ReprojError>>& rightDev);

		static bool Optimize(
			const vlp::MultiScanObjCodes& allObjCodes,
			vlp::CameraProperty& camera, cv::Mat& G,
			std::vector<cv::Mat>& multiX_w2g,
			std::vector<cv::Mat>& multiZ_g2c,
			std::map<int, std::vector<vlp::ReprojError>>& dev);
	};

}// namespace vlp 
