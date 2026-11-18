#include "IO.h"

namespace vlp {
	bool CheckFileSuffix(const std::string& filePath, const std::string& suffix)
	{
		std::size_t found = filePath.find_last_of(".");
		if (std::string::npos == found) {
			return false;
		}
		std::string fileNameSuffix = filePath.substr(found + 1U);
		if (suffix == fileNameSuffix) {
			return true;
		}
		return false;
	}

	bool LoadCGNames(const std::string& filePath,
		std::vector<std::string>& cameraNames,
		std::vector<std::string>& galvoNames)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}
		cv::FileNode node = in["feature"];
		cameraNames.clear();
		galvoNames.clear();
		for (auto it = node.begin(); it != node.end(); ++it) {
			std::string camera;
			int galvo;
			(*it)["camera"] >> camera;
			(*it)["galvo"] >> galvo;
			cameraNames.emplace_back(camera);
			galvoNames.emplace_back(std::to_string(galvo));
		}
		in.release();
		return true;
	}

	bool LoadCamParameters(const std::string& filePath,
		vlp::CameraProperty& leftCam, vlp::CameraProperty& rightCam)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}
		
		in["K_0"] >> leftCam.K_;
		in["distCoeffs_0"] >> leftCam.distCoeffs_;
		in["K_1"] >> rightCam.K_;
		in["distCoeffs_1"] >> rightCam.distCoeffs_;
		in.release();
		return true;
	}

	bool LoadCamParametersFromYml(const std::string& filePath,
		vlp::CameraProperty& camera)
	{
		const std::string suffix = "yml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}

		in["_intrinsicL"] >> camera.K_;
		in["_distortionL"] >> camera.distCoeffs_;		
		in.release();
		return true;
	}

	bool LoadGalvoParameters(const std::string& filePath, cv::Mat& G)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}

		std::map<std::string, double> data;
		cv::FileNode node = in["galvo"];
		for (const auto& item : node) {
			std::string key = item.name();
			double value = (double)item;
			data[key] = value;
		}

		G = cv::Mat::zeros(21, 1, CV_64FC1);
		G.at<double>(0, 0) = data["alpha"];
		G.at<double>(1, 0) = data["beta"];
		G.at<double>(2, 0) = data["e"];

		G.at<double>(3, 0) = data["ka1"];
		G.at<double>(4, 0) = data["ka2"];
		G.at<double>(5, 0) = data["ka3"];

		G.at<double>(6, 0) = data["kb1"];
		G.at<double>(7, 0) = data["kb2"];
		G.at<double>(8, 0) = data["kb3"];

		G.at<double>(9, 0) = data["ka4"];
		G.at<double>(10, 0) = data["ka5"];
		G.at<double>(11, 0) = data["ka6"];
		G.at<double>(12, 0) = data["ka7"];
		G.at<double>(13, 0) = data["ka8"];
		G.at<double>(14, 0) = data["ka9"];

		G.at<double>(15, 0) = data["kb4"];
		G.at<double>(16, 0) = data["kb5"];
		G.at<double>(17, 0) = data["kb6"];
		G.at<double>(18, 0) = data["kb7"];
		G.at<double>(19, 0) = data["kb8"];
		G.at<double>(20, 0) = data["kb9"];

		in.release();
		return true;
	}

	bool LoadImgCodes(const std::string& filePath,
		std::map<int, std::vector<CGCodeBlock>>& leftData,
		std::map<int, std::vector<CGCodeBlock>>& rightData)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}
		leftData.clear();
		cv::FileNode leftNode = in["CAMERA0"];
		for (const auto& node : leftNode) {
			std::string scan = node.name();
			int scanCount = std::stoi(scan.substr(5));

			std::vector<CGCodeBlock> blocks;
			for (const auto& view : node) {
				CGCodeBlock block;
				block.scanCount_ = (int)view["scan"];
				block.frame_ = (int)view["frame"];
				block.galvoX_ = (int)view["galvoX"];
				block.galvoY_ = (int)view["galvoY"];

				cv::FileNode imgCodesNode = view["imgCodes"];
				for (const auto& code : imgCodesNode) {
					std::string codeName = code.name();
					int key = std::stoi(codeName.substr(5));

					std::vector<cv::Point2f> points;
					for (const auto& pointNode : code) {
						float x = (float)pointNode["x"];
						float y = (float)pointNode["y"];
						points.emplace_back(x, y);
					}
					block.imgCodes_[key] = points;					
				}
				blocks.emplace_back(block);
			}
			leftData[scanCount] = blocks;
		}

		rightData.clear();
		cv::FileNode rightNode = in["CAMERA1"];
		for (const auto& node : rightNode) {
			std::string scan = node.name();
			int scanCount = std::stoi(scan.substr(5));

			std::vector<CGCodeBlock> blocks;
			for (const auto& view : node) {
				CGCodeBlock block;
				block.scanCount_ = (int)view["scan"];
				block.frame_ = (int)view["frame"];
				block.galvoX_ = (int)view["galvoX"];
				block.galvoY_ = (int)view["galvoY"];

				cv::FileNode imgCodesNode = view["imgCodes"];
				for (const auto& code : imgCodesNode) {
					std::string codeName = code.name();
					int key = std::stoi(codeName.substr(5));

					std::vector<cv::Point2f> points;
					for (const auto& pointNode : code) {
						float x = (float)pointNode["x"];
						float y = (float)pointNode["y"];
						points.emplace_back(x, y);
					}
					block.imgCodes_[key] = points;
				}
				blocks.emplace_back(block);
			}
			rightData[scanCount] = blocks;
		}

		in.release();
		return true;
	}

	bool LoadImgCodesFromXml(const std::string& filePath,
		std::map<int, std::vector<CGCodeBlock>>& data)
	{
		const std::string suffix = "xml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		data.clear();

		tinyxml2::XMLDocument doc;
		if (doc.LoadFile(filePath.c_str()) != 0)
		{
			doc.PrintError();
			exit(1);
		}
		tinyxml2::XMLElement* root = doc.RootElement();
		for (tinyxml2::XMLElement* viewportnode = root->FirstChildElement("viewport");
			viewportnode != nullptr; viewportnode = viewportnode->NextSiblingElement())
		{
			std::string digitalqstr(viewportnode->Attribute("digital"));
			size_t commaPos = digitalqstr.find(',');
			if (commaPos == std::string::npos) {
				continue;
			}
			std::string firstPart = digitalqstr.substr(0, commaPos);
			std::string	secondPart = digitalqstr.substr(commaPos + 1);

			CGCodeBlock block;
			block.galvoX_ = std::stoi(firstPart) * 100;
			block.galvoY_ = std::stoi(secondPart) * 100;

			std::map<int, std::vector<cv::Point2f>> codemarkers;
			for (tinyxml2::XMLElement* blocknode = viewportnode->FirstChildElement("block");
				blocknode != nullptr; blocknode = blocknode->NextSiblingElement())
			{
				int id = blocknode->IntAttribute("id");
				std::vector<cv::Point2f> temppts;
				for (tinyxml2::XMLElement* ptnode = blocknode->FirstChildElement("pt");
					ptnode != nullptr; ptnode = ptnode->NextSiblingElement())
				{
					float x = ptnode->FloatAttribute("x");
					float y = ptnode->FloatAttribute("y");
					temppts.emplace_back(x, y);
				}
				codemarkers.emplace(id, temppts);
			}
			block.imgCodes_ = codemarkers;
			data[1].emplace_back(block);
		}
		return true;
	}

	bool WriteImgCodes(const std::string& filePath,
		const std::map<int, std::vector<CGCodeBlock>>& leftData,
		const std::map<int, std::vector<CGCodeBlock>>& rightData)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		cv::FileStorage fs(filePath, cv::FileStorage::WRITE);

		fs << "CAMERA0" << "{";
		for (const auto& scan : leftData) {
			fs << "SCAN_" + std::to_string(scan.first) << "["; // vector<CGCodeBlock> start
			auto& viewports = scan.second;
			for (size_t i = 0; i < viewports.size(); i++) {
				fs << "{";
				fs << "scan" << viewports[i].scanCount_;
				fs << "frame" << viewports[i].frame_;
				fs << "galvoX" << viewports[i].galvoX_;
				fs << "galvoY" << viewports[i].galvoY_;

				auto& codes = viewports[i].imgCodes_;
				fs << "imgCodes" << "{"; // codes start
				for (const auto& code : codes) {
					fs << "CODE_" + std::to_string(code.first) << "[";
					for (const auto& point : code.second) {
						fs << "{:";
						fs << "x" << point.x;
						fs << "y" << point.y;
						fs << "}";
					}
					fs << "]";
				}
				fs << "}";// codes end
				fs << "}";
			}
			fs << "]"; // vector<CGCodeBlock> end
		}
		fs << "}";

		fs << "CAMERA1" << "{";
		for (const auto& scan : rightData) {
			fs << "SCAN_" + std::to_string(scan.first) << "[";
			auto& viewports = scan.second;
			for (size_t i = 0; i < viewports.size(); i++) {
				fs << "{";
				fs << "scan" << viewports[i].scanCount_;
				fs << "frame" << viewports[i].frame_;
				fs << "galvoX" << viewports[i].galvoX_;
				fs << "galvoY" << viewports[i].galvoY_;

				auto& codes = viewports[i].imgCodes_;
				fs << "imgCodes" << "{";
				for (const auto& code : codes) {
					fs << "CODE_" + std::to_string(code.first) << "[";
					for (const auto& point : code.second) {
						fs << "{:";
						fs << "x" << point.x;
						fs << "y" << point.y;
						fs << "}";
					}
					fs << "]";
				}
				fs << "}";
				fs << "}";
			}
			fs << "]";
		}
		fs << "}";

		fs.release();
		return true;
	}

	bool LoadObjCodes(const std::string& filePath,
		std::map<int, std::vector<cv::Point3f>>& objCodes)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}

		objCodes.clear();
		for (const auto& node : in.root()) {
			std::string str = node.name();
			if (str.find("CODE_") == 0) {
				int key = std::stoi(str.substr(5));
				std::vector<cv::Point3f> points;
				for (const auto& point : node) {
					float x = (float)point["x"];
					float y = (float)point["y"];
					float z = (float)point["z"];
					points.emplace_back(x, y, z);
				}
				objCodes[key] = points;
			}
		}
		in.release();
		return true;
	}

	bool LoadObjCodesFromXml(const std::string& filePath,
		std::map<int, std::vector<cv::Point3f>>& objCodes)
	{
		const std::string suffix = "xml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		tinyxml2::XMLDocument doc;
		if (doc.LoadFile(filePath.c_str()) != 0)
		{
			doc.PrintError();
			exit(1);
		}

		objCodes.clear();
		tinyxml2::XMLElement* root = doc.RootElement();
		for (tinyxml2::XMLElement* blocknode = root->FirstChildElement("block");
			blocknode != nullptr; blocknode = blocknode->NextSiblingElement())
		{
			int id = blocknode->IntAttribute("id");

			std::vector<cv::Point3f> temppts;
			for (tinyxml2::XMLElement* ptnode = blocknode->FirstChildElement("pt");
				ptnode != nullptr; ptnode = ptnode->NextSiblingElement())
			{
				float x = ptnode->FloatAttribute("x");
				float y = ptnode->FloatAttribute("y");
				float z = ptnode->FloatAttribute("z");
				temppts.emplace_back(x, y, z);
			}
			objCodes.emplace(id, temppts);
		}

		return true;
	}

	bool WriteCGParameters(const std::string& filePath,
		const cv::Mat& G1, const cv::Mat& K1,
		const cv::Mat& distCoeffs1, const cv::Mat& T1_g2c,
		const cv::Mat& G2, const cv::Mat& K2,
		const cv::Mat& distCoeffs2, const cv::Mat& T2_g2c,
		const cv::Mat& T_lg2rg)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
		fs << "G1" << G1;
		fs << "K1" << K1;
		fs << "distCoeffs1" << distCoeffs1;
		fs << "T1_g2c" << T1_g2c;

		fs << "G2" << G2;
		fs << "K2" << K2;
		fs << "distCoeffs2" << distCoeffs2;
		fs << "T2_g2c" << T2_g2c;

		fs << "T_lg2rg" << T_lg2rg;

		fs.release();
		return true;
	}

	bool LoadCGParameters(const std::string& filePath,
		cv::Mat& G1, cv::Mat& K1,
		cv::Mat& distCoeffs1, cv::Mat& T1_g2c,
		cv::Mat& G2, cv::Mat& K2,
		cv::Mat& distCoeffs2, cv::Mat& T2_g2c,
		cv::Mat& T_lg2rg)
	{
		const std::string suffix = "yaml";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		cv::FileStorage in;
		in.open(filePath, cv::FileStorage::READ);
		if (!in.isOpened()) {
			return false;
		}

		in["G1"] >> G1;
		in["K1"] >> K1;
		in["distCoeffs1"] >> distCoeffs1;
		in["T1_g2c"] >> T1_g2c;

		in["G2"] >> G2;
		in["K2"] >> K2;
		in["distCoeffs2"] >> distCoeffs2;
		in["T2_g2c"] >> T2_g2c;

		in["T_lg2rg"] >> T_lg2rg;

		in.release();
		return true;
	}

	bool WriteReprojError(const std::string& filePath,
		const std::map<int, std::vector<vlp::ReprojError>>& dev)
	{
		const std::string suffix = "txt";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		std::ofstream out(filePath);
		out << std::fixed << std::setprecision(5);
		for (const auto& ele : dev) {
			auto& reproj = ele.second;
			for (const auto& err : reproj) {
				out << err.point_.x << "	"
					<< err.point_.y << "	"
					<< err.finalXError_ << "	"
					<< err.finalYError_ << "	"
					<< err.finalError_ << "\n";
			}
		}
		out.close();
		return true;
	}

	bool WritePnPRt(const std::string& filePath,
		const std::map<int, std::vector<CGCodeBlock>>& multiScanCodes)
	{
		const std::string suffix = "txt";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		std::ofstream out(filePath);
		out << std::fixed << std::setprecision(5);
		for (const auto& scan : multiScanCodes) {
			int count = scan.first;
			auto& viewports = scan.second;
			for (const auto& view : viewports) {
				out << count << " "
					<< view.galvoX_ << " "
					<< view.galvoY_ << " "
					<< view.rvec_.at<double>(0, 0) << " "
					<< view.rvec_.at<double>(1, 0) << " "
					<< view.rvec_.at<double>(2, 0) << " "
					<< view.tvec_.at<double>(0, 0) << " "
					<< view.tvec_.at<double>(1, 0) << " "
					<< view.tvec_.at<double>(2, 0) << "\n";
			}
		}
		out.close();
		return true;
	}

	void LoadPattern(const std::string& folderPath,
		std::vector<cv::Mat>& patterns)
	{
		patterns.clear();
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_regular_file()) {
				std::string filePath = entry.path().string();
				cv::Mat image = cv::imread(filePath);
				if (!image.empty()) {
					patterns.emplace_back(image);
				}
			}
		}
	}

	void LoadFeatureImgs(const std::string& folderPath,
		std::vector<cv::Mat>& imgs, std::vector<int>& galvoX,
		std::vector<int>& galvoY)
	{
		imgs.clear();
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_regular_file()) {
				std::string filePath = entry.path().string();
				cv::Mat image = cv::imread(filePath, 0);
				if (!image.empty()) {
					imgs.emplace_back(image);
				}			
			}
		}
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_regular_file()) {
				std::string name = entry.path().stem().string();
				std::istringstream stream(name);
				std::string part;
				std::vector<std::string> parts;
				while (std::getline(stream, part, '#')) {
					parts.emplace_back(part);
				}

				galvoX.emplace_back(std::stoi(parts[parts.size() - 2]));
				galvoY.emplace_back(std::stoi(parts[parts.size() - 1]));				
			}
		}	
	}

	void LoadFilePaths(const std::string& folderPath,
		std::vector<std::string>& filePaths,
		std::vector<int>& galvoX, std::vector<int>& galvoY)
	{
		filePaths.clear();
		galvoX.clear();
		galvoY.clear();
		for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
			if (entry.is_regular_file()) {			
				filePaths.emplace_back(entry.path().string());

				std::string name = entry.path().stem().string();
				std::istringstream stream(name);
				std::string part;
				std::vector<std::string> parts;
				while (std::getline(stream, part, '#')) {
					parts.emplace_back(part);
				}
				galvoX.emplace_back(std::stoi(parts[parts.size() - 2]));
				galvoY.emplace_back(std::stoi(parts[parts.size() - 1]));
			}
		}
	}

	bool WritePointCloud(const std::string& filePath,
		const std::vector<cv::Point3d>& points)
	{
		const std::string suffix = "txt";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}

		std::ofstream out(filePath);
		for (const auto& p : points) {
			out << p.x << " " << p.y << " " << p.z << "\n";
		}
		out.close();
		return true;
	}

	bool WritePointCloudBinary(const std::string& filePath,
		const std::vector<cv::Point3d>& points)
	{
		const std::string suffix = "bin";
		if (!CheckFileSuffix(filePath, suffix)) {
			return false;
		}
		std::ofstream ofs(filePath, std::ios::binary);
		if (!ofs) {
			return false;
		}

		size_t size = points.size();
		ofs.write(reinterpret_cast<const char*>(&size), sizeof(size));
		ofs.write(reinterpret_cast<const char*>(points.data()), size * sizeof(cv::Point3d));
		ofs.close();
		return true;
	}

}// namespace vlp