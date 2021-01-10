#include "savaldopapersheetdetector.h"



std::unique_ptr<AbstractQuadDetector> SavaldoPaperSheetDetector::createClone() const
{
	return std::unique_ptr<AbstractQuadDetector>(new SavaldoPaperSheetDetector(*this));
}

std::vector<std::vector<cv::Point>> SavaldoPaperSheetDetector::detectCandidates(const cv::Mat& src) const
{
	CV_Assert(!src.empty());
	CV_Assert(src.depth() == CV_8U);
	CV_Assert(src.channels() >= 3);		// a grayscale image can't be converted to HSV

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	// Compute a 2D histogram of saturation-value pairs
	cv::Mat1f histSV;
	cv::calcHist(srcChannelsHSV, std::vector{ 1, 2 }, cv::Mat(), histSV, std::vector{ 256, 256 }, std::vector{ 0.0f, 256.0f, 0.0f, 256.0f });

	int maxIdx[2];
	cv::minMaxIdx(histSV, nullptr, nullptr, nullptr, maxIdx);
	int dominantSat = maxIdx[0];
	int dominantVal = maxIdx[1];

	cv::Mat3b devMat;
	cv::absdiff(srcHSV, cv::Scalar(0, dominantSat, dominantVal), devMat);
	cv::Scalar meanDev = cv::mean(devMat);
	//meanDev[0] = std::sqrt(meanDev[0]);

	cv::Mat1b srcBin;
	cv::inRange(srcHSV, cv::Scalar{ 0, dominantSat - meanDev[1], dominantVal - meanDev[2] }, cv::Scalar{ 179, dominantSat + meanDev[1], dominantVal + meanDev[2] }, srcBin);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(srcBin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		
	std::vector<std::vector<cv::Point>> candidates = refineContours(contours, src);
	
	if (candidates.empty())
		candidates.push_back(std::vector<cv::Point>{ {0, 0}, { 0, src.rows - 1 }, { src.cols - 1, src.rows - 1 }, { src.cols - 1, 0 } });

	return candidates;
}	// detectCandiates
