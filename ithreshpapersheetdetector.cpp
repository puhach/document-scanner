#include "ithreshpapersheetdetector.h"


std::unique_ptr<AbstractQuadDetector> IthreshPaperSheetDetector::createClone() const
{
	return std::unique_ptr<AbstractQuadDetector>(new IthreshPaperSheetDetector(*this));
}	// createClone

std::vector<std::vector<cv::Point>> IthreshPaperSheetDetector::detectCandidates(const cv::Mat& src) const
{
	CV_Assert(!src.empty());
	CV_Assert(src.depth() == CV_8U);
	CV_Assert(src.channels() >= 3);		// a grayscale image can't be converted to HSV

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	// Downscale and upscale the image to filter out useless details
	cv::Mat pyr;
	cv::pyrDown(srcHSV, pyr, cv::Size(srcHSV.cols / 2, srcHSV.rows / 2));
	cv::pyrUp(pyr, srcHSV, srcHSV.size());

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	
	std::vector<std::vector<cv::Point>> candidates;

	for (int i = 1; i < srcChannelsHSV.size(); ++i)
	{
		cv::Mat1b channel = srcChannelsHSV[i];

		for (int threshLevel = 1; threshLevel <= this->thresholdLevels; ++threshLevel)
		{
			cv::Mat1b channelBin;
			cv::threshold(channel, channelBin, threshLevel * 255.0 / (this->thresholdLevels + 1.0), 255,
				i == 1 ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);	// for a value channel use another threshold

            // Remove small dark regions (holes) in the paper sheet			
			cv::morphologyEx(channelBin, channelBin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            //cv::dilate(channelBin, channelBin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            
			double minVal, maxVal;
			cv::minMaxLoc(channelBin, &minVal, &maxVal);
			if (minVal > 254 || maxVal < 1)	// all black or all white
				continue;

			std::vector<std::vector<cv::Point>> contours;    
			cv::findContours(channelBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
			auto&& refinedContours = refineContours(contours, channelBin);
			std::move(refinedContours.begin(), refinedContours.end(), std::back_inserter(candidates));
		}	// threshLevel
	}	// for i channel

	if (candidates.empty())
		candidates.push_back(std::vector<cv::Point>{ {0, 0}, { 0, src.rows - 1 }, { src.cols - 1, src.rows - 1 }, { src.cols - 1, 0 } });

	return candidates;
}	// detectCandidates

