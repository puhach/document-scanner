#include <iostream>
#include <vector>
#include <algorithm>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


class DocumentScanner
{
public:
	DocumentScanner() = default;

	// TODO: implement copy/move semantics

	cv::Mat rectify1(const cv::Mat& src);	// TODO: consider making it const
	cv::Mat rectify2(const cv::Mat& src);
	cv::Mat rectify3(const cv::Mat& src);
	cv::Mat rectify4(const cv::Mat& src);
	cv::Mat rectify(const cv::Mat& src);

	cv::Mat rectify6(const cv::Mat& src);
};	// DocumentScanner

cv::Mat DocumentScanner::rectify6(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	cv::imshow("test", srcChannelsHSV[0]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[1]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	cv::Mat1f histS;
	cv::calcHist(srcChannelsHSV, std::vector{ 1 }, cv::Mat(), histS, std::vector{ 256 }, std::vector{0, 256.0f});

	int maxIdx[2];
	cv::minMaxIdx(histS, nullptr, nullptr, nullptr, maxIdx);
	int dominantSat = maxIdx[0];

	cv::Mat devMat;
	cv::absdiff(srcChannelsHSV[1], cv::Scalar(dominantSat), devMat);
	cv::Scalar meanDev = cv::mean(devMat);
	//meanDev[0] = std::sqrt(meanDev[0]);

	//cv::Mat1b mask;
	//cv::inRange(srcChannelsHSV[1], cv::Scalar{ dominantSat - meanDev[0] }, cv::Scalar{dominantSat+meanDev[0]}, mask);

	cv::Mat1b mask;
	cv::threshold(srcChannelsHSV[1], mask, dominantSat+meanDev[0], 255, cv::THRESH_BINARY_INV);

	cv::imshow("test", mask);
	cv::waitKey();

	return mask;
}


cv::Mat DocumentScanner::rectify(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	cv::imshow("test", srcChannelsHSV[0]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[1]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	// Compute the 2D histogram of saturation-value pairs
	cv::Mat1f histSV;
	cv::calcHist(srcChannelsHSV, std::vector{ 1, 2 }, cv::Mat(), histSV, std::vector{ 256, 256 }, std::vector{ 0.0f, 256.0f, 0.0f, 256.0f });
	//cv::calcHist(std::vector{ srcChannelsHSV }, std::vector{ 0, 2 }, cv::Mat(), histSV, std::vector{ 180, 256 }, std::vector{ 0.0f, 180.0f, 0.0f, 256.0f });

	//cv::Mat1f histS;
	//cv::reduce(histSV, histS, 1, cv::REDUCE_SUM);

	int maxIdx[2];
	cv::minMaxIdx(histSV, nullptr, nullptr, nullptr, maxIdx);
	int dominantSat = maxIdx[0];
	int dominantVal = maxIdx[1];

	cv::Mat3b devMat;
	cv::absdiff(srcHSV, cv::Scalar(0, dominantSat, dominantVal), devMat);
	//cv::Scalar meanDev = cv::mean(devMat);
	cv::Scalar meanDev = cv::mean(devMat);
	//meanDev[0] = std::sqrt(meanDev[0]);


}


cv::Mat DocumentScanner::rectify4(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	cv::imshow("test", srcChannelsHSV[0]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[1]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	// Compute the 2D histogram of saturation-value pairs
	cv::Mat1f histSV;
	cv::calcHist(srcChannelsHSV, std::vector{ 1, 2 }, cv::Mat(), histSV, std::vector{ 256, 256 }, std::vector{ 0.0f, 256.0f, 0.0f, 256.0f });
	//cv::calcHist(std::vector{ srcChannelsHSV }, std::vector{ 0, 2 }, cv::Mat(), histSV, std::vector{ 180, 256 }, std::vector{ 0.0f, 180.0f, 0.0f, 256.0f });

	cv::Mat1f histS;
	cv::reduce(histSV, histS, 1, cv::REDUCE_SUM);

	int maxIdx[2];
	cv::minMaxIdx(histS, nullptr, nullptr, nullptr, maxIdx);
	int dominantSat = maxIdx[0];

	cv::Mat1b devMat;
	cv::absdiff(srcChannelsHSV[1], cv::Scalar(dominantSat), devMat);
	//cv::Scalar meanDev = cv::mean(devMat);
	cv::Scalar meanDev = cv::mean(devMat);
	//meanDev[0] = std::sqrt(meanDev[0]);

	int fromSat = std::max(0, dominantSat - int(meanDev[0]));
	int toSat = std::min(histSV.rows, dominantSat + int(meanDev[0]) + 1);
	// TODO: fix the range
	cv::Mat1b smask;
	cv::inRange(srcChannelsHSV[1], std::vector{ fromSat }, std::vector{ toSat }, smask);
	cv::imshow("test", smask);
	cv::waitKey();

	cv::Mat1f histROI = histSV(cv::Range(fromSat, toSat), cv::Range::all());
	cv::minMaxIdx(histROI, nullptr, nullptr, nullptr, maxIdx);
	int dominantVal = maxIdx[1];

	//cv::bitwise_or(srcChannelsHSV[2], )
	cv::Mat devMat1;
	cv::absdiff(srcChannelsHSV[2], cv::Scalar(dominantVal), devMat1);
	cv::Scalar meanDev1 = cv::mean(devMat1, smask);
	//meanDev1[0] = std::sqrt(meanDev1[0]);

	cv::Mat1b srcBin;
	cv::threshold(srcChannelsHSV[2], srcBin, dominantVal - meanDev1[0], 255, cv::THRESH_BINARY);
	//cv::threshold(srcChannelsHSV[2], srcBin, dominantVal, 255, cv::THRESH_BINARY);

	cv::imshow("test", srcBin);
	cv::waitKey();

	return srcBin;
}


cv::Mat DocumentScanner::rectify3(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	cv::imshow("test", srcChannelsHSV[0]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[1]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	// Compute the 2D histogram of hue-value pairs
	cv::Mat1f histHV;
	cv::calcHist(srcChannelsHSV, std::vector{ 0, 2 }, cv::Mat(), histHV, std::vector{ 180, 256 }, std::vector{ 0.0f, 180.0f, 0.0f, 256.0f });
	//cv::calcHist(std::vector{ srcChannelsHSV }, std::vector{ 0, 2 }, cv::Mat(), histSV, std::vector{ 180, 256 }, std::vector{ 0.0f, 180.0f, 0.0f, 256.0f });

	cv::Mat1f histH;
	cv::reduce(histHV, histH, 1, cv::REDUCE_SUM);

	int maxIdx[2];
	cv::minMaxIdx(histH, nullptr, nullptr, nullptr, maxIdx);
	int dominantHue = maxIdx[0];

	cv::Mat1b devMat;
	cv::absdiff(srcChannelsHSV[0], cv::Scalar(dominantHue), devMat);
	//cv::Scalar meanDev = cv::mean(devMat);
	cv::Scalar meanDev = cv::mean(devMat);
	meanDev[0] = std::sqrt(meanDev[0]);

	int fromRow = dominantHue - int(meanDev[0]);
	int toRow = dominantHue + int(meanDev[0]);
	// TODO: fix the range
	cv::Mat1b hmask;
	cv::inRange(srcChannelsHSV[0], std::vector{ fromRow }, std::vector{toRow}, hmask);
	cv::imshow("test", hmask);
	cv::waitKey();

	cv::Mat1f histROI = histHV(cv::Range(fromRow, toRow), cv::Range::all());
	cv::minMaxIdx(histROI, nullptr, nullptr, nullptr, maxIdx);
	int dominantVal = maxIdx[1];

	cv::Mat devMat1;
	cv::Mat1b valROI = srcChannelsHSV[2](cv::Range(fromRow, toRow), cv::Range::all());
	cv::absdiff(valROI, cv::Scalar(dominantVal), devMat1);
	cv::Scalar meanDev1 = cv::mean(devMat1);
	meanDev1[0] = std::sqrt(meanDev1[0]);

	cv::Mat1b srcBin;
	cv::threshold(srcChannelsHSV[2], srcBin, dominantVal-meanDev[0], 255, cv::THRESH_BINARY);
	//cv::threshold(srcChannelsHSV[2], srcBin, dominantVal, 255, cv::THRESH_BINARY);

	cv::imshow("test", srcBin);
	cv::waitKey();

	return srcBin;
}

cv::Mat DocumentScanner::rectify2(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);

	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	// Compute the 2D histogram of hue-value pairs
	cv::Mat histHue, histVal;
	//int hueChannels[] = { 0 }, valChannels = { 2 };
	//int histSizeHue[] = { 180 }, histSizeVal = { 256 };
	////float range[] = { 0, 179 }; 
	//const float hueRange[] = {0, 179};
	//cv::calcHist(&srcHSV, 1, hueChannels, cv::Mat(), histHue, 1, histSizeHue, { hueRange }, true, false);

	cv::calcHist(srcChannelsHSV, std::vector{ 0 }, cv::Mat(), histHue, std::vector{ 180 }, std::vector{ 0.0f, 179.0f });
	cv::calcHist(srcChannelsHSV, std::vector{ 2 }, cv::Mat(), histVal, std::vector{ 256 }, std::vector{ 0.0f, 255.0f });

	int dominantHue, dominantVal, maxIdx[2];
	cv::minMaxIdx(histHue, nullptr, nullptr, nullptr, maxIdx);
	dominantHue = maxIdx[0];
	cv::minMaxIdx(histVal, nullptr, nullptr, nullptr, maxIdx);
	dominantVal = maxIdx[0];

	cv::Mat devMat;
	cv::absdiff(srcHSV, cv::Scalar(dominantHue, 0, dominantVal), devMat);
	cv::Scalar meanDev = cv::mean(devMat);

	cv::Mat1b hueMask, valMask;
	cv::inRange(srcChannelsHSV[0], { dominantHue - meanDev[0] }, { dominantHue + meanDev[0] }, hueMask);
	cv::inRange(srcChannelsHSV[2], { dominantVal - meanDev[2] }, { dominantVal + meanDev[2] }, valMask);

	cv::imshow("test", hueMask);
	cv::waitKey();

	cv::imshow("test", valMask);
	cv::waitKey();

	cv::Mat1b mask;
	cv::bitwise_and(hueMask, valMask, mask);
	cv::imshow("test", mask);
	cv::waitKey();

	return mask;
}

cv::Mat DocumentScanner::rectify1(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);
	//cv::Mat1b srcGray;
	//cv::cvtColor(src, srcGray, CV_)

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	cv::Mat1b srcChannelsHSV[3];
	cv::split(srcHSV, srcChannelsHSV);
		
	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	// Compute the 2D histogram of hue-value pairs
	cv::Mat histHV;
	int histChannels[] = { 0, 2 };
	int histSize[] = { 180, 256 };
	const float hueRange[] = { 0, 179 }, valRange[] = { 0, 255 }, *ranges[] = { hueRange, valRange };
	cv::calcHist(&srcHSV, 1, histChannels, cv::Mat(), histHV, 2, histSize, ranges, true, false);

	cv::Point maxLoc;
	double maxVal;
	cv::minMaxLoc(histHV, nullptr, &maxVal, nullptr, &maxLoc);
	//cv::minMaxLoc(srcChannelsHSV[0], nullptr, &maxVal);
	//cv::minMaxLoc(srcChannelsHSV[2], nullptr, &maxVal);
	//int dominantSat = hueRange[1] * maxLoc.y / histSize[0];
	int dominantHue = maxLoc.y;
	int dominantVal = maxLoc.x;

	cv::Mat devHSV;
	cv::absdiff(srcHSV, cv::Scalar(dominantHue, 0, dominantVal), devHSV);
	cv::Scalar meanDev = cv::mean(devHSV);

	cv::Mat1b mask;
	cv::inRange(srcHSV, cv::Scalar{ dominantHue - meanDev[0], 0, dominantVal - meanDev[2] }, cv::Scalar{ dominantHue + meanDev[0], 255, dominantVal + meanDev[2] }, mask);
	cv::imshow("test", mask);
	cv::waitKey();

	cv::Mat1b srcBin = mask;

	//int thresh = dominantVal - static_cast<int>(meanDev[2]);

	//cv::Mat1b srcBin;
	//cv::threshold(srcChannelsHSV[2], srcBin, thresh, 255, cv::THRESH_BINARY);
	////cv::adaptiveThreshold(channels[2], srcBin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, 1);
	////cv::imshow("test", channels[2]);
	//
	//cv::imshow("test", srcBin);
	//cv::waitKey();

	//cv::dilate(srcBin, srcBin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1,-1), 3);
	cv::morphologyEx(srcBin, srcBin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)), cv::Point(-1,-1), 3);
	cv::imshow("test", srcBin);
	cv::waitKey();

	std::vector<std::vector<cv::Point>> contours;
	//cv::findContours(srcBin, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	cv::findContours(srcBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	
	cv::Mat srcDecorated = src.clone();
	cv::drawContours(srcDecorated, contours, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	cv::imshow("test", srcDecorated);
	cv::waitKey();

	return cv::Mat();
}

int main(int argc, char* argv[])
{
	try
	{
		//cv::Mat imSrc = cv::imread("./images/scanned-form.jpg", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used
		cv::Mat imSrc = cv::imread("./images/ref2.jpg", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used
		//cv::Mat imSrc = cv::imread("./images/sunglass.png", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used

		DocumentScanner scanner;
		scanner.rectify(imSrc);		// TODO: perhaps, pass a file name as a window title

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}
