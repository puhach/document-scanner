#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>


class DocumentScanner
{
public:
	DocumentScanner() = default;	// TODO: perhaps, add a window parameter

	// TODO: implement copy/move semantics

	const cv::String& getWindowName() const noexcept { return this->windowName; }
	void setWindowName(const cv::String& windowName) { this->windowName = windowName; }

	cv::Mat rectify1(const cv::Mat& src);	// TODO: consider making it const
	cv::Mat rectify2(const cv::Mat& src);
	cv::Mat rectify3(const cv::Mat& src);
	cv::Mat rectify4(const cv::Mat& src);
	cv::Mat rectify5(const cv::Mat& src);
	cv::Mat rectify6(const cv::Mat& src);
	cv::Mat rectify7(const cv::Mat& src);
	cv::Mat rectify8(const cv::Mat& src);
	cv::Mat rectify9(const cv::Mat& src);

	cv::Mat rectify10(const cv::Mat& src);

	cv::Mat rectify(const cv::Mat& src, std::vector<cv::Point> &quad, int width, int height);

	bool prepare(const cv::Mat& src, std::vector<cv::Point> &quad);

	bool display(const cv::Mat &image);

private:

	//static void onSelectionChanged(int pos, void* userData);

	static void onMouseEvent(int event, int x, int y, int flags, void *userData);

	void drawSelection();

	constexpr static int minPointRadius = 3;
	constexpr static int minLineWidth = 1;

	cv::Mat src, srcDecorated;
	cv::String windowName;
	//std::vector<std::vector<cv::Point2f>> candidates;
	std::vector<std::vector<cv::Point>> candidates;
	int bestCandIdx = -1;	
	//cv::Point2f* ptDragged = nullptr;
	cv::Point* ptDragged = nullptr;
};	// DocumentScanner

bool DocumentScanner::display(const cv::Mat& image)
{
	cv::imshow(this->windowName, image);
	return (cv::waitKey() & 0xFF) != 27;
}

/*
void DocumentScanner::onSelectionChanged(int pos, void* userData)
{
	DocumentScanner* scanner = static_cast<DocumentScanner*>(userData);
	scanner->drawSelection();
}
*/

void DocumentScanner::onMouseEvent(int event, int x, int y, int flags, void* userData)
{
	DocumentScanner* scanner = static_cast<DocumentScanner*>(userData);

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
		{
			CV_Assert(!scanner->candidates.empty());
			CV_Assert(scanner->bestCandIdx >= 0 && scanner->bestCandIdx < scanner->candidates.size());

			// Find the closest point

			// v1
			/*
			auto& cand = scanner->candidates[scanner->bestCandIdx];
			auto acc = std::accumulate(cand.begin(), cand.end(), std::pair<const cv::Point*, double>{ nullptr, std::numeric_limits<double>::infinity() },
				[p0 = cv::Point(x, y)](const auto& minp, const auto& p){
				double d = cv::norm(p - p0);
				if (d < minp.second)
					return std::make_pair(&p, d);
				else
					return minp;
			});

			if (acc.second > 0.01 * cv::norm(cv::Point(scanner->src.rows, scanner->src.cols)))
				scanner->ptDragged = nullptr;
			*/

			// v2			
			cv::Point p0{ x, y };
			//cv::Point2f p0( static_cast<float>(x), static_cast<float>(y) );
			double minDist = std::numeric_limits<double>::infinity();
			for (auto& p : scanner->candidates[scanner->bestCandIdx])
			{
				if (double d = cv::norm(p - p0); d < minDist)
				{
					minDist = d;
					scanner->ptDragged = &p;
				}
			}

			if (minDist > std::max(minPointRadius+1.0, 0.01 * cv::norm(cv::Point(scanner->src.rows, scanner->src.cols))))	// not close enough?
				scanner->ptDragged = nullptr;

			scanner->drawSelection();
		}

		break;	// mouse down

	case cv::EVENT_LBUTTONUP:
		scanner->ptDragged = nullptr;
		scanner->drawSelection();
		break;	// mouse up

	case cv::EVENT_MOUSEMOVE:

		if (scanner->ptDragged)
		{
			if ((flags & cv::EVENT_FLAG_LBUTTON) && x >= 0 && y >= 0 && x < scanner->src.cols && y < scanner->src.rows)
			//if ((flags & cv::EVENT_FLAG_LBUTTON) )
			{				
				scanner->ptDragged->x = x;
				scanner->ptDragged->y = y;
			}	// left button down
			else
			{
				// 1) A user might have tried to drag the point outside the window
				// 2) A user might have released the mouse button outside the window, so we didn't detect it
				scanner->ptDragged = nullptr;				
			}

			scanner->drawSelection();
		}	// ptDragged != nullptr

		break;	// mouse move
	}	// switch
}	// onMouseEvent

void DocumentScanner::drawSelection()
{
	CV_Assert(this->bestCandIdx >= 0 && this->bestCandIdx < this->candidates.size());
		
	this->src.copyTo(this->srcDecorated);

	// Draw the lines
	double diag = cv::norm(cv::Point(this->src.rows, this->src.cols));
	int lineWidth = std::max(minLineWidth, static_cast<int>(0.001 * diag));
	cv::polylines(this->srcDecorated, this->candidates[this->bestCandIdx], true, cv::Scalar(0,255,0), lineWidth, cv::LINE_AA);
	
	// Draw the vertices
	int rInner = std::max(minPointRadius, static_cast<int>(0.007 * diag));
	int rOuter = std::max(rInner+1, static_cast<int>(0.008 * diag));	
	for (const auto& p : this->candidates[this->bestCandIdx])
	{
		cv::circle(this->srcDecorated, p, rOuter, &p == this->ptDragged ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
		cv::circle(this->srcDecorated, p, rInner, cv::Scalar(0,0,255), -1, cv::LINE_AA);
	}	// for
}	// drawSelection

// TODO: add a parameter to specify the algorithm
bool DocumentScanner::prepare(const cv::Mat& src, std::vector<cv::Point>& quad)
{
	CV_Assert(src.depth() == CV_8U);
	this->src = src;
	//this->windowName = windowName;
	this->ptDragged = nullptr;

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	// Downscale and upscale the image to filter out useless details
	cv::Mat pyr;
	cv::pyrDown(srcHSV, pyr, cv::Size(srcHSV.cols / 2, srcHSV.rows / 2));
	cv::pyrUp(pyr, srcHSV, srcHSV.size());

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);


	this->candidates.clear();

	for (int i = 1; i < srcChannelsHSV.size(); ++i)
		//for (int i = 1; i <= 1; ++i)
	{
		cv::Mat1b channel = srcChannelsHSV[i];

		constexpr int threshLevels = 3;	// TODO: maybe add this as a parameter

		for (int threshLevel = 1; threshLevel <= threshLevels; ++threshLevel)
		{
			cv::Mat1b channelBin;
			cv::threshold(channel, channelBin, threshLevel * 255.0 / (threshLevels + 1.0), 255,
				i == 1 ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);	// for a value channel use another threshold

			//cv::dilate(channelBin, channelBin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
			cv::morphologyEx(channelBin, channelBin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

			//cv::imshow("test", channelBin);
			//cv::waitKey();

			double minVal, maxVal;
			cv::minMaxLoc(channelBin, &minVal, &maxVal);
			if (minVal > 254 || maxVal < 1)	// all black or all white
				continue;

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(channelBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			//cv::findContours(channelBin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			/*cv::Mat tmp = src.clone();
			cv::drawContours(tmp, contours, -1, cv::Scalar(255, 0, 0), 4);
			cv::imshow("test", tmp);
			cv::waitKey();*/

			for (auto& contour : contours)
			{
				std::vector<cv::Point> contourApprox;
				cv::approxPolyDP(contour, contourApprox, 0.02 * cv::arcLength(contour, true), true);

				double approxArea = cv::contourArea(contourApprox, false);

				if (contourApprox.size() != 4 || approxArea < 0.5 * channel.rows * channel.cols
					|| approxArea >= 0.99 * channel.rows * channel.cols || !cv::isContourConvex(contourApprox))
					continue;
								
				this->candidates.push_back(contourApprox);
			}	// for each contour
		}	// threshLevel
	}	// for i channel

	if (this->candidates.empty())
		this->candidates.push_back(std::vector<cv::Point>{ {0, 0}, { 0, src.rows - 1 }, { src.rows-1, src.cols-1 }, { 0, src.cols - 1 } });
	
	std::vector<double> rank(this->candidates.size(), 0);
	this->bestCandIdx = 0;
	for (int i = 0; i < this->candidates.size(); ++i)
	{
		for (int j = 0; j < this->candidates.size(); ++j)
		{
			if (i == j)
				continue;

			double maxDist = 0;
			for (int v = 0; v < 4; ++v)
			{
				double d = cv::norm(this->candidates[j][v] - this->candidates[i][v]);
				maxDist = std::max(d, maxDist);
			}	// v

			rank[i] += std::exp(-maxDist);
		}	// j

		if (rank[i] > rank[this->bestCandIdx])
			this->bestCandIdx = i;
	}	// i

	// A mouse handler will let the user move the vertices in case our automatic selection was not correct
	cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(this->windowName, &DocumentScanner::onMouseEvent, this);

	// Another option would be to create a slider to choose between available candidates
	/*
	cv::createTrackbar("Selection", windowName, &this->bestCandIdx
		, static_cast<int>(this->candidates.size()) - 1
		, &DocumentScanner::onSelectionChanged, this);
		*/

	drawSelection();	// draw the boundaries of the paper sheet we have just found

	int key = -1;
	do
	{
		cv::imshow(this->windowName, this->srcDecorated);
		key = cv::waitKey(10);
	} while (key < 0);

	cv::destroyWindow(this->windowName);

	quad = this->candidates[this->bestCandIdx];

	return (key & 0xFF) != 27;	// not escape
}	// prepare

cv::Mat DocumentScanner::rectify(const cv::Mat& src, std::vector<cv::Point> &quad, int width, int height)
{
	CV_Assert(!src.empty());
	CV_Assert(quad.size() == 4);

	cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);

	//std::vector<cv::Point> destVertices{ { 0,0 }, {width - 1, 0}, {width - 1, height - 1}, {0, height-1} };
	//std::vector<cv::Point> destVertices{ { 0,0 }, {0, height - 1}, {width - 1, height - 1}, {width - 1, 0} };
	std::vector<cv::Point> destVertices{ {width - 1, 0}, { 0,0 }, {0, height - 1}, {width - 1, height - 1} };

	std::vector<cv::Point2f> quadf;
	std::vector<cv::Point2f> destvertf;
	std::transform(quad.begin(), quad.end(), std::back_inserter(quadf), [](const cv::Point& p) { return cv::Point2f(p.x, p.y); });
	std::transform(destVertices.begin(), destVertices.end(), std::back_inserter(destvertf), [](const cv::Point& p) { return cv::Point2f(p.x, p.y); });

	//if (cv::Mat homography = cv::findHomography(quad, destVertices); homography.empty())
	if (cv::Mat homography = cv::findHomography(quadf, destvertf); homography.empty())
		return src;
	else
	{
		cv::Mat dst = cv::Mat::zeros(cv::Size(width,height), CV_8UC3);
		cv::warpPerspective(src, dst, homography, cv::Size(width, height));
		return dst;
	}
}	// rectify

cv::Mat DocumentScanner::rectify10(const cv::Mat& src)
{
	cv::namedWindow("test", cv::WINDOW_NORMAL);

	CV_Assert(src.depth() == CV_8U);

	cv::Mat srcHSV;
	cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);

	// Downscale and upscale the image to filter out useless details
	cv::Mat pyr;
	cv::pyrDown(srcHSV, pyr, cv::Size(srcHSV.cols / 2, srcHSV.rows / 2));
	cv::pyrUp(pyr, srcHSV, srcHSV.size());

	//cv::Mat1b srcChannelsHSV[3];
	std::vector<cv::Mat1b> srcChannelsHSV;
	cv::split(srcHSV, srcChannelsHSV);


	cv::imshow("test", srcChannelsHSV[0]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[1]);
	cv::waitKey();
	cv::imshow("test", srcChannelsHSV[2]);
	cv::waitKey();

	std::vector<std::vector<cv::Point>> candidates;

	for (int i = 1; i < srcChannelsHSV.size(); ++i)
	//for (int i = 1; i <= 1; ++i)
	{
		cv::Mat1b channel = srcChannelsHSV[i];

		constexpr int threshLevels = 10;

		for (int threshLevel = 1; threshLevel <= threshLevels; ++threshLevel)
		{
			cv::Mat1b channelBin;
			cv::threshold(channel, channelBin, threshLevel*255.0/(threshLevels+1), 255, 
				i==1 ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);	// for a value channel use another threshold
			
			//cv::dilate(channelBin, channelBin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
			cv::morphologyEx(channelBin, channelBin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

			cv::imshow("test", channelBin);
			cv::waitKey();

			double minVal, maxVal;
			cv::minMaxLoc(channelBin, &minVal, &maxVal);
			if (minVal > 254 || maxVal < 1)	// all black or all white
				continue;

			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(channelBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			//cv::findContours(channelBin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
			cv::Mat tmp = src.clone();
			cv::drawContours(tmp, contours, -1, cv::Scalar(255,0,0), 4);
			cv::imshow("test", tmp);
			cv::waitKey();

			for (auto& contour : contours)
			{
				std::vector<cv::Point> contourApprox;
				cv::approxPolyDP(contour, contourApprox, 0.02 * cv::arcLength(contour, true), true);

				double approxArea = cv::contourArea(contourApprox, false);

				if (contourApprox.size() != 4 || approxArea < 0.5 * channel.rows * channel.cols 
					|| approxArea >= 0.99*channel.rows*channel.cols || !cv::isContourConvex(contourApprox))
						continue;

				/*long long dx = std::max(std::abs(contourApprox[0].x - contourApprox[1].x), std::abs(contourApprox[0].x - contourApprox[1].x));
				long long dy = std::max(std::abs(contourApprox[1].y - contourApprox[1].y), std::abs(contourApprox[0].y - contourApprox[1].y));

				if (dx * dy == 1LL * channel.rows * channel.cols)
					continue;*/

				candidates.push_back(contourApprox);
			}	// for each contour
		}	// threshLevel
	}	// for i channel

	if (candidates.empty())
		return src;

	cv::Mat srcCopy = src.clone();
	//cv::drawContours(srcCopy, candidates, -1, cv::Scalar(0, 255, 0), 2);
	cv::Scalar colors[] = { {255,0,0}, {0,255,0}, {0,0,255} };
	for (int i = 0; i < candidates.size(); ++i)
	{
		///cv::drawContours(srcCopy, candidates, i, cv::Scalar(10*i/256, 64*i%256, 32*i%256), 5, cv::LINE_AA, cv::noArray(), 0);
		cv::polylines(srcCopy, candidates[i], true, colors[i%3], 4);
	}
	cv::imshow("test", srcCopy);
	cv::waitKey();

	std::vector<double> rank(candidates.size(), 0);
	int bestCandIdx = 0;
	for (int i = 0; i < candidates.size(); ++i)
	{
		for (int j = 0; j < candidates.size(); ++j)
		{
			if (i == j)
				continue;

			double maxDist = 0;
			for (int v = 0; v < 4; ++v)
			{
				double d = cv::norm(candidates[j][v] - candidates[i][v]);
				maxDist = std::max(d, maxDist);
			}	// v

			rank[i] += std::exp(-maxDist);			
		}	// j

		if (rank[i] > rank[bestCandIdx])
			bestCandIdx = i;
	}	// i

	/*
	std::vector<int> rank(candidates.size(), 0);
	int bestCandIdx = 0;
	for (int i = 0; i < candidates.size(); ++i)
	{
		for (int j = 0; j < candidates.size(); ++j)
		{
			if (i == j)
				continue;

			double maxDist = 0;
			for (int v = 0; v<4; ++v)
			{
				double d = cv::norm(candidates[j][v] - candidates[i][v]);

				maxDist = std::max(d, maxDist);
			}	// v

			if (maxDist < 10)
			{
				++rank[i];
				if (rank[i] > rank[bestCandIdx])
					bestCandIdx = i;
			}
		}	// j
	}	// i
	*/

	srcCopy = src.clone();
	cv::polylines(srcCopy, candidates[bestCandIdx], true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
	cv::imshow("test", srcCopy);
	cv::waitKey();

	/*
	for (int v = 0; v < 4; ++v)
	{
		std::vector<int> xv, yv;
		for (const auto& cand : candidates)
		{
			xv.push_back(cand[v].x);
			yv.push_back(cand[v].y);
		}	// cand

		//std::sort(xv.begin(), xv.end());
		//std::sort(yv.begin(), yv.end());
		auto xvMinMax = std::minmax_element(xv.begin(), xv.end());
		auto yvMinMax = std::minmax_element(yv.begin(), yv.end());

		constexpr int step = 5;
		std::vector<int> xbuckets((*xvMinMax.second - *xvMinMax.first)/step + 1, 0);				
		for (int x : xv)
		{
			++xbuckets[(x - *xvMinMax.first) / step];
		}

		std::vector<int> ybuckets((*yvMinMax.second - *yvMinMax.first) / step + 1, 0);
		for (int y : yv)
		{
			++ybuckets[(y - *yvMinMax.second) / step];
		}

		auto xbMaxIt = std::max_element(xbuckets.begin(), xbuckets.end());
		auto 
	}	// v
	*/

	return srcCopy;
}

cv::Mat DocumentScanner::rectify9(const cv::Mat& src)
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

	/*
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(srcChannelsHSV[1], contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat srcCopy = src.clone();
	cv::drawContours(srcCopy, contours, -1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA, hierarchy);
	cv::imshow("test", srcCopy);
	cv::waitKey();
	*/

	cv::Mat srcBin;
	cv::threshold(srcChannelsHSV[1], srcBin, 50, 255, cv::THRESH_BINARY_INV);
	cv::imshow("test", srcBin);
	cv::waitKey();

	cv::Mat srcCopy = src.clone();
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(srcBin, lines, 1, CV_PI / 180, 1500);
	for (auto& line : lines)
	{
		std::cout << line << std::endl;

		float rho = line[0], theta = line[1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		cv::line(srcCopy, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}

	cv::imshow("test", srcCopy);
	cv::waitKey();
	return srcCopy;
}

cv::Mat DocumentScanner::rectify8(const cv::Mat& src)
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

	std::vector<cv::Point2d> corners;
	cv::goodFeaturesToTrack(srcChannelsHSV[1], corners, 4, 0.1, 100, cv::noArray(), 3, false);

	cv::Mat srcCopy = src.clone();
	for (auto& p : corners)
	{
		cv::circle(srcCopy, p, 5, cv::Scalar(0, 255, 0), -1);
	}

	cv::imshow("test", srcCopy);
	cv::waitKey();

	return srcCopy;
}

cv::Mat DocumentScanner::rectify7(const cv::Mat& src)
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

	cv::Mat gradx, grady;
	cv::Sobel(srcChannelsHSV[1], gradx, CV_16S, 1, 0);
	cv::Sobel(srcChannelsHSV[1], grady, CV_16S, 0, 1);

	cv::Mat gradxf, gradyf;
	gradx.convertTo(gradxf, CV_32F);
	grady.convertTo(gradyf, CV_32F);

	//cv::Mat gradx2 = gradxf.mul(gradxf);
	//cv::Mat grady2 = gradyf.mul(gradyf);
	////cv::Mat mag2 = gradx2 + grady2;
	//cv::Mat mag2;
	//cv::add(gradx2, grady2, mag2, cv::Mat(), CV_32F);
	//cv::Mat mag1;
	//cv::sqrt(mag2, mag1);
	
	cv::Mat mag;
	cv::magnitude(gradxf, gradyf, mag);

	/*cv::Mat diff;
	cv::absdiff(mag1, mag2, diff);
	cv::imshow("test", diff);
	cv::waitKey();*/

	/*cv::Scalar meanx, devx, meany, devy;
	cv::meanStdDev(gradx, meanx, devx);
	cv::meanStdDev(grady, meany, devy);*/
	cv::Scalar mean, dev;
	cv::meanStdDev(mag, mean, dev);

	cv::Mat edges;
	////cv::Canny(srcChannelsHSV[1], edges, dominantSat, dominantSat + meanDev[0]);
	cv::Canny(gradx, grady, edges, mean[0], mean[0]+dev[0]);

	cv::imshow("test", edges);
	cv::waitKey();

	cv::Mat srcCopy = src.clone();
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 30, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::line(srcCopy, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	cv::imshow("test", srcCopy);
	cv::waitKey();

	return edges;
}

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


cv::Mat DocumentScanner::rectify5(const cv::Mat& src)
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

	cv::Mat1b srcBin;
	cv::inRange(srcHSV, cv::Scalar{ 0, dominantSat - meanDev[1], dominantVal - meanDev[2] }, cv::Scalar{ 179, dominantSat + meanDev[1], dominantVal + meanDev[2] }, srcBin);
	cv::imshow("test", srcBin);
	cv::waitKey();

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(srcBin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	
	cv::Mat srcCopy = src.clone();
	cv::drawContours(srcCopy, contours, -1, cv::Scalar(0,255,0), 4, cv::LINE_AA, hierarchy, 0);
	//cv::imshow("test", srcCopy);
	//cv::waitKey();

	auto it = std::max_element(contours.begin(), contours.end(), [](const auto& c1, const auto &c2) { return cv::contourArea(c1) < cv::contourArea(c2); });
	int contourIndex = it - contours.begin();
	cv::drawContours(srcCopy, contours, contourIndex, cv::Scalar(0, 255, 0), 4);	
	cv::imshow("test", srcCopy);
	cv::waitKey();


	return srcBin;
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
		cv::Mat imSrc = cv::imread("./images/scanned-form.jpg", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used
		//cv::Mat imSrc = cv::imread("./images/mozart2.jpg", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used
		//cv::Mat imSrc = cv::imread("./images/sunglass.png", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used

		DocumentScanner scanner;
		scanner.setWindowName("my");
		//scanner.rectify(imSrc);		
		std::vector<cv::Point> quad;
		if (scanner.prepare(imSrc, quad))	// TODO: perhaps, pass a file name as a window title
		{
			cv::Mat out = scanner.rectify(imSrc, quad, 500, 707);
			scanner.display(out);
		}

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}
