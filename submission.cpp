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



/************************************************************************************
*
*	Paper detection classes
*
*************************************************************************************/

// A parent class for detectors which return a quad rather than a bounding box
class AbstractQuadDetector
{
public:
	virtual ~AbstractQuadDetector() = default;
	// TODO: define copy/move semantics

	//virtual std::vector<cv::Point> detect(const cv::Mat &image) = 0;	// TODO: perhaps, make it const?

	std::vector<cv::Point> detect(const cv::Mat& image) const;

private:
	virtual std::vector<std::vector<cv::Point>> detectCandidates(const cv::Mat& image) const = 0;
	virtual std::vector<cv::Point> selectBestCandidate(const std::vector<std::vector<cv::Point>>& candidates) const = 0;
};	// AbstractQuadDetector

std::vector<cv::Point> AbstractQuadDetector::detect(const cv::Mat& image) const
{
	return selectBestCandidate(detectCandidates(image));
}


// A parent class for paper sheet detectors
class AbstractPaperSheetDetector : public AbstractQuadDetector
{
public:

	constexpr double getMinAreaPct() const noexcept { return this->minAreaPct; }
	constexpr void setMinAreaPct(double minAreaPct) 
	{ 
		this->minAreaPct = minAreaPct>=0 && minAreaPct<=this->maxAreaPct ? 
			minAreaPct : throw std::invalid_argument("Min area percentage must be in range 0..<max area percentage>."); 
	}

	constexpr double getMaxAreaPct() const noexcept { return this->maxAreaPct; }
	constexpr void setMaxAreaPct(double maxAreaPct) 
	{ 
		this->maxAreaPct = maxAreaPct>=this->minAreaPct && maxAreaPct<=1 ? 
			maxAreaPct : throw std::invalid_argument("Max area percentage must be in range <min area percentage>..1."); 
	}

	constexpr void setAreaPctRange(double minAreaPct, double maxAreaPct)
	{
		this->minAreaPct = minAreaPct >= 0 && minAreaPct <= maxAreaPct ? minAreaPct : throw std::invalid_argument("Min area percentage must be in range 0..<max area percentage>.");
		this->maxAreaPct = maxAreaPct >= minAreaPct && maxAreaPct <= 1 ? maxAreaPct : throw std::invalid_argument("Max area percentage must be in range <min area percentage>..1.");
	}

	constexpr double getApproximationAccuracyPct() const noexcept { return this->approxAccuracyPct; }
	constexpr void setApproximationAccuracyPct(double approxAccuracyPct) 
	{ 
		this->approxAccuracyPct = approxAccuracyPct >= 0 ? approxAccuracyPct : throw std::invalid_argument("Approximation accuracy percentage can't be negative."); 
	}

protected:

	constexpr AbstractPaperSheetDetector(double minAreaPct = 0.5, double maxAreaPct = 0.99, double approximationAccuracyPct = 0.02);

	// Approximate the contours and remove inappropriate ones
	virtual std::vector<std::vector<cv::Point>> refineContours(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat &image) const;

private:
	virtual std::vector<cv::Point> selectBestCandidate(const std::vector<std::vector<cv::Point>>& candidates) const override; 

	double minAreaPct, maxAreaPct;	// the min and max fractions of the image area that the paper sheet must occupy to be considered for detection
	double approxAccuracyPct;	// the fraction of the contour length which defines the maximum distance between the original curve and its approximation
};	// AbstractPaperSheetDetector


constexpr AbstractPaperSheetDetector::AbstractPaperSheetDetector(double minAreaPct, double maxAreaPct, double approxAccuracyPct) 
	: minAreaPct(minAreaPct>=0 && minAreaPct<=1 ? minAreaPct : throw std::invalid_argument("Min area percentage must be in range 0..1."))
	, maxAreaPct(maxAreaPct>=minAreaPct && maxAreaPct<=1 ? maxAreaPct : throw std::invalid_argument("Max area percentage must be in range <min area percentage>..1"))
	, approxAccuracyPct(approxAccuracyPct>=0 ? approxAccuracyPct : throw std::invalid_argument("Approximation accuracy percentage can't be negative."))
{
}

std::vector<std::vector<cv::Point>> AbstractPaperSheetDetector::refineContours(const std::vector<std::vector<cv::Point>>& contours, const cv::Mat &image) const
{
	const double imageArea = 1.0 * image.cols * image.rows;

	std::vector<std::vector<cv::Point>> refinedContours;
	for (const auto& contour : contours)
	{
		std::vector<cv::Point> contourApprox;
		cv::approxPolyDP(contour, contourApprox, this->approxAccuracyPct * cv::arcLength(contour, true), true);

		if (contourApprox.size() != 4 || !cv::isContourConvex(contourApprox))
			continue;

		double approxArea = cv::contourArea(contourApprox, false);

		//if (approxArea < 0.5*imageArea || approxArea > 0.99*imageArea)
		if (approxArea < this->minAreaPct*imageArea || approxArea > this->maxAreaPct*imageArea)
			continue;

		refinedContours.push_back(std::move(contourApprox));
	}	// for each contour

	return refinedContours;
}

std::vector<cv::Point> AbstractPaperSheetDetector::selectBestCandidate(const std::vector<std::vector<cv::Point>>& candidates) const
{
	if (candidates.empty())
		throw std::runtime_error("The list of candidates is empty.");

	for (int i = 1; i < candidates.size(); ++i)
		if (candidates[i].size() != candidates[i-1].size())
			throw std::runtime_error("The candidates have different number of vertices.");

	std::vector<double> rank(candidates.size(), 0);
	int bestCandIdx = 0;
	for (int i = 0; i < candidates.size(); ++i)
	{
		for (int j = 0; j < candidates.size(); ++j)
		{
			if (i == j)
				continue;

			double maxDist = 0;			
			for (int v = 0; v < candidates[i].size(); ++v)
			{
				double d = cv::norm(candidates[j][v] - candidates[i][v]);
				maxDist = std::max(d, maxDist);
			}	// v

			rank[i] += std::exp(-maxDist);
		}	// j

		if (rank[i] > rank[bestCandIdx])
			bestCandIdx = i;
	}	// i

	// Another working option
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

	return candidates[bestCandIdx];
}	// selectBestCandidate


// A paper sheet quad detector based on iterative thresholding of the saturation and value channels
class IthreshPaperSheetDetector : public AbstractPaperSheetDetector
{
public:
	IthreshPaperSheetDetector(int thresholdLevels = 7)
		: thresholdLevels(thresholdLevels)
	{
	}

	// TODO: define copy/move semantics
		
private:
	
	virtual std::vector<std::vector<cv::Point>> detectCandidates(const cv::Mat& image) const override;

	int thresholdLevels;
};	// IthreshPaperSheetDetector


std::vector<std::vector<cv::Point>> IthreshPaperSheetDetector::detectCandidates(const cv::Mat& src) const
{
	CV_Assert(src.depth() == CV_8U);
	// TODO: can a grayscale image be converted to HSV?
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

		constexpr int threshLevels = 3;	// TODO: maybe add this as a parameter

		for (int threshLevel = 1; threshLevel <= threshLevels; ++threshLevel)
		{
			cv::Mat1b channelBin;
			cv::threshold(channel, channelBin, threshLevel * 255.0 / (threshLevels + 1.0), 255,
				i == 1 ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);	// for a value channel use another threshold

			//cv::dilate(channelBin, channelBin, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
			cv::morphologyEx(channelBin, channelBin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

			double minVal, maxVal;
			cv::minMaxLoc(channelBin, &minVal, &maxVal);
			if (minVal > 254 || maxVal < 1)	// all black or all white
				continue;

			std::vector<std::vector<cv::Point>> contours;	// TODO: perhaps, make it local to reduce memory allocations
			cv::findContours(channelBin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			auto&& refinedContours = refineContours(contours, channelBin);
			std::move(refinedContours.begin(), refinedContours.end(), std::back_inserter(candidates));
		}	// threshLevel
	}	// for i channel

	if (candidates.empty())
		candidates.push_back(std::vector<cv::Point>{ {0, 0}, { 0, src.rows - 1 }, { src.cols - 1, src.rows - 1 }, { src.cols - 1, 0 } });
		//candidates.push_back(std::vector<cv::Point>{ {0, 0}, { 0, src.rows - 1 }, { src.rows - 1, src.cols - 1 }, { 0, src.cols - 1 } });

	return candidates;
}	// detectCandidates

// A paper sheet detector based on a dominant saturation-value pair
class SavaldoPaperSheetDetector : public AbstractPaperSheetDetector
{
public:
private:
	virtual std::vector<std::vector<cv::Point>> detectCandidates(const cv::Mat& image) const override;
	//virtual std::vector<cv::Point> selectBestCandidate(const std::vector<std::vector<cv::Point>>& candidates) const override;
};	// SavaldoPaperSheetDetector

std::vector<std::vector<cv::Point>> SavaldoPaperSheetDetector::detectCandidates(const cv::Mat& src) const
{
	CV_Assert(src.depth() == CV_8U);

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

//std::vector<cv::Point> SavaldoPaperSheetDetector::selectBestCandidate(const std::vector<std::vector<cv::Point>>& candidates) const
//{
//	CV_Assert(!candidates.empty());
//
//	auto it = std::max_element(candidates.begin(), candidates.end(), [](const auto& c1, const auto& c2) { 
//			return cv::contourArea(c1) < cv::contourArea(c2); 
//		});
//
//	return *it;
//}	// selectBestCandidate


class DocumentScanner
{
public:
	DocumentScanner() = default;	// TODO: perhaps, add a window parameter

	// TODO: implement copy/move semantics

	const cv::String& getWindowName() const noexcept { return this->windowName; }
	void setWindowName(const cv::String& windowName) { this->windowName = windowName; }

	void setDetector(std::unique_ptr<AbstractPaperSheetDetector> detector) { this->paperDetector = std::move(detector); }

	cv::Mat rectify(const cv::Mat& src, std::vector<cv::Point> &quad, int width, int height, bool canonicalView = true);

	bool prepare(const cv::Mat& src, std::vector<cv::Point> &quad);

	bool display(const cv::Mat &image);

private:

	static void onMouseEvent(int event, int x, int y, int flags, void *userData);

	void drawSelection();

	std::vector<cv::Point2f> arrangeVerticesClockwise(const std::vector<cv::Point> &quad);

	constexpr static int minPointRadius = 3;
	constexpr static int minLineWidth = 1;

	cv::Mat src, srcDecorated;
	cv::String windowName;
	//std::vector<std::vector<cv::Point>> candidates;
	//int bestCandIdx = -1;	
	std::vector<cv::Point> bestQuad;
	cv::Point* ptDragged = nullptr;		// raw pointers are fine if the pointer is non-owning
	std::unique_ptr<AbstractPaperSheetDetector> paperDetector = std::make_unique<IthreshPaperSheetDetector>();	// TODO: add getter/setter
};	// DocumentScanner


bool DocumentScanner::prepare(const cv::Mat& src, std::vector<cv::Point>& quad)
{	
	this->src = src;
	this->ptDragged = nullptr;

	// Detect the paper sheet boundaries
	this->bestQuad = quad = this->paperDetector->detect(src);

	// A mouse handler will let the user move the vertices in case our automatic selection was not correct
	cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(this->windowName, &DocumentScanner::onMouseEvent, this);
		
	drawSelection();	// draw the boundaries of the paper sheet we have just found

	int key = -1;
	do
	{
		cv::imshow(this->windowName, this->srcDecorated);
		key = cv::waitKey(10);
	} while (key < 0);

	cv::destroyWindow(this->windowName);

	return (key & 0xFF) != 27;	// not escape
}	// prepare

cv::Mat DocumentScanner::rectify(const cv::Mat& src, std::vector<cv::Point>& quad, int width, int height, bool canonicalView)
{
	CV_Assert(!src.empty());
	CV_Assert(quad.size() == 4);

	cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);

	// In order to perform perspective correction, the source and destination points must be consistently ordered (clockwise order is used)
	std::vector<cv::Point2f> srcQuadF = arrangeVerticesClockwise(quad);

	// In a canonical view mode the width denotes a measure of the horizontal dimension of a correctly aligned document as seen 
	// in a frontal view, i.e. it doesn't depend on how the document is actually positioned in the input image. 
	// Otherwise, the width corresponds to the dimension which is closer to horizontal in the input image.
	bool rotateImage = false;
	if (canonicalView)
	{
		// Estimate width and height of the document from the source image
		double wSrc = cv::norm(srcQuadF[0] - srcQuadF[1]), hSrc = cv::norm(srcQuadF[0] - srcQuadF[3]);

		// Try to determine the orientation 
		if ((wSrc >= hSrc) != (width >= height))
		{
			std::swap(width, height);
			rotateImage = true;
		}
	}	// canonical view

	// Define the destination points in the clockwise order (must be consistent with the source points order)
	std::vector<cv::Point2f> dstQuadF{ { 0.0f, 0.0f }, {width - 1.0f, 0.0f}, {width - 1.0f, height - 1.0f}, {0.0f, height - 1.0f} };

	if (cv::Mat m = cv::getPerspectiveTransform(srcQuadF, dstQuadF); m.empty())
		return src;		// if we failed to obtain a perspective transform matrix, just return the source image
	else
	{
		// Perform perspective correction using the obtained transformation matrix 
		cv::Mat dst; 
		cv::warpPerspective(src, dst, m, cv::Size(width, height));

		if (rotateImage)
			cv::rotate(dst, dst, cv::ROTATE_90_CLOCKWISE);

		return dst;
	}
}	// rectify

std::vector<cv::Point2f> DocumentScanner::arrangeVerticesClockwise(const std::vector<cv::Point>& quad)
{
	// Find the top left vertex, i.e. the closest vertex to the (0,0) corner
	auto accTopLeft = std::accumulate(quad.begin() + 1, quad.end(),
		std::pair<double, const cv::Point*>(cv::norm(quad[0]), &quad[0]),
		[&quad](const auto& acc, const auto& p) {
			if (double d = cv::norm(p); d < acc.first)		// distance to (0,0)
				return std::make_pair(d, &p);
			else
				return acc;
		});

	//const cv::Point& p0 = *accTopLeft.second;
	//
	//cv::Point u0 = -p0;	// the vector from p0 to (0,0)
	//if (u0.x == 0 && u0.y == 0)	// in case p0 is in the top-left corner, u0 may point anywhere outside the image rectangle
	//	u0.x = u0.y = -1;

	// Compute angles between u0 = [upwards from the top-left vertex] and the vectors from the top-left vertex to each other vertex
	std::vector<double> angles(quad.size());
	std::transform(quad.begin(), quad.end(), angles.begin(), [&p0 = *accTopLeft.second, u0 = cv::Point(accTopLeft.second->x, -1)](const auto& p) {
		cv::Point u = p - p0;	// the vector from p0 to p
		//cv::Point u0 = -p0;		// the vector from p0 to (0,0)

		// Compute the dot product of u0 and u: u0.x*u.x+u0.y*u.y = |u0|*|u|*cos(angle)
		int dp = u0.x * u.x + u0.y * u.y;

		// Compute the cross product of u0 and u: u0.x*u.y - u0.y*u.x = |u0|*|u|*sin(angle)
		int cp = u0.x * u.y - u0.y * u.x;

		// cp/dp = sin(angle)/cos(angle) = tan(angle)
		// Domain error may occur if cp and dp are both zero:
		// https://en.cppreference.com/w/cpp/numeric/math/atan2
		double angle = cp || dp ? std::atan2(cp, dp) : 0;

		if (angle < 0)	// atan2 returns values from [-pi, +pi]
		{
			angle += 2 * std::acos(-1.0);	// add 2*pi to obtain a positive angle
		}

		return angle;
	});	// transform


	// Arrange the vertices in the clockwise order starting from the top-left vertex

	std::vector<int> indices(quad.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&angles](int idx1, int idx2) {
		return angles[idx1] < angles[idx2];
		});

	std::vector<cv::Point2f> quadF;
	quadF.reserve(quad.size());
	for (int idx : indices)
	{
		quadF.push_back(quad[idx]);
	}

	return quadF;
}	// arrangeVerticesClockwise

bool DocumentScanner::display(const cv::Mat& image)
{
	cv::imshow(this->windowName, image);
	return (cv::waitKey() & 0xFF) != 27;
}

void DocumentScanner::onMouseEvent(int event, int x, int y, int flags, void* userData)
{
	DocumentScanner* scanner = static_cast<DocumentScanner*>(userData);

	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:
	{
		CV_Assert(!scanner->bestQuad.empty());
		CV_Assert(!scanner->src.empty());

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
		double minDist = std::numeric_limits<double>::infinity();
		for (auto& p : scanner->bestQuad)
		{
			if (double d = cv::norm(p - p0); d < minDist)
			{
				minDist = d;
				scanner->ptDragged = &p;
			}
		}

		if (minDist > std::max(minPointRadius + 1.0, 0.01 * cv::norm(cv::Point(scanner->src.rows, scanner->src.cols))))	// not close enough?
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
			CV_Assert(!scanner->src.empty());

			if ((flags & cv::EVENT_FLAG_LBUTTON) && x >= 0 && y >= 0 && x < scanner->src.cols && y < scanner->src.rows)
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
	CV_Assert(!this->src.empty());
	CV_Assert(!this->bestQuad.empty());

	this->src.copyTo(this->srcDecorated);

	// Draw the lines
	double diag = cv::norm(cv::Point(this->src.rows, this->src.cols));
	int lineWidth = std::max(minLineWidth, static_cast<int>(0.001 * diag));
	cv::polylines(this->srcDecorated, this->bestQuad, true, cv::Scalar(0, 255, 0), lineWidth, cv::LINE_AA);

	// Draw the vertices
	int rInner = std::max(minPointRadius, static_cast<int>(0.007 * diag));
	int rOuter = std::max(rInner + 1, static_cast<int>(0.008 * diag));
	for (const auto& p : this->bestQuad)
	{
		cv::circle(this->srcDecorated, p, rOuter, &p == this->ptDragged ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
		cv::circle(this->srcDecorated, p, rInner, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
	}	// for
}	// drawSelection




int main(int argc, char* argv[])
{
	try
	{
		//cv::Mat imSrc = cv::imread("./images/scanned-form.jpg", cv::IMREAD_COLOR);	// TODO: not sure what reading mode should be used
		cv::Mat imSrc = cv::imread("./images/mozart2.jpg", cv::IMREAD_COLOR);	// EXIF is important
		//cv::Mat imSrc = cv::imread("./images/sens2.jpg", cv::IMREAD_COLOR);	// EXIF is important


		DocumentScanner scanner;
		scanner.setWindowName("my");
		scanner.setDetector(std::make_unique<SavaldoPaperSheetDetector>());		// TEST!
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
