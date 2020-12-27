#include <iostream>
#include <memory>
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
	
	std::vector<cv::Point> detect(const cv::Mat& image) const;

	std::unique_ptr<AbstractQuadDetector> clone() const { return createClone(); }	// NVI idiom

protected:
	AbstractQuadDetector() = default;
	
	// Restrict copy/move operations since this is a polymorphic type

	AbstractQuadDetector(const AbstractQuadDetector&) = default;
	AbstractQuadDetector(AbstractQuadDetector&&) = default;

	AbstractQuadDetector& operator = (const AbstractQuadDetector&) = delete;
	AbstractQuadDetector& operator = (AbstractQuadDetector&&) = delete;

private:	
	virtual std::unique_ptr<AbstractQuadDetector> createClone() const = 0;
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

	// Restrict copy/move operations since this is a polymorphic type

	AbstractPaperSheetDetector(const AbstractPaperSheetDetector&) = default;
	AbstractPaperSheetDetector(AbstractPaperSheetDetector&&) = default;

	AbstractPaperSheetDetector& operator = (const AbstractPaperSheetDetector&) = delete;
	AbstractPaperSheetDetector& operator = (AbstractPaperSheetDetector&&) = delete;
		

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
	constexpr IthreshPaperSheetDetector(int thresholdLevels = 15)
		: thresholdLevels(thresholdLevels>=1 && thresholdLevels<=255 ? thresholdLevels : throw std::invalid_argument("The number of threshold levels must be in range 1..255."))
	{
	}

	constexpr int getThresholdLevels() const noexcept { return this->thresholdLevels; }

	constexpr void setThresholdLevels(int thresholdLevels) 
	{ 
		this->thresholdLevels = thresholdLevels >= 1 && thresholdLevels <= 255 ? 
			thresholdLevels : throw std::invalid_argument("The number of threshold levels must be in range 1..255."); 
	}

protected:

	// Restrict copy/move operations since this is a polymorphic type

	IthreshPaperSheetDetector(const IthreshPaperSheetDetector&) = default;
	IthreshPaperSheetDetector(IthreshPaperSheetDetector&&) = default;

	IthreshPaperSheetDetector& operator = (const IthreshPaperSheetDetector&) = delete;
	IthreshPaperSheetDetector& operator = (IthreshPaperSheetDetector&&) = delete;

private:
	
	virtual std::unique_ptr<AbstractQuadDetector> createClone() const override;

	virtual std::vector<std::vector<cv::Point>> detectCandidates(const cv::Mat& image) const override;

	int thresholdLevels;    
};	// IthreshPaperSheetDetector


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


// A paper sheet detector based on a dominant saturation-value pair
class SavaldoPaperSheetDetector : public AbstractPaperSheetDetector
{
public:
	SavaldoPaperSheetDetector() = default;	

protected:
	
	// Restrict copy/move operations since it's a polymorphic type

	SavaldoPaperSheetDetector(const SavaldoPaperSheetDetector&) = default;
	SavaldoPaperSheetDetector(SavaldoPaperSheetDetector&&) = default;

	SavaldoPaperSheetDetector& operator = (const SavaldoPaperSheetDetector&) = delete;
	SavaldoPaperSheetDetector& operator = (SavaldoPaperSheetDetector&&) = delete;

private:

	virtual std::unique_ptr<AbstractQuadDetector> createClone() const override;

	virtual std::vector<std::vector<cv::Point>> detectCandidates(const cv::Mat& image) const override;
    
};	// SavaldoPaperSheetDetector


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


/****************************************************************************************************************************************
 * 
 * A document scanner class takes in an image of a document and performs perspective correction
 * 
 * **************************************************************************************************************************************/

class DocumentScanner
{
public:
	DocumentScanner(const std::string& windowName, std::unique_ptr<AbstractPaperSheetDetector> paperDetector = std::make_unique<IthreshPaperSheetDetector>());

	// Copying is not allowed to prevent multiple instances using the same window.
	// Move operations change the address of the document scanner pointer, which is passed to setMouseCallback. Therefore any function that needs 
	// mouse handling has to call setMouseCallback after move. We also have to take care that the old handler doesn't get called any longer. 
	// The ptDragged is a non-owning pointer to the selected vertex from the bestQuad vector. This address is supposed to stay the same after move, 
	// but we reset it in the beginning of the prepare function anyway.
	DocumentScanner(const DocumentScanner& other) = delete;
	DocumentScanner(DocumentScanner&& other) = default;

	DocumentScanner& operator = (const DocumentScanner& other) = delete;
	DocumentScanner& operator = (DocumentScanner&& other) = default;

	const std::string& getWindowName() const noexcept { return this->windowName; }
	void setWindowName(const std::string& windowName) { this->windowName = windowName; }

	bool isViewInvariant() const noexcept { return this->viewInvariant; }
	void setViewInvariantMode(bool viewInvariant) noexcept { this->viewInvariant = viewInvariant; }

	void setDetector(std::unique_ptr<AbstractPaperSheetDetector> detector) { this->paperDetector = std::move(detector); }

	cv::Mat rectify(const cv::Mat& src, std::vector<cv::Point> &quad, int width, int height);

	bool prepare(const cv::Mat& src, std::vector<cv::Point> &quad);

	bool display(const cv::Mat &image);

private:

	static void onMouseEvent(int event, int x, int y, int flags, void *userData);

	void drawSelection();

	std::vector<cv::Point2f> arrangeVerticesClockwise(const std::vector<cv::Point> &quad);

	constexpr static int minPointRadius = 3;
	constexpr static int minLineWidth = 1;

	std::string windowName;
	std::unique_ptr<AbstractPaperSheetDetector> paperDetector; // = std::make_unique<IthreshPaperSheetDetector>();
	//cv::String windowName;
	bool viewInvariant = true;
	cv::Mat src, srcDecorated;
	std::vector<cv::Point> bestQuad;
	cv::Point* ptDragged = nullptr;		// raw pointers are fine if the pointer is non-owning	
};	// DocumentScanner

DocumentScanner::DocumentScanner(const std::string& windowName, std::unique_ptr<AbstractPaperSheetDetector> paperDetector)
	: windowName(windowName)
	, paperDetector(std::move(paperDetector))
{
}

bool DocumentScanner::prepare(const cv::Mat& src, std::vector<cv::Point>& quad)
{	
	this->src = src;   // the source image is used by the mouse handler
	this->ptDragged = nullptr;     // reset the selected point

	// Detect the paper sheet boundaries
	this->bestQuad = this->paperDetector->detect(src);

    // It would generally be better to use RAII to guarantee that mouse handling gets disabled even if an exception is raised, 
    // but since the only function (except this one) where a user can click the window is display, it is easier just to disable 
    // mouse handling there.
	//struct WindowRAII
	//{
	//	WindowRAII(const std::string &windowName, DocumentScanner *scanner)
	//		: windowName(windowName)
	//	{
	//		// A mouse handler will let the user move the vertices in case our automatic selection was not correct
	//		cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
	//		cv::setMouseCallback(windowName, &DocumentScanner::onMouseEvent, scanner);
	//	}

	//	~WindowRAII()
	//	{
	//		cv::destroyWindow(windowName);
	//	}

	//	const std::string& windowName;
	//} windowRAII(this->windowName, this);


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

	cv::setMouseCallback(this->windowName, nullptr);	// disable mouse handling to prevent unwanted attempts to move the vertices 
	
	quad = this->bestQuad;     // a user might have moved the points

	return (key & 0xFF) != 27;	// not escape
}	// prepare

cv::Mat DocumentScanner::rectify(const cv::Mat& src, std::vector<cv::Point>& quad, int width, int height)
{
	CV_Assert(!src.empty());
	CV_Assert(quad.size() == 4);
    CV_Assert(width > 0 && height > 0);

	cv::namedWindow(this->windowName, cv::WINDOW_AUTOSIZE);

	// In order to perform perspective correction, the source and destination points must be consistently ordered (clockwise order is used)
	std::vector<cv::Point2f> srcQuadF = arrangeVerticesClockwise(quad);

	// In a view-invariant mode the width refers to the horizontal dimension of a correctly aligned document as seen in
	// a frontal view, i.e. it doesn't depend on how the document is actually positioned in the input image. Otherwise, 
	// the width measures the dimension which is closer to horizontal in the input image. The same applies to the height.
	bool rotateImage = false;
	if (this->viewInvariant)
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


	// Compute angles between u0 pointing upwards from the top-left vertex and the vectors from the top-left vertex to each other vertex
	std::vector<double> angles(quad.size());
	std::transform(quad.begin(), quad.end(), angles.begin(), [&p0 = *accTopLeft.second, u0 = cv::Point(0, -1)](const auto& p) {
		cv::Point u = p - p0;	// the vector from p0 to p

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
	cv::setMouseCallback(this->windowName, nullptr);	// make sure that mouse handling is disabled
	cv::imshow(this->windowName, image);
	return (cv::waitKey() & 0xFF) != 27;
}   // display

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


void printUsage()
{    
	std::cout << "Usage: doscan [-h]"
				 " --input=<input image file>" 
				 " [--output=<output file>]"
				 " [--view_invariant=<true or false>]"
				 " [--width=<a positive integer or zero>]"
                 " [--height=<a positive integer or zero>]"
                 " [--aspect_ratio=<a positive float>]"
				 " [--paper_detector=<1 - Ithresh, 2 - Savaldo>]"
				 " [--threshold_levels=<integer (1..255)>]"
				 " [--min_area_pct=<float (0..max_area_pct)>]"
				 " [--max_area_pct=<float (min_area_pct..1)>]"
				 " [--approx_accuracy_pct=<float (0..1)>]" << std::endl;
}


int main(int argc, char* argv[])
{
	try
	{
        static const cv::String keys =
			"{help h usage ?        |                     | Print the help message  }"
			"{input                 |<none>               | The file path of the image to be rectified }"
			"{output                |                     | If not empty, specifies the output file path where the rectified image will be saved to }"
			"{view_invariant        |true                 | Determines whether the document's aspect ratio should be treated as view-invariant }"
            "{width                 |500                  | The rectified document's width (if zero, it is deduced from the height and the aspect ratio) }"
            "{height                |0                    | The rectified document's height (if zero, it is deduced from the width and the aspect ratio) }"
            "{aspect_ratio          |0.7071               | The rectified document's aspect ratio (unused if both width and height are specified) }"
			"{paper_detector        |1                    | The algorithm to be used for paper sheet detection (1 - Ithresh, 2 - Savaldo) }"
			"{threshold_levels      |15                   | The number of threshold levels for the Ithresh paper sheet detector }"
			"{min_area_pct          |0.5                  | The minimal fraction of the original image that the paper sheet must occupy to be considered for detection (0..max_area_pct) }"
			"{max_area_pct          |0.99                 | The maximal fraction of the original image that the paper sheet can occupy to be considered for detection (min_area_pct..1) }"
			"{approx_accuracy_pct   |0.02                 | The accuracy of contour approximation with respect to the contour length (0..1) }";
		
		cv::CommandLineParser parser(argc, argv, keys);
		parser.about("Document Scanner\n(c) Yaroslav Pugach");

		if (parser.has("help"))
		{
			printUsage();
			return 0;
		}

		std::string inputFile = parser.get<std::string>("input");
		std::string outputFile = parser.get<std::string>("output");
		bool viewInvariant = parser.get<bool>("view_invariant");
        int width = parser.get<int>("width");
        int height = parser.get<int>("height");
        double aspectRatio = parser.get<double>("aspect_ratio");
        int paperDetectorIdx = parser.get<int>("paper_detector");
        int thresholdLevels = parser.get<int>("threshold_levels");
        double minAreaPct = parser.get<double>("min_area_pct");
        double maxAreaPct = parser.get<double>("max_area_pct");
        double approxAccuracyPct = parser.get<double>("approx_accuracy_pct");
        		
		if (!parser.check())
		{
			parser.printErrors();
			printUsage();
			return -1;
		}

		
        // Load the input image
        
		cv::Mat imgInput = cv::imread(inputFile, cv::IMREAD_COLOR);	// EXIF is important 
		if (imgInput.empty())
            throw std::runtime_error("Failed to load the input image from \"" + inputFile + "\". Make sure this file exists.");

        
		// Deduce the width and height from the aspect ratio
		
		if (width == 0)
        {
            width = static_cast<int>(height * aspectRatio);            
        }
        
        if (height == 0 && aspectRatio > 0)
        {
            height = static_cast<int>(width / aspectRatio);            
        }
        
        
        // Create a paper sheet detector
        
        std::unique_ptr<AbstractPaperSheetDetector> paperSheetDetector;
        switch (paperDetectorIdx)
        {
            case 1:
                paperSheetDetector = std::make_unique<IthreshPaperSheetDetector>(thresholdLevels);
                break;
            case 2:
                paperSheetDetector = std::make_unique<SavaldoPaperSheetDetector>();
                break;
            default:
                throw std::invalid_argument("Incorrect paper detection algorithm specified. Supported options: 1 - Ithresh, 2 - Savaldo.");
        }   // switch
        
        paperSheetDetector->setAreaPctRange(minAreaPct, maxAreaPct);
        paperSheetDetector->setApproximationAccuracyPct(approxAccuracyPct);
        
        
        // Set up the document scanner 
        
		DocumentScanner scanner(inputFile, std::move(paperSheetDetector));	
		scanner.setViewInvariantMode(viewInvariant);
        
		std::vector<cv::Point> quad;
		if (scanner.prepare(imgInput, quad))
		{                       
            cv::Mat imgOutput = scanner.rectify(imgInput, quad, width, height);
			
            if (!outputFile.empty())
                cv::imwrite(outputFile, imgOutput);
            
			scanner.display(imgOutput);            
		}

		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}
