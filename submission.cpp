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

	std::vector<cv::Point2f> arrangeVertices(const std::vector<cv::Point> &quad);

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
	std::vector<cv::Point2f> srcQuadF = arrangeVertices(quad);

	// In a view-invariant mode the width refers to the horizontal dimension of a correctly aligned document as seen in
	// a frontal view, i.e. it doesn't depend on how the document is actually positioned in the input image. Otherwise, 
	// the width measures the side of the warped document which connects its top-left vertex to the top-right vertex as 
	// it is shown in the image. The height is defined in a similar way.
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

std::vector<cv::Point2f> DocumentScanner::arrangeVertices(const std::vector<cv::Point>& quad)
{
	CV_Assert(quad.size() == 4);

	// Pick the topmost vertex as a starting point. Since we measure the angles from the vector pointing upwards from this point, 
	// the angle corresponding to this vertex will be zero. And we must ensure that every next vertex we reach in a clockwise order
	// is connected to the previous one. Therefore, the upwards vector must never point inside the quad. 
	auto topmostIt = std::min_element(quad.begin(), quad.end(), [](const auto& p1, const auto& p2) {
			return p1.y < p2.y;
		});

	// Compute distances (Manhattan) to all the vertices from the starting point
	std::vector<int> dist(quad.size());
	std::transform(quad.begin(), quad.end(), dist.begin(), [&p0 = *topmostIt](const auto& p) {
		return abs(p.x - p0.x) + abs(p.y - p0.y);
	});

	// Compute angles between u0 pointing upwards from the topmost vertex and the vectors from the topmost vertex to each other vertex
	std::vector<double> angles(quad.size());
	std::transform(quad.begin(), quad.end(), angles.begin(), [&p0 = *topmostIt, u0 = cv::Point(0, -1)](const auto& p) {
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


	// Find the angle to the most distant vertex
	std::vector<int> indices(quad.size());
	std::iota(indices.begin(), indices.end(), 0);
	double mostDistantAngle = accumulate(indices.begin(), indices.end(), 0.0,
		[&angles, &dist, maxDist = *max_element(dist.begin(), dist.end())](double acc, int idx) {
			return dist[idx] == maxDist ? std::max(acc, angles[idx]) : acc;
		});

	// Determine whether the angle to the most distant vertex is also the largest angle
	bool largest = none_of(angles.begin(), angles.end(), [mostDistantAngle](double ang) {
			return ang > mostDistantAngle;
		});


	// Arrange the vertices in the clockwise order starting from the topmost vertex
	std::sort(indices.begin(), indices.end(), [&angles, mostDistantAngle, largest, &dist](int idx1, int idx2) {
		double a1 = angles[idx1], a2 = angles[idx2];
		if (a1 < a2)
			return true;
		else if (a1 > a2)
			return false;
		else // same angles
		{
			// Choose the closer one if the angle is less than the angle to the most distant vertex.
			// Choose the distant one if the angle is greater than the angle to most distant vertex.
			if (a1 < mostDistantAngle)
				return dist[idx1] < dist[idx2];
			else if (a1 > mostDistantAngle)
				return dist[idx1] > dist[idx2];
			else
			{
				// In case the angle to the vertex is the same as the angle to the most distant vertex, 
				// pick the distant vertex first if there is no other vertex with a larger angle.
				// Otherwise, pick the closer one.
				if (largest)
					return dist[idx1] > dist[idx2];
				else
					return dist[idx1] < dist[idx2];
			}   // a1 == a2 == most distant angle
		}   // same angles
		});


	// Find the best circular shift
	int bestShift = 0;
	long long sign[4][2] = { {+1, +1}, {-1, +1}, {-1, -1}, {+1, -1} };	// +1 - minimize, -1 - maximize
	long long minDist = std::numeric_limits<long long>::max();	// min total distance to the corners
	for (int shift = 0; shift < quad.size(); ++shift)
	{
		// Compute the distances (positive or negative) to the top-left corner: a smaller distance from the right/bottom
		// corresponds to a larger distance from the left/top 
		long long d = 0;
		for (int i = 0; i < quad.size(); ++i)
		{
			auto shiftedIdx = indices[(i + shift) % quad.size()];
			d += sign[i][0] * quad[shiftedIdx].x;
			d += sign[i][1] * quad[shiftedIdx].y;
		}

		// We want the total distance from vertices to the corresponding corners to be minimal
		if (d < minDist)
		{
			bestShift = shift;
			minDist = d;
		}
	}	// for shift

	// Fianlly, arrange the vertices according to the determined indices and the best circular shift
	std::vector<cv::Point2f> quadF;
	quadF.reserve(quad.size());
	for (int i = 0; i < indices.size(); ++i)
	{
		quadF.push_back(quad[indices[(i+bestShift)%quad.size()]]);
	}

	return quadF;
}	// arrangeVertices

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
