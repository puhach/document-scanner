#ifndef DOCUMENTSCANNER_H
#define DOCUMENTSCANNER_H

#include "ithreshpapersheetdetector.h"

#include <opencv2/core.hpp>

#include <string>
#include <memory>
#include <vector>


// A document scanner class takes in an image of a document and performs perspective correction

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


#endif	// DOCUMENTSCANNER_H