#ifndef ITHRESHPAPERSHEETDETECTOR_H
#define ITHRESHPAPERSHEETDETECTOR_H

#include "abstractpapersheetdetector.h"

#include <opencv2/core.hpp>

#include <stdexcept>
#include <vector>
#include <memory>


// A paper sheet quad detector based on iterative thresholding of the saturation and value channels

class IthreshPaperSheetDetector : public AbstractPaperSheetDetector
{
public:
	// constexpr constructors are implicitly inline
	constexpr IthreshPaperSheetDetector(int thresholdLevels = 15)
		: thresholdLevels(thresholdLevels >= 1 && thresholdLevels <= 255 ? thresholdLevels : throw std::invalid_argument("The number of threshold levels must be in range 1..255."))
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


#endif	// ITHRESHPAPERSHEETDETECTOR_H