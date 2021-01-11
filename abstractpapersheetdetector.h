#ifndef ABSTRACTPAPERSHEETDETECTOR_H
#define ABSTRACTPAPERSHEETDETECTOR_H


#include "abstractquaddetector.h"

#include <opencv2/core.hpp>

#include <vector>
#include <stdexcept>


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

	// Since constexpr constructors are implicitly inline, the definition is provided here to avoid linker errors (an inline function shall be defined 
	// in every translation unit in which it is odr-used and shall have exactly the same definition in every case)
	// https://stackoverflow.com/questions/16219711/undefined-symbols-for-constexpr-function
	// https://stackoverflow.com/questions/27345284/is-it-possible-to-declare-constexpr-class-in-a-header-and-define-it-in-a-separat
	constexpr AbstractPaperSheetDetector(double minAreaPct = 0.5, double maxAreaPct = 0.99, double approxAccuracyPct = 0.02)
		: minAreaPct(minAreaPct >= 0 && minAreaPct <= 1 ? minAreaPct : throw std::invalid_argument("Min area percentage must be in range 0..1."))
		, maxAreaPct(maxAreaPct >= minAreaPct && maxAreaPct <= 1 ? maxAreaPct : throw std::invalid_argument("Max area percentage must be in range <min area percentage>..1"))
		, approxAccuracyPct(approxAccuracyPct >= 0 ? approxAccuracyPct : throw std::invalid_argument("Approximation accuracy percentage can't be negative."))
	{	}

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



#endif	// ABSTRACTPAPERSHEETDETECTOR_H