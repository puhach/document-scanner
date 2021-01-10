#ifndef SAVALDOPAPERSHEETDETECTOR_H
#define SAVALDOPAPERSHEETDETECTOR_H


#include "abstractpapersheetdetector.h"

#include <vector>
#include <memory>



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



#endif	// SAVALDOPAPERSHEETDETECTOR_H