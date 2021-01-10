#ifndef ABSTRACTQUADDETECTOR_H
#define ABSTRACTQUADDETECTOR_H


#include <opencv2/core.hpp>

#include <vector>
#include <memory>



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




#endif	// ABSTRACTQUADDETECTOR_H