#include "abstractquaddetector.h"



std::vector<cv::Point> AbstractQuadDetector::detect(const cv::Mat& image) const
{
	return selectBestCandidate(detectCandidates(image));
}
