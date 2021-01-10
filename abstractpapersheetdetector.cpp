#include "abstractpapersheetdetector.h"



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
