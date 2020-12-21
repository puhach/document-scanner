#include <iostream>
#include <vector>
#include <algorithm>
#include <exception>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int main(int argc, char* argv[])
{
	try
	{
		cv::Mat imSrc = cv::imread("./images/scanned-form.jpg", cv::IMREAD_UNCHANGED);	// TODO: not sure what reading mode should be used
		cv::imshow("Input", imSrc);
		cv::waitKey();

		cv::destroyAllWindows();
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}
