#include <iostream>
#include <string>
#include <memory>
//#include <vector>
//#include <exception>
//
#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/calib3d.hpp>
//#include <opencv2/highgui.hpp>

#include "documentscanner.h"
#include "ithreshpapersheetdetector.h"
#include "savaldopapersheetdetector.h"



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
