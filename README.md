# Document Scanner

The automated document scanner application implemented in this project extracts and rectifies the warped document image.

The following assumptions have been made with respect to the input data:

* The document is considerably lighter than the background

* The document is fully contained in the image 

* The document occupies the larger part of the image

* The original document has a rectangular shape

* The aspect ratio of the document is known

It is worth mentioning that the notion of width, height, and aspect ratio can be somewhat ambiguous. For example, when the document in the image is rotated by 90 degrees, one may think of the width and height of the document in its canonical view, but others can treat them as the measures of the horizontal and verticle dimensions as they are actually seen in the image. Besides that, it is not always possible to say what the "true" orientation of the document is. We are likely to know it for a text document, but less likely when it's a diagram. Therefore, the application can operate in two modes:

* In the *view-invariant* mode the width and height refer to the horizontal and vertical dimensions of the document as if it were seen in the most convenient position. The program will try to match the sides of the document according to the aspect ratio of the warped document in the input image and rotate it appropriately. This mode is enabled by default.

* In the *view-dependent* mode the width refers to the side of the document which connects the top-left vertex to the top-right vertex (also the bottom-left and the bottom-right vertices) of the document as it is shown in the image. Similarly, the height refers to the side of the warped document which connects its top-left vertex to the bottom-left one (as well as the top-right and the bottom-right vertices). This mode can be useful in case a user only wants to rectify the document, but doesn't want it to be automatically rotated. It also comes to the rescue when the document in the image is warped too much, which makes it difficult to estimate the aspect ratio and match the sides automatically. 

The aspect ratio is always defined as the ratio of the document's width to the document's height (regardless of which side is larger).

In order to detect a document in the input image the following algorithms are implemented:

* *Ithresh*, which performs iterative thresholding of the image channels. It is usually more accurate than the second method, but also more computationally expensive.  

* *Savaldo*, which is based on the idea of segmenting the image by the dominant saturation-value pair. It is generally less reliable than Ithresh, but works pretty well in most cases. Also it is more efficient.

The usage details for these and other settings will be described below.

## Set Up

It is assumed that OpenCV 4.x, C++17 compiler, and cmake 2.18.12 or newer are installed on the system.

### Specify OpenCV_DIR in CMakeLists

Open CMakeLists.txt and set the correct OpenCV directory in the following line:

```
set(OpenCV_DIR /opt/opencv/4.4.0/installation/lib/cmake/opencv4)
```

Depending on the platform and the way OpenCV was installed, it may be needed to provide the path to cmake files explicitly. On my KUbuntu 20.04 after building OpenCV 4.4.0 from sources the working `OpenCV_DIR` looks like <OpenCV installation path>/lib/cmake/opencv4. On Windows 8.1 after installing a binary distribution of OpenCV 4.2.0 it is C:\OpenCV\build.


### Build the Project

In the root of the project folder create the `build` directory unless it already exists. Then from the terminal run the following:

```
cd build
cmake ..
```

This should generate the build files. When it's done, compile the code:

```
cmake --build . --config release
```


## Usage

The program has to be run from the command line. It takes in the path to the image containing the warped document and several optional parameters: 

```
doscan  --input=<input image file>
		[--output=<output file>]
		[--view_invariant=<true or false>]
		[--width=<a positive integer or zero>]
        [--height=<a positive integer or zero>]
        [--aspect_ratio=<a positive float>]
		[--paper_detector=<1 - Ithresh, 2 - Savaldo>]
		[--threshold_levels=<integer (1..255)>]
		[--min_area_pct=<float (0..max_area_pct)>]
		[--max_area_pct=<float (min_area_pct..1)>]
		[--approx_accuracy_pct=<float (0..1)>]"
	 	[--help]
```

Parameter    | Meaning 
------------ | --------------------------------------
help, ? | Prints the help message.
input | The file path of the image to be rectified.
output | If not empty, specifies the output file path where the rectified image will be saved to.
view_invariant | Determines whether the document's aspect ratio should be treated as view-invariant (true by default). 
width | The rectified document's width (if zero, it is deduced from the height and the aspect ratio). Defaults to 500. 
height | The rectified document's height (if zero, it is deduced from the width and the aspect ratio). Defaults to 0. 
aspect_ratio | The rectified document's aspect ratio (unused if both width and height are specified). Defaults to 0.7071.
paper_detector | The algorithm to be used for paper sheet detection (1 - Ithresh, 2 - Savaldo). Defaults to 1.
threshold_levels | The number of threshold levels for the Ithresh paper sheet detector. Default value is 15.
min_area_pct | The minimal fraction of the original image that the paper sheet must occupy to be considered for detection (0..max_area_pct). Default value is 0.5.
max_area_pct | The maximal fraction of the original image that the paper sheet can occupy to be considered for detection (min_area_pct..1). Default value is 0.99.
approx_accuracy_pct | The accuracy of contour approximation with respect to the contour length (0..1). Default value is 0.02.


Sample usage (linux):
```
./doscan --input=../images/dragon-medium.jpg 
```

The application will detect the document using default parameters. The user may adjust the document boundaries by dragging the vertices. 

![document detection](./assets/detection.jpg)

To rectify the document press any key. 

![rectified document](./assets/rectified.jpg)

Pressing *Escape* will quit the application.
