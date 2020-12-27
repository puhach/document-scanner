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

* In the *view-dependent* mode the width refers to the dimension of the document which looks more horizontal in the image. Similarly, the height refers to the dimension of the document which looks more vertical in the input image. This mode can be useful in case a user only wants to rectify the document, but doesn't want it to be automatically rotated. It also comes to the rescue when the document in the image is warped too much, which makes it difficult to estimate the aspect ratio and match the sides automatically. 

The aspect ratio is always defined as the ratio of the document's width to the document's height (regardless of which side is larger).

![document scanner](./assets/cover.jpg)

The following algorithms are implemented for detecting a document in the input image:

* *Ithresh*, which performs iterative thresholding of the image channels. It is usually more accurate than the second method, but also more computationally expensive.  

* *Savaldo*, which is based on the idea of segmenting the image by the dominant saturation-value pair. It is generally less reliable than Ithresh, but works pretty well in most cases. Also it is more efficient.

The usage details for these and other settings will be described below.
