---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/grayed_solidWhiteCurve.jpg "Grayscale"
[image2]: ./test_images_output/canned_solidWhiteCurve.jpg "After Canny transform"
[image3]: ./test_images_output/masked_solidWhiteCurve.jpg "ROI"
[image4]: ./test_images_output/hough_solidWhiteCurve.jpg "Hough transform"
[image5]: ./test_images_output/solidWhiteCurve.jpg "Final image"


---

### Reflection

## Pipeline description

My pipeline consisted of 5 steps. First, I converted the images to grayscale,
![alt text][image1]
then I use Canny transform for detecting edges.
![alt text][image2]
After that, I extract region of interest (not before, because I'd have detected edge of ROI). My region of
interest is trapezoid, that's a bit wider than expected lanes, and it's edges has similar angle to lane lines.
![alt text][image3]
Then I use Hough transform, to get corners of lines. I chose small rho and theta, because I think that pixels that are
connected should have really similar coefficients of line they belong to. I'd like to find loger lines, so I chose a bit
higher min_line_length parameter, but I also know that line can have breaks, so I thought 10 pixels for maximum_gap
shuold be fare enough. And that all of the pipeline, then I draw lines on original image.
![alt text][image4]
![alt text][image5]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculating line
coefficients, than getting rid of lines that are horizontal (small slope), and then I calculate average for lines that
I think are correct. I divided them to left and right line, based on slope sign. Then I make average for each line
separately, and calculate line from highest point for particular line (the smallest y coordinate of all lines belonging
to left or right), to the bottom of the image. Using y values I calculate x values, by having line equation with mean
coefficients.

If you'd like to include images to show how the pipeline works, here is how to include an image: 




## Potential shortcomings with current pipeline


One potential shortcoming would be what would happen when car drives through the curve - I assume that right line goes
from right to left, and when we are in right curve it's not so obvious.

Another shortcoming could be if we drive through lane that's really wide, it could go of the ROI.


## Possible improvements to pipeline

A possible improvement would be to consider another algorithm for drawing line - I assume it's linear, but maybe better
would be to make e.g. a spline, that will allow to draw lines from few parts, that aren't in the same direction.

Another potential improvement could be to choose something other than average for calculating coefficients (it might be
also connected with spline) as it's vulnerable for gross errors.