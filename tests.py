#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import statistics
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_line = []
    min_left_y = []
    right_line = []
    min_right_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1) / (x2-x1)
            if math.fabs(slope) > 0.2:
                # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                if slope > 0:
                    right_line.append(np.polyfit([x1,x2], [y1,y2], 1))
                    min_right_y.append(min(y1, y2))
                else:
                    left_line.append(np.polyfit([x1,x2], [y1,y2], 1))
                    min_left_y.append(min(y1, y2))
    right_line_avg = [statistics.mean([coef[::2][0] for coef in right_line]), statistics.mean([coef[1::2][0] for coef in right_line])] if [coef[::2][0] for coef in right_line] else None
    min_right_y = min(min_right_y) if min_right_y else None
    left_line_avg = [statistics.mean([coef[::2][0] for coef in left_line]), statistics.mean([coef[1::2][0] for coef in left_line])] if [coef[::2][0] for coef in left_line] else None
    min_left_y = min(min_left_y) if min_left_y else None
    if left_line_avg is not None and right_line_avg is not None:
        for slope, intercept, min_y in [[left_line_avg[0], left_line_avg[1], min_left_y], [right_line_avg[0], right_line_avg[1], min_right_y]]:
            cv2.line(img,
                     (int((img.shape[0] - intercept) / slope), img.shape[0]),
                     (int((min_y - intercept) / slope), min_y),
                     color,
                     thickness)
    # cv2.line(img,
    #          ( int((img.shape[0]-left_line_avg[1])/left_line_avg[0]), img.shape[0] ),
    #          ( int((min_left_y-left_line_avg[1])/left_line_avg[0]), min_left_y ),
    #          color,
    #          thickness)
    # cv2.line(img, (x1, img.shape[] y1), (x2, y2), color, thickness)


    # right_x1 = 700;
    # right_x2 = 700;
    # right_y1 = 400;
    # right_y2 = 400;
    #
    # left_x1 = 300;
    # left_x2 = 300;
    # left_y1 = 400;
    # left_y2 = 400;
    #
    # prev_right_x1 = right_x1;
    # prev_right_x2 = right_x2;
    # prev_right_y1 = right_y1;
    # prev_right_y2 = right_y2;
    # prev_left_x1 = left_x1
    # prev_left_y1 = left_y1
    # prev_left_x2 = left_x2
    # prev_left_y2 = left_y2
    #
    # right_avg = 0.63;
    # left_avg = -0.71;
    #
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         slope = ((y2 - y1) / (x2 - x1))
    #         # slope is positive with given origin, right line; right line working
    #         if (slope > 0):
    #
    #             if (slope - right_avg < 0.1):
    #                 right_avg = (slope + right_avg) / 2
    #
    #             if (x1 < right_x1):
    #                 right_x1 = x1
    #
    #             if (x2 > right_x2):
    #                 right_x2 = x2
    #
    #             if (y1 < right_y1):
    #                 right_y1 = y1
    #
    #             if (y2 > right_y2):
    #                 right_y2 = y2;
    #         # slope is negative with given origin, left line
    #         else:
    #             print(slope)
    #             print(left_avg)
    #             if (abs(slope - left_avg) < 0.03):
    #                 left_avg = (slope + left_avg) / 2
    #             else:
    #                 print("using old line")
    #                 left_x1 = prev_left_x1
    #                 left_y1 = prev_left_y1
    #                 left_x2 = prev_left_x2
    #                 left_y2 = prev_left_y2
    #
    #             if (x1 < left_x1):
    #                 left_x1 = x1
    #
    #             if (x2 > left_x2):
    #                 left_x2 = x2
    #
    #             if (y1 > left_y1):
    #                 left_y1 = y1
    #
    #             if (y2 < left_y2):
    #                 left_y2 = y2
    #
    # prev_left_x1 = left_x1;
    # prev_left_x2 = left_x2;
    # prev_left_y1 = left_y1;
    # prev_left_y2 = left_y2;
    # print("\nprinting line: "+str(left_x1)+" "+str(left_x2)+"\n")
    # cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), (255, 0, 0), 10)
    # cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), (255, 0, 0), 10)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    grayed = grayscale(image)
    canned = canny(grayed, 40, 200)
    masked = region_of_interest(canned, [
        np.array([[100, image.shape[0]], [470, 310], [500, 310], [image.shape[1], image.shape[0]]])])
    rho = 1
    theta = np.pi / 180
    threshold = 20
    min_line_length = 40
    max_line_gap = 10
    hough = hough_lines(masked, rho, theta, threshold, min_line_length, max_line_gap)
    weighted = weighted_img(hough, image)
    result = weighted
    return result


white_output = 'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# for filename in os.listdir("test_images/"):
#     image = mpimg.imread('test_images/' + filename)
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
#     grayed = grayscale(image)
#     plt.subplot(1, 2, 2)
#     plt.imshow(grayed, cmap='gray')
#     canned = canny(grayed, 40, 200)
#     plt.imshow(canned)
#     masked = region_of_interest(canned, [
#         np.array([[100, image.shape[0]], [470, 310], [500, 310], [image.shape[1], image.shape[0]]])])
#     plt.imshow(masked, cmap='gray')
#
#     rho = 1
#     theta = np.pi / 180
#     threshold = 20
#     min_line_length = 40
#     max_line_gap = 10
#     # line_image = np.copy(image)*0 #creating a blank to draw lines on
#
#     hough = hough_lines(masked, rho, theta, threshold, min_line_length, max_line_gap)
#     weighted = weighted_img(hough, image)
#     result = weighted
#     mpimg.imsave('test_images_output/' + filename, result)
# plt.imshow(hough)