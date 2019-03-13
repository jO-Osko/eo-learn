from enum import Enum

import cv2 as cv
import numpy as np
from eolearn.core import EOTask, FeatureType


class AdaptiveThresholdMethod(Enum):
    MEAN = cv.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN = cv.ADAPTIVE_THRESH_GAUSSIAN_C


class SimpleThresholdMethod(Enum):
    BINARY = cv.THRESH_BINARY
    BINARY_INV = cv.THRESH_BINARY_INV
    TRUNC = cv.THRESH_TRUNC
    TOZERO = cv.THRESH_TOZERO
    TOZERO_INV = cv.THRESH_TOZERO_INV


class ThresholdType(Enum):
    BINARY = cv.THRESH_BINARY
    BINARY_INV = cv.THRESH_BINARY_INV


class Thresholding(EOTask):
    """
    Task to compute thresholds of the image using basic and adaptive thresholding methods.
    Depending on the image, we can also use bluring methods that sometimes improve our results.

    With adaptive thresholding we detect edges and with basic thresholding we connect field into one area - segmentation.
    (There is a lot of room for improvment of both methods).

    The task uses methods from cv2 library.
    """

    def __init__(self, input_feature_name, rgb_indices, output_feature_name, base_image_index=0,
                 correction_factor=3.5, simple_th_value=127,
                 simple_th_max_value=255, adaptive_th=AdaptiveThresholdMethod.MEAN,
                 thresh_type=ThresholdType.BINARY, simple_th=SimpleThresholdMethod.BINARY,
                 block_size=11, c=2, mask_th=10, max_value=255, otsu=False):
        """
        :param input_feature_name: Name of the input feature
        :type input_feature_name: str
        :param rgb_indices: Indices corresponding to B,G,R data in input feature
        :type rgb_indices: (int, int, int)
        :param output_feature_name: Name of output feature
        :type output_feature_name: str
        :param base_image_index: Index of image to use for thresholding
        :type base_image_index: int
        :param correction_factor: Correction factor for rgb images
        :type correction_factor: float
        :param adaptive_th: adaptive thresholding method, ADAPTIVE_THRESH_MEAN_C=threshold value is the mean of neighbourhood area
                                                          ADAPTIVE_THRESH_GAUSSIAN_C=threshold value is the weighted sum of neighbourhood values where weights are a gaussian window
        :type adaptive_th: AdaptiveThresholdMethod
        :param thresh_type: thresholding type used in adaptive thresholding
        :type thresh_type: ThresholdType
        :param simple_th: simple thesholding method
        :type simple_th: SimpleThresholdMethod
        :param block_size: it decides the size of neighbourhood area
        :type block_size: int (must be odd)
        :param c: a contstant which is subtracted from mean or weighted mean calculated
        :type c: int
        :param mask_th: which values do we want on our mask
        :type mask_th: int
        :param max_value:
        :type max_value: int
        :param otsu: flag that tells us if we want otsu binarization or not
        :type otsu: bool
        :param simple_th_value: threshold value for simple threshold
        :type simple_th_value: int
        :param simple_th_max_value: maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
        :type simple_th_max_value: int
        """

        self.input_feature_name = input_feature_name
        self.output_feature_name = output_feature_name

        self.rgb_indices = rgb_indices

        self.base_image_index = base_image_index

        self.correction_factor = correction_factor

        self.adaptive_th = adaptive_th

        self.thresh_type = thresh_type

        self.simple_th = simple_th

        self.block_size = block_size
        if not self.block_size % 2:
            raise ValueError("Block size must be an odd number")

        self.c = c

        self.mask_th = mask_th
        if self.mask_th < 0 or self.mask_th > 255:
            raise ValueError("Mask threshold must be between 0 and 255")

        self.max_value = max_value
        if self.max_value > 255 or self.max_value < 0:
            raise ValueError("maxValue must be between 0 and 255")

        self.otsu = int(otsu) * cv.THRESH_OTSU

        self.simple_th_value = simple_th_value
        if simple_th_value > 255 or simple_th_value < 0:
            raise ValueError("simple_th_value must be between 0 and 255")

        self.simple_th_max_value = simple_th_max_value
        if simple_th_max_value > 255 or simple_th_max_value < 0:
            raise ValueError("simple_th_maxValue must be between 0 and 255")

    def execute(self, eopatch):

        img_true_color = eopatch[FeatureType.DATA][self.input_feature_name][self.base_image_index]
        img_true_color = cv.cvtColor(img_true_color[..., self.rgb_indices], cv.COLOR_BGR2RGB)
        img_grayscale = (cv.cvtColor(img_true_color.copy(), cv.COLOR_RGB2GRAY) * 255).astype(np.uint8)

        img_true_color = np.clip(img_true_color*self.correction_factor, 0, 1)

        img2 = img_true_color

        th_adaptiv = cv.adaptiveThreshold(img_grayscale, self.max_value, self.adaptive_th.value, self.thresh_type.value, self.block_size, self.c)

        mask = (th_adaptiv < self.mask_th) * 255

        img2[mask != 0] = (255, 0, 0)

        ret, result = cv.threshold(img2, self.simple_th.value, self.simple_th_max_value, self.thresh_type.value + self.otsu)

        eopatch[FeatureType.DATA_TIMELESS][self.output_feature_name] = result
        n, m, _ = result.shape
        eopatch[FeatureType.DATA_TIMELESS][self.output_feature_name + "_mask"] = mask.reshape((n, m, 1))
        return eopatch


class Bluring:
    AVAILABLE_BLURING_METHODS = {
        'none',
        'medianBlur',
        'GaussianBlur',
        'bilateralFilter'
    }

    def __init__(self, img, sigmaY, borderType, blur_method='none',
                 gKsize=(5, 5), sigmaX=0, mKsize=5, d=9, sigmaColor=75,
                 sigmaSpace=75):
        """
        :param img: a image that will be used
        :type img: 2D array or 3D array
        :param blur_method: image blurring (smoothing) methods
        :type blur method: str

        => GaussianBlur params:
        :patam gKsize: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero's and then they are computed from sigma
        :type gKsize: Size
        :param sigmaX: Gaussian kernel standard deviation in X direction
        :type sigmaX: double
        :param sigmaY: Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height
        :type sigmaY: double
        :param borderType: pixel extrapolation method
        :type borderType: int

        => medianBlur params:
        :param mKsize: aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7...
        :type mKsize: int

        => bilateralFilter params:
        :param d: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace
        :type d: int
        :param sigmaColor: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
        :type sigmaColor: double
        :param sigmaSpace: 	Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace
        :type sigmaSpace: double
        :param borderType: border mode used to extrapolate pixels outside of the image
        :type borderType: int
        """
        self.img = img
        self.blur_method = blur_method
        if (self.blur_method not in self.AVAILABLE_BLURING_METHODS):
            raise ValueError("Bluring method must be one of these: {}".format(
                self.AVAILABLE_BLURING_METHODS))

        self.gKsize = gKsize
        if (self.gKsize % 2 != 1 | self.gKsize != 0 | self.gKsize < 0):
            raise ValueError("gKsize must be odd and positive or 0")
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

        self.mKsize = mKsize
        if (self.mKsize % 2 != 1 | self.mKsize <= 1):
            raise ValueError("mKsize must be odd and greater than 1")

        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.borderType = borderType

    def _blur(self):
        if (self.blur_method == 'bilateralFilter'):
            self.img = cv.bilateralFilter(self.img, self.d, self.sigmaColor,
                                          self.sigmaSpace, self.borderType)
        elif (self.blur_method == 'medianBlur'):
            self.img = cv.medianBlur(self.img, self.mKsize)
        elif (self.blur_method == 'GaussianBlur'):
            self.img = cv.GaussianBlur(self.img, self.gKsize, self.sigmaX,
                                       self.sigmaY, self.borderType)
        return self.img
