import cv2
from unittest import TestCase
from Kalman_Filter import Matrix
from Robust_may_13_tuning import HoughDetect_May_13


class TestHoughDetect_May_13(TestCase):
    def test_HoughDetect_May_13(self):
        x = Matrix([[960. * 1], [330. * 1]])  # 960, 330 are close to half of the original image height and width
        P = Matrix([[10000., 0.], [0., 10000.]])
        self.houghDetect = HoughDetect_May_13(cv2.imread('../road.jpg'), x, P, resolution=1)