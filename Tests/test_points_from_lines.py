from unittest import TestCase

from Kalman_Filter import Matrix
from intersection_library import points_from_lines


class TestPoints_from_lines(TestCase):
    def test_points_from_lines(self):
        self.points_from_lines = points_from_lines([], Matrix([[960. * 1], [330. * 1]]))
        self.assertEquals(self.points_from_lines, (960, 330))
