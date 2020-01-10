from unittest import TestCase

from intersection_library import lines_from_points


class TestLines_from_points(TestCase):
    def test_lines_from_points(self):
        self.lines_from_points = lines_from_points([(362, 681, 707, 448)])
        self.assertEquals(self.lines_from_points[0][0][0], 233)
        self.assertEquals(self.lines_from_points[0][0][1], 345)
        self.assertEquals(self.lines_from_points[0][0][2], -319291)

