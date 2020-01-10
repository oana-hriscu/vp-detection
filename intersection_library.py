# Module to compute the intersection point given the end points of lines generated by probabilistic Hough Transform

import numpy as np
from AOP import Aspects


# @Aspects.param_validator
from Kalman_Filter import Matrix


def cross_product(x, y):
    res = np.dot(np.matrix([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]), np.array(y).T)
    return np.array(res)


# @Aspects.param_validator
def lines_from_points(points):
    lines = []
    for x1, y1, x2, y2 in points:
        point_1 = np.array([x1, y1, 1])
        point_2 = np.array([x2, y2, 1])
        line = cross_product(point_1, point_2)
        lines.append(line)

    return lines


# @Aspects.param_validator
def points_from_lines(lines, state):
    intersections = []
    for line_right, line_left in lines:
        right_slope = - (line_right[0][0] / line_right[0][1])
        left_slope = - (line_left[0][0] / line_left[0][1])
        intersection = cross_product(line_right[0], line_left[0])
        if intersection[0][2] != 0 and (left_slope != right_slope):
            intersection = intersection / intersection[0][2]
            intersection = np.array(intersection)
            if intersection[0][0] >= 0 and intersection[0][1] >= 0:
                intersections.append(intersection[0])

    if len(intersections) != 0:
        x_coordinates = list(zip(*intersections))[0]
        y_coordinates = list(zip(*intersections))[1]

        v_x = int(np.median(x_coordinates))
        v_y = int(np.median(y_coordinates))

    else:
        v_x = int(state.value[0][0])
        v_y = int(state.value[1][0])

    return v_x, v_y


# print(cross_product(np.array([5, 3, 1]), np.array([4, 2, 1])))
