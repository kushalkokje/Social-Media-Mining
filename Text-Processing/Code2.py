'''Assignment #2

Please add your code where indicated. You may conduct a superficial test of
your code by executing this file in a python interpreter.

'''

def distance(x, y):
    """Calculate the Euclidean distance between two vectors.

    Args:
        x: A Numpy array of numeric values.
        y: A Numpy array of numeric values of same length as x.

    Returns:
        The Euclidean distance between the two vectors.

    Raises:
        ValueError: if x and y are invalid (e.g., of different lengths)

    """
    # YOUR CODE HERE

    try:
        return np.sqrt(np.sum(np.square(x - y)))
    except ValueError:
        print("X and Y should be of same length")


def nearest_neighbor(v, points, labels):
    """Classifies vector `v` according to its nearest neighbor.

    Args:
        v: A Numpy array of numeric values of length V.
        points: A 2d-Numpy array of numeric values of dimensional (n, V).
        labels: A Numpy array of string or numeric values of length n.

    Returns:
        The label of the nearest neighbor to `v` in `points`.

    Raises:
        ValueError: if arguments are invalid (e.g., of invalid shapes)

    """
    # YOUR CODE HERE

    if(len(v) != int(points.shape[1]) or int(points.shape[0]) != len(labels) ):
        raise ValueError

    # assign maximum distance possible. Max value of a float type in python
    min_dist = 1.7976931348623157e+308
    point_label = None
    return_val = []

    for val,j in zip(points,range(len(points))):
        dist = 0.0
        for i in range(len(val)):
            dist += (val[i] - v[i])**2
        dist = dist**(1/2)

        print("Distance for label %s is %f " %(labels[j],dist))
        if dist < min_dist:
           return_val = val
           min_dist = dist
           point_label = labels[j]

    print("Minimum distance is %f" %min_dist)
    print("label is %s" %point_label)
    return (return_val)


# DO NOT EDIT CODE BELOW THIS LINE

import unittest
import numpy as np


class TestHW02(unittest.TestCase):

    def test_distance(self):
        x = np.array([4, 3])
        y = np.array([4, 0])
        self.assertEqual(distance(x, y), 3)
        self.assertNotEqual(distance(x, y), 4)
        self.assertIsNotNone(distance(x, y))

        x = np.array([4,5,6])
        y = np.array([[1,2,3],[5,8,9],[1,3,4]])
        z = np.array(['A','B','C'])
        self.assertEqual(list(nearest_neighbor(x,y,z)),list(np.array([1,3,4])))

if __name__ == '__main__':
    unittest.main()
