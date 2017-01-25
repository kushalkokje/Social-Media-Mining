'''Assignment #1

Please add your code where indicated. You may conduct a superficial test of
your code by executing this file in a python interpreter.

'''

def square(n):
    """Calculate the square of a number.

    Args:
        n: An integer or a real number to be squared.

    Returns:
        he square of the number `n`

    """
    # YOUR CODE HERE

    return n**2





# DO NOT EDIT CODE BELOW THIS LINE

import unittest


class TestHW01(unittest.TestCase):

    def test_square(self):
        self.assertEqual(square(5), 25)


if __name__ == '__main__':
    unittest.main()
