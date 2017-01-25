'''Lab 6: Text Processing Review and Adjacency Matrices

Please add your code where indicated. You may conduct a superficial test of
your code by executing this file in a python interpreter.

'''
import re
import numpy

def argmax(sequence):
    """Return the index of the highest value in a list.

    This is a warmup exercise.

    Args:
        sequence (list): A list of numeric values.

    Returns:
        int: The index of the highest value in `sequence`.

    """
    # YOUR CODE HERE
    max_val = 0
    indx  = 0

    for idx,val in enumerate(sequence,start = 0):
        if val > max_val:
            max_val = val
            indx = idx

    return indx


def tokenize(string, lowercase=False):
    """Extract words from a string containing English words.

    Handling of hyphenation, contractions, and numbers is left to your
    discretion.

    Tip: you may want to look into the `re` module

    Args:
        string (str): A string containing English.
        lowercase (bool, optional): Convert words to lowercase.

    Returns:
        list: A list of words.

    """
    # YOUR CODE HERE
    if lowercase:
       return string.lower().split()
    else:
       return string.split()

def text2wordfreq(string, lowercase=False):
    """Calculate word frequencies for a text written in English.

    Handling of hyphenation and contractions is left to your discretion.

    Your function should make use of the `tokenize` function above.

    Args:
        string (str): A string containing English.
        lowercase (bool, optional): Convert words to lowercase before calculating their
            frequency.

    Returns:
        dict: A dictionary with words as keys and frequencies as values.

    """
    # YOUR CODE HERE

    tokens = tokenize(string,lowercase)
    hist = {}

    for key in tokens:
        if key in hist:
            hist[key] += 1
        else:
            hist[key] = 1

    return hist


def lexical_density(string):
    """Calculate the lexical density of a string containing English words.

    The lexical density of a sequence is defined to be the number of
    unique words divided by the number of total words. The lexical
    density of the sentence "The dog ate the hat." is 4/5.

    Ignore capitalization. For example, "The" should be counted as the same
    type as "the".

    This function should use the `text2wordfreq` function.

    Args:
        string (str): A string containing English.

    Returns:
        float: Lexical density.

    """
    # YOUR CODE HERE

    hist = text2wordfreq(string,True)
    return len(hist.keys())/sum(hist.values())


def hashtags(string):
    """Extract hashtags from a string.

    For example, the string `"RT @HouseGOP: The #StateOfTheUnion isn't
    strong."` contains the hashtag `#StateOfTheUnion`.

    Args:
        string (str): A string containing English.

    Returns:
        list: A list, possibly empty, containing hashtags.

    """
    return [htag.strip("#") for htag in string.split() if htag.startswith("#")]



def jaccard_similarity(text1, text2):
    """Calculate Jaccard Similarity between two texts.

    The Jaccard similarity (coefficient) or Jaccard index is defined to be the
    ratio between the size of the intersection between two sets and the size of
    the union between two sets. In this case, the two sets we consider are the
    set of words extracted from `text1` and `text2` respectively.

    This function should ignore capitalization. A word with a capital
    letter should be treated the same as a word without a capital letter.

    You are encouraged to use the `text2wordfreq` function.

    Args:
        text1 (str): A string containing English words.
        text2 (str): A string containing English words.

    Returns:
        float: Jaccard similarity

    """
    # YOUR CODE HERE
    setA = set(text1.lower().split())
    setB = set(text2.lower().split())

    return len(setA & setB)/len(setA | setB)

def adjacency_matrix_from_edges(pairs):
    """Construct and adjacency matrix from a list of edges.

    An adjacency matrix is a square matrix which records edges between vertices.

    This function turns a list of edges, represented using pairs of comparable
    elements (e.g., strings, integers), into a square adjacency matrix.

    For example, the list of pairs ``[('a', 'b'), ('b', 'c')]`` defines a tree
    with root node 'b' which may be represented by the adjacency matrix:

    ```
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]]
    ```

    where rows and columns correspond to the vertices ``['a', 'b', 'c']``.

    Vertices should be ordered using the usual Python sorting functions. That
    is vertices with string names should be alphabetically ordered and vertices
    with numeric identifiers should be sorted in ascending order.

    Args:
        pairs (list of [int] or list of [str]): Pairs of edges

    Returns:
        (array, list): Adjacency matrix and list of vertices. Note
            that this function returns *two* separate values, a Numpy
            array and a list.

    """
    # YOUR CODE HERE
    import numpy as np

    edges = list(set([node for pair in pairs for node in pair]))
    edges.sort()

    matrixSize = len(edges)
    adjMatrix = np.zeros((matrixSize,matrixSize))

    for row,col in pairs:
        i = edges.index(row)
        j = edges.index(col)
        adjMatrix[i,j] = 1

    adjMatrix = np.array(adjMatrix)
    return (adjMatrix + adjMatrix.transpose(),edges)




# DO NOT EDIT CODE BELOW THIS LINE

import unittest

import numpy as np


class TestLab06(unittest.TestCase):

    def test_argmax(self):
        self.assertEqual(argmax([0, 1, 2]), 2)
        self.assertEqual(argmax([3, 1, 2]), 0)

    def test_tokenize(self):
        words = tokenize('Colorless green ideas sleep furiously.', True)
        self.assertIn('green', words)
        self.assertIn('colorless', words)
        words = tokenize('The rain  in spain is  mainly in the plain.', False)
        self.assertIn('The', words)
        self.assertIn('rain', words)

    def test_text2wordfreq(self):
        counts = text2wordfreq('Colorless green ideas sleep furiously. Green ideas in trees.', True)
        self.assertEqual(counts['green'], 2)
        self.assertEqual(counts['sleep'], 1)
        self.assertIn('colorless', counts)
        self.assertNotIn('hello', counts)

    def test_lexical_density(self):
        self.assertAlmostEqual(lexical_density("The cat"), 1)
        self.assertAlmostEqual(lexical_density("The cat in the hat."), 4/5)

        tweet = """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work. #SOTU http://t.co/aa7FWRCdyn"""
        self.assertEqual(len(hashtags(tweet)), 2)

    def test_jaccard_similarity(self):
        text1 = "Eight million Americans"
        text2 = "Americans in the South"
        self.assertAlmostEqual(jaccard_similarity(text1, text2), 1/6)

    def test_hashtags(self):
        tweet = """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work. #SOTU http://t.co/aa7FWRCdyn"""
        self.assertEqual(len(hashtags(tweet)), 2)

    def test_adjacency_matrix_from_edges(self):
        pairs = [('a', 'b'), ('b', 'c')]
        expected = np.array(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
        A, nodes = adjacency_matrix_from_edges(pairs)
        self.assertEqual(nodes, ['a', 'b', 'c'])
        np.testing.assert_array_almost_equal(A, expected)


if __name__ == '__main__':
    unittest.main()
