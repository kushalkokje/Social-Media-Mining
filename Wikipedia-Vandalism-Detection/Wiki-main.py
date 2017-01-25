"""Assignment 5: Vandalism Detection

Please add your code where indicated by "YOUR CODE HERE". You may conduct a
superficial test of your code by executing this file in a python interpreter.

This assignment asks you to predict the class membership of a Wikipedia edit.
The edit is either vandalism or an edit made in good faith. For this assignment
you may ignore the possibility of edge cases.

This assignment requires interacting with the English Wikipedia's Mediawiki API
so you will need a connection to the Internet. Part of the assignment involves
learning to use the Mediawiki API.

"""

import requests

WIKIPEDIA_API_ENDPOINT = 'https://en.wikipedia.org/w/api.php'


def page_ids(titles):
    """Look up the Wikipedia page ids by title.

    For example, the Wikipedia page id of "Albert Einstein" is 736. (This page
    has a small page id because it is one of the very first pages on
    Wikipedia.)

    A useful reference for the Mediawiki API is this page:
    https://www.mediawiki.org/wiki/API:Info

    Args:
        titles (list of str): List of Wikipedia page titles.

    Returns:
        list of int: List of Wikipedia page ids.

    """
    # The following lines of code (before `YOUR CODE HERE`) are suggestions
    params = {
        'action': 'query',
        'prop': 'info',
        'titles': '|'.join(titles),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
    }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    # YOUR CODE HERE
    page_id = [i['pageid'] for i in payload['query']['pages']]
    return page_id
    pass


def page_lengths(ids):
    """Find the length of a page according to the Mediawiki API.

    A page's length is measured in bytes which, for English-language pages, is
    approximately equal to the number of characters in the page's source.

    A useful reference for the Mediawiki API is this page:
    https://www.mediawiki.org/wiki/API:Info

    Args:
        ids (list of str): List of Wikipedia page ids.

    Returns:
        list of int: List of page lengths.

    """
    # YOUR CODE HERE

    page_id_str = [str(i) for i in ids]

    params = {
    'action': 'query',
    'prop': 'info',
    'pageids': '|'.join(page_id_str),
    'format': 'json',
    'formatversion': 2,  # version 2 is easier to work with
     }

    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    page_length = [rec['length'] for rec in payload['query']['pages']]
    return page_length
    pass


def recent_revision_ids(id, n):
    """Find the revision ids of recent revisions to a single Wikipedia page.

    The Wikipedia page is identified by its page id and only the `n` most
    recent revision ids are returned.

    hRttps://www.mediawiki.org/wiki/API:evisions

    Args:
        id (int): Wikipedia page id
        n (int): Number of revision ids to return.

    Returns:
        list of int: List of length `n` of revision ids.

    """
    # YOUR CODE HERE

    params = {
        'action': 'query',
        'prop': 'revisions',
        'pageids':id,
        'format': 'json',
        'rvlimit':n,
        'formatversion': 2,  # version 2 is easier to work with
    }

    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()

    rev_ids = [rec['revid'] for rec in payload['query']['pages'][0]['revisions']]
    return rev_ids

    pass


def revisions(revision_ids):
    """Fetch the content of revisions.

    Revisions are identified by their revision ids.

    https://www.mediawiki.org/wiki/API:Revisions

    Args:
        revision_ids (list of int): Wikipedia revision ids

    Returns:
        list of str: List of length `n` of revision contents.

    """
    # YOUR CODE HERE

    rev_ids = [str(id) for id in revision_ids]

    params = {
            'action': 'query',
            'prop': 'revisions',
            'revids':'|'.join(rev_ids),
            #'pageids': '736',
            'format': 'json',
            #'rvlimit':10,
            'rvprop' : 'content',
            'formatversion': 2,  # version 2 is easier to work with
            }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()

    revision_contents = [rec['content'] for rec in payload['query']['pages'][0]['revisions']]

    return revision_contents
    pass


# RECOMMENDED EXERCISES
#
# The following exercises are recommended.
#

def is_vandalism(revision_id):
    """Classify a revision as vandalism or non-vandalism.

    For example, revision 738515242 to the page `Indiana` adds the text
    "INdiana best stae ever so ye". This is an uncomplicated case of vandalism.
    This change may be viewed at the following url:

    https://en.wikipedia.org/w/index.php?title=Indiana&diff=738515242&oldid=737943373

    Tip: You may want to compare the parent revision to the current revision.

    Args:
        revision_id (int): Wikipedia revision id

    Returns:
        bool: `True` if revision is vandalism, otherwise `False`.

    """
    # YOUR CODE HERE

    import difflib as df
    import sys
    import nltk
    import string

    revision_ids = []
    revision_ids.append(revision_id)

    # get parent id for the revision_id parameter
    params = {
            'action': 'query',
            'prop': 'revisions',
            'revids': revision_id,
            'format': 'json',
            'rvprop' : 'ids|comment',
            'formatversion': 2,  # version 2 is easier to work with
            }

    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()

    # Get the users revision comment for the revision id.
    # Length of the comment is an important feature in classifying the spam wiki editors
    rev_comment = payload['query']['pages'][0]['revisions'][0]['comment']
    rev_comment_len = len(rev_comment.split())

    parentid = payload['query']['pages'][0]['revisions'][0]['parentid']


    # append the parent id to the lst revision_ids
    revision_ids.append(parentid)

    # Get the content of the current and the parent revision
    rev_content = revisions(revision_ids)

    # tokenize the content of the revisions to figure out the difference


    text1 = rev_content[0]
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = text1.translate(translate_table)
    new_rev = nltk.word_tokenize(no_punctuation)
    new_rev = nltk.Text(new_rev)

    text2 = rev_content[1]
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = text2.translate(translate_table)
    old_rev = nltk.word_tokenize(no_punctuation)
    old_rev = nltk.Text(old_rev)

    # use the ndiff function from the difflib module to get the difference in the revisions
    diff = df.ndiff(old_rev, old_rev)

    # parse diff object into a list
    diff = list(diff)

    # Get the changed string in the current revision
    new_str = []
    for ele in diff:
        if '+' in ele:
            new_str.append(ele.replace('+',"").strip())
    len_new_str = len(new_str)

    upper_new_str = [upper(str) for str in new_str]

    # Get the old string from the parent revsion that as been removed
    old_str = []
    for ele in diff:
        if '-' in ele:
            new_str.append(ele.replace('-',"").strip())
    len_old_str = len(old_str)


    # Wiki authentic users usually use more words to decsribe the revision made while spammers use less words as comment.
    # If the length of the comment for the revision is less than 5 words , classify it as case of vandalism
    if rev_comment_len < 5:
        return True
    # If all the words in the revised content are in uppercase classify as case of vandalism
    elif upper_new_str == new_str:
        return True
    # If the edited version has word length less than 7, classify as case of vanadalism
    # Authentic users usually use more words while editing the document
    # For example "Get life losers !!!" having word length as 4 is clearly a case of vandalism]
    elif len_new_str < 7:
        return True
    # more lexical features like vulgar words , non english words in the edited string can be used to identify Vandalism
    # users can add more features as required
    else:
        return False




# DO NOT EDIT CODE BELOW THIS LINE

import unittest


class TestAssignment5(unittest.TestCase):

    def test_page_ids(self):
        titles = ['Albert Einstein']
        ids = [736]
        self.assertEqual(page_ids(titles), ids)

    def test_page_lengths(self):
        ids = [736]
        expected = [137000]  # NOTE: this number changes over time
        lengths = page_lengths(ids)
        self.assertEqual(len(lengths), 1)
        self.assertGreater(lengths[0], 0)
        self.assertGreater(lengths[0], expected[0] >> 3)
        self.assertLess(lengths[0], expected[0] << 3)

    def test_recent_revisions(self):
        n = 3
        id = 736
        revisions = recent_revision_ids(id, n)
        self.assertEqual(len(revisions), n)
        for revid in revisions:
            self.assertGreater(revid, 720000000)

    def test_revisions(self):
        revids = [746929653]  # edit to Albert Einstein
        revisions_ = revisions(revids)
        self.assertEqual(len(revisions_), len(revids))
        self.assertGreater(len(revisions_[0]), 50000)  # page is more than 50000 characters

    def test_is_vandalism(self):
        revision_id = 738515242  # clear case of vandalism
        self.assertTrue(is_vandalism(revision_id))


if __name__ == '__main__':
    unittest.main()
