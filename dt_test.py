import unittest
import pprint

from sklearn.datasets import load_iris

import dt


class TestDT(unittest.TestCase):

    iris_dataset = load_iris()

    my_dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]

    my_labels = ['no surfacing', 'flippers']


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_splitDataSet(self):
        subdataset = dt.splitDataSet(self.my_dataset, 0, 1)

        # print(pprint.pformat(subdataset))
        # [[1, 'yes'], [1, 'yes'], [0, 'no']]

        # TODO here

if __name__ == '__main__':
    unittest.main()
