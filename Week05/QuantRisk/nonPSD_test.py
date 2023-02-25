#%%
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import eigh
from numpy.linalg import inv
from nonPSD import *
#%%
class TestFunctions(unittest.TestCase):

    def setUp(self):
        n = 5
        sigma = np.full((n,n),0.9)
        for i in range(n):
            sigma[i,i] = 1
        sigma[0,1] = 0.7357
        sigma[1,0] = 0.7357
        self.sigma = sigma

    def test_chol_psd(self):
        expected_sigma = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.7357    ,  0.67730754,  0.        ,  0.        ,  0.        ],
       [ 0.9       ,  0.3511994 ,  0.25818401,  0.        ,  0.        ],
       [ 0.9       ,  0.3511994 , -0.12913664,  0.22356814,  0.        ],
       [ 0.9       ,  0.3511994 , -0.12913664, -0.22372278,         np.nan]])
        assert_array_almost_equal(chol_psd(self.sigma),expected_sigma)

    def test_near_psd(self):
        expected_sigma = np.array([[1.        , 0.736514  , 0.89430865, 0.90309626, 0.89090188],
       [0.736514  , 1.        , 0.89430865, 0.90309626, 0.89090188],
       [0.89430865, 0.89430865, 1.        , 0.91710358, 0.8777941 ],
       [0.90309626, 0.90309626, 0.91710358, 1.        , 0.90590163],
       [0.89090188, 0.89090188, 0.8777941 , 0.90590163, 1.        ]])
        assert_array_almost_equal(near_psd(self.sigma,0.01),expected_sigma)

    def test_higham(self):
        expected_sigma = np.array([[1.        , 0.73570337, 0.89999783, 0.89999783, 0.89999783],
       [0.73570337, 1.        , 0.89999783, 0.89999783, 0.89999783],
       [0.89999783, 0.89999783, 1.        , 0.90000139, 0.90000139],
       [0.89999783, 0.89999783, 0.90000139, 1.        , 0.90000139],
       [0.89999783, 0.89999783, 0.90000139, 0.90000139, 1.        ]])
        assert_array_almost_equal(higham(self.sigma,0.01,100),expected_sigma)


    def test_simulate_PCA(self):
        expected_sigma = np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
        assert_array_almost_equal(simulate_PCA(self.sigma,0.9,nsim=25000),expected_sigma)



if __name__ == '__main__':
    unittest.main()
