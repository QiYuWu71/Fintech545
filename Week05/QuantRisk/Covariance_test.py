#%%
import pandas as pd
import numpy as np
from scipy.linalg import fractional_matrix_power
from Covariance import *

import unittest
import numpy as np
#%%
class TestFunctions(unittest.TestCase):
    
    def test_gen_weight(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        lam = 0.5
        w_scale = gen_weight(lam,X)
        expected_w_scale = np.array([0.57142857,0.28571429,0.14285714])
        np.testing.assert_array_almost_equal(w_scale, expected_w_scale)

    def test_weight_cov(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        lam = 0.5
        cov = weight_cov(lam,X)
        expected_cov = np.array([[6.42857143, 6.42857143, 6.42857143],
       [6.42857143, 6.42857143, 6.42857143],
       [6.42857143, 6.42857143, 6.42857143]])
        np.testing.assert_array_almost_equal(cov, expected_cov)
    
    def test_EWvar_EWcorr(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        lam = 0.5
        cov = EWvar_EWcorr(X,lam)
        expected_cov = np.array([[6.42857143, 6.42857143, 6.42857143],
       [6.42857143, 6.42857143, 6.42857143],
       [6.42857143, 6.42857143, 6.42857143]])
        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_Pvar_Pcorr(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        cov = Pvar_Pcorr(X)
        expected_cov = np.array([[6., 6., 6.],
       [6., 6., 6.],
       [6., 6., 6.]])
        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_EWvar_Pcorr(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        lam = 0.5
        cov = EWvar_Pcorr(X,lam)
        expected_cov = np.array([[6.42857143, 6.42857143, 6.42857143],
       [6.42857143, 6.42857143, 6.42857143],
       [6.42857143, 6.42857143, 6.42857143]])
        np.testing.assert_array_almost_equal(cov, expected_cov)
    def test_EWcorr(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        lam = 0.97
        expected_output = np.array([[1.        , 2.44967907, 2.44967907],
       [2.44967907, 1.        , 2.44967907],
       [2.44967907, 2.44967907, 1.        ]])
        np.testing.assert_allclose(EWcorr(X, lam), expected_output)
        
    def test_Pvar_EWcorr(self):
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        lam = 0.97
        expected_output = np.array([[ 6.        , 14.69807442, 14.69807442],
       [14.69807442,  6.        , 14.69807442],
       [14.69807442, 14.69807442,  6.        ]])
        np.testing.assert_allclose(Pvar_EWcorr(X, lam), expected_output)
if __name__ == '__main__':
    unittest.main()
