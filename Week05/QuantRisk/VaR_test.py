#%%
import unittest
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from VaR import *
#%%
class TestFunctions(unittest.TestCase):
    
    def test_normal_VaR(self):
        data = np.array([1,2,3,4,5])
        alpha = 0.05
        VaR = normal_VaR(data, alpha)
        self.assertAlmostEqual(VaR, 2.32617, places=2)
    
    def test_gen_weight(self):
        X = np.array([1,2,3,4,5])
        lam = 0.5
        w_scale = gen_weight(lam, X)
        for i in range(len(w_scale)):
            self.assertAlmostEqual(w_scale[i], [0.51612903, 0.25806452, 0.12903226, 0.06451613, 0.03225806][i])
    
    def test_normal_VaR_Ew(self):
        X = np.array([1,2,3,4,5])
        alpha = 0.05
        lam = 0.5
        VaR_Ew = normal_VaR_Ew(lam, X, alpha)
        self.assertAlmostEqual(VaR_Ew, 7.07169824, places=2)
    
    def test_mle_t(self):
        x = np.array([1,2,3,4,5])
        parameters = np.array([2,6,3])
        L = mle_t(parameters, x)
        self.assertAlmostEqual(L, 14.232928 , places=2)
    
    def test_t_VaR(self):
        X = np.array([1,2,3,4,5])
        alpha = 0.05
        VaR_t = t_VaR(X, mle_t, alpha)
        self.assertAlmostEqual(VaR_t, -0.6323551797330591, places=6)
    
    def test_AR_VaR(self):
        data = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]])
        n = data.shape[0]
        X = data[0:n-1,0].reshape(-1,1)
        Y = data[1:n,0].reshape(-1,1)
        alpha = 0.05
        VaR_AR = AR_VaR(X, Y, alpha)
        self.assertAlmostEqual(VaR_AR, 8.863069653004056e-16, places=6)
    
    def test_His_VaR(self):
        X = np.array([1,2,3,4,5])
        alpha = 0.05
        VaR_Hs = His_VaR(X, alpha)
        self.assertAlmostEqual(VaR_Hs, -1.0, places=6)

if __name__ == '__main__':
    unittest.main()
