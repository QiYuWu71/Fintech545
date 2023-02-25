import unittest
import numpy as np
from Covariance import *
from ES import *
from nonPSD import *
from VaR import *


class TestFunctions(unittest.TestCase):
    
    def test_VaR_ES(self):
        sim_x = np.array([1,2,3,4,5])
        alpha = 0.05
        VaR, ES = VaR_ES(sim_x, alpha)
        self.assertAlmostEqual(VaR, -1.5, places=6)
        self.assertAlmostEqual(ES, -1, places=6)
    def test_mle_t(self):
        parameters = [1, 2, 3]
        x = np.array([1, 2, 3, 4, 5])
        expected_L = 12.01945
        L = mle_t(parameters, x)
        self.assertAlmostEqual(L, expected_L, places=2)

    def test_generalized_t_fitting(self):
        rt = np.array([1, 2, 3, 4, 5])
        sim_n = 100
        sim_t = generalized_t_fitting(rt, sim_n)
        self.assertEqual(sim_t.shape, (sim_n,))

if __name__ == '__main__':
    unittest.main()  
