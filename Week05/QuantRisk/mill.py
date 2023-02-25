import numpy as np
import pandas as pd

def return_calculate(prices,method='DISCRETE',dateColumn='date'):
    prices.set_index('Date',inplace=True)
    lst_prices = prices.shift(1)
    delta = prices/lst_prices

    if method.upper() == 'DISCRETE':
        delta = delta-1.0
    elif method.upper() =='LOG':
        delta = np.log(delta)
    else:
        raise ValueError('Error: {} does not exist', method.upper())

    return delta