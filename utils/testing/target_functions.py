import numpy as np

class BaseTargetFunction:
    pass

class NextAbsReturn(BaseTargetFunction):
    def __init__(self):
        self.name = 'Next Abs Return'

    def get_target(self, marketdata):
        return np.abs(np.log(marketdata.dataframe['price'].shift(-1) / marketdata.dataframe['price']).fillna(0))
    
    def get_naive_prediction(self, price_prev, price_curr):
        return np.abs(np.log(price_curr / price_prev))


class NextSquaredReturn(BaseTargetFunction):
    def __init__(self):
        self.name = 'Next Squared Return'

    def get_target(self, marketdata):
        return np.square(np.log(marketdata.dataframe['price'].shift(-1) / marketdata.dataframe['price']).fillna(0))
    
    def get_naive_prediction(self, price_prev, price_curr):
        return np.power(np.log(price_curr / price_prev), 2)