from Project_2.marketdata.market_data import MarketData
from Project_2.utils.testing.target_functions import BaseTargetFunction

from statsmodels.tsa.arima.model import ARIMA

import numpy as np

class BasePredictionModel:
    pass

class NaivePredictionModel(BasePredictionModel):
    def __init__(self, target_class: BaseTargetFunction):
        self.current_price_value = None
        self.target_class = target_class
        self.name = 'Naive Prediction Model'

        self.test_period = False

    def start_test_period(self):
        self.test_period = True

    def __call__(self, current_row):
        prev_price_value = self.current_price_value
        current_price_value = current_row['price']
        self.current_price_value = current_price_value

        if prev_price_value is None:
            return 0
        else:
            return self.target_class.get_naive_prediction(prev_price_value, current_price_value)

class AR(BasePredictionModel):
    def __init__(self, target_class: BaseTargetFunction, lag: int):
        self.lag = lag
        self.name = 'AR Prediction Model'

        self.test_period = False
        self.prices = []
        self.model = None
        self.model_fit = None

    def start_test_period(self):
        self.test_period = True

    def update_model(self):
        if len(self.prices) >= self.lag:
            abs_price_returns = np.power(np.log(np.array(self.prices[1:]) / np.array(self.prices[:-1])), 2)

            self.model = ARIMA(abs_price_returns, order=(self.lag, 0, 0))
            self.model_fit = self.model.fit()

    def __call__(self, current_row):
        self.prices.append(current_row['price'])

        if not self.test_period:
            return 0
        
        # Обновляем модель только если накопилось достаточно данных
        if len(self.prices) >= self.lag:
            self.update_model()
            forecast = self.model_fit.forecast(steps=1)
            return forecast[0]
        else:
            return 0
        
from arch import arch_model
import numpy as np

class GARCHPredictionModel(BasePredictionModel):
    def __init__(self, target_class: BaseTargetFunction, lag: int):
        self.lag = lag
        self.name = 'GARCH Prediction Model'

        self.test_period = False
        self.prices = []
        self.model = None
        self.model_fit = None

    def start_test_period(self):
        self.test_period = True

    def update_model(self):
        if len(self.prices) >= self.lag:
            price_returns = np.log(np.array(self.prices[1:]) / np.array(self.prices[:-1]))
            
            # Fit a GARCH model
            self.model = arch_model(price_returns, vol='Garch', p=self.lag, q=self.lag)
            self.model_fit = self.model.fit(disp='off')

    def __call__(self, current_row):
        self.prices.append(current_row['price'])

        if not self.test_period:
            return 0
        
        # Update the model only if enough data is available
        if len(self.prices) >= self.lag:
            self.update_model()
            forecast = self.model_fit.forecast(horizon=1)
            return forecast.variance.iloc[-1, 0]  # Return the predicted variance
        else:
            return 0
