import numpy as np
from typing import List
from sklearn.metrics import r2_score, mean_squared_error

from Project_2.marketdata.market_data import MarketData
from Project_2.utils.volatility_models.base_prediction_model import BasePredictionModel
from Project_2.utils.testing.target_functions import BaseTargetFunction

class TestSingleData:
    def __init__(self, marketdata: MarketData, target_function: BaseTargetFunction, prediction_model: BasePredictionModel):
        self.marketdata = marketdata
        self.period_length = len(marketdata.dataframe)
        self.marketdata.set_target(target_function)
        self.prediction_model = prediction_model

    def run(self):
        predictions = []
        targets = []

        cnt = 0
        train_bound = int(self.period_length * 0.8)

        while True:
            row, target = self.marketdata.get_next_row_target()
            if row is None:
                break

            cnt += 1
            if cnt == train_bound:
                self.prediction_model.start_test_period()
            
            prediction = self.prediction_model(row)

            if cnt >= train_bound:
                predictions.append(prediction)
                targets.append(target)

        return predictions, targets
    
    def evaluate(self, predictions, targets):
        experiment_name = f'Data: {self.marketdata.name}, Target: {self.marketdata.target_name}, Model: {self.prediction_model.name}'

        return (experiment_name, np.corrcoef(predictions, targets)[0, 1], len(predictions))

    def run_and_evaluate(self):
        predictions, targets = self.run()
        return self.evaluate(predictions, targets)
    
class TestingPipeline:
    def __init__(self, marketdata_list: List[MarketData], target_function: BaseTargetFunction, prediction_model: BasePredictionModel):
        self.marketdata_list = marketdata_list
        self.target_function = target_function
        self.prediction_model = prediction_model

    def run(self):
        results = []
        for marketdata in self.marketdata_list:
            test_single_data = TestSingleData(marketdata, self.target_function, self.prediction_model)
            results.append(test_single_data.run_and_evaluate())
        return results
