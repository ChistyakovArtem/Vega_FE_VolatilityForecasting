import pandas as pd
import yfinance as yf
from Project_2.utils.testing.target_functions import BaseTargetFunction

class MarketData:
    def __init__(self, path = None, yahoo_params = None):

        if path is not None and yahoo_params is not None:
            raise ValueError('Only one data source can be provided')

        self.dataframe = None
        self.target = None
        self.ptr = 0
        self.target_name = None
        if path is not None:
            self.dataframe = pd.read_csv(path)
            self.name = path
            return

        if yahoo_params is not None:
            stock = yf.Ticker(yahoo_params['stock'])
            self.dataframe = stock.history(period=yahoo_params['period'], interval=yahoo_params['interval'])
            self.dataframe['price'] = self.dataframe['Close']
            self.name = yahoo_params['stock'] + '_' + yahoo_params['period'] + '_' + yahoo_params['interval']
            return
        
        raise ValueError('No data source provided')

    def set_target(self, target_function: BaseTargetFunction):
        self.target = target_function.get_target(self)
        self.target_name = target_function.name

    def get_next_row_target(self):
        if self.ptr >= len(self.dataframe):
            return None, None
        row = self.dataframe.iloc[self.ptr]
        target = self.target.iloc[self.ptr]
        self.ptr += 1
        return row, target