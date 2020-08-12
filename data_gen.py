#%%
import datetime
from typing import *

import pandas as pd


def default_labeller(
    data: pd.DataFrame, current_date: datetime, latest_date: datetime
) -> Union[str, None]:
    base_price = data.loc[current_date, "Close"]
    target_price = base_price + 0.1 * base_price
    sell_price = base_price + 0.01 * base_price
    end_date = current_date + datetime.timedelta(weeks=10)
    if end_date > latest_date:
        return None
    # print(current_date, base_price, target_price, sell_price)
    target_df = data.loc[
        current_date + datetime.timedelta(hours=1) : end_date, ["Open", "High", "Close"]
    ]
    if (
        len(
            target_df[
                (target_df["Open"] >= target_price) & (target_df["High"] > target_price)
            ]
        )
        > 1
    ):
        return "buy"
    elif len(
        target_df[
            (target_df["Open"] <= sell_price) & (target_df["Close"] <= sell_price)
        ]
    ):
        return "sell"
    else:
        return "hold"


class StocksDataGen:
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        self.__data = data

    def gen_data(
        self,
        window: datetime.timedelta,
        labeller: Callable[[pd.DataFrame], Union[str, None]] = default_labeller,
    ) -> pd.DataFrame:
        local_data = []
        label = []
        earliest = self.data.index.min()
        latest = self.data.index.max()
        for date in self.data.index:
            if date - window < earliest:
                continue
            current_data = self.data.loc[date - window : date, :]
            current_label = labeller(self.data, date, latest)
            if latest - date < datetime.timedelta(weeks=18):
                print(date, latest, current_label)
            if current_label is None:
                break
            local_data.append(current_data)
            label.append(current_label)
        result = pd.DataFrame({"data": local_data, "label": label})
        return result


#%%
from finnhub_api import Finnhub, FinnHubInterval
from api_keys import finnhub_api

data = Finnhub(finnhub_api)
data.get_stock_data(
    ticker="aapl",
    resolution=FinnHubInterval.H,
    start=datetime.datetime(2019, 12, 11, 9, 0, 0),
    end=datetime.datetime(2020, 3, 15, 10, 0, 0),
)

#%%
data.get_cache("aapl", FinnHubInterval.H)
#%%
aapl_dg = StocksDataGen(data.get_cache("aapl", FinnHubInterval.H))
dataset = aapl_dg.gen_data(datetime.timedelta(days=1))

