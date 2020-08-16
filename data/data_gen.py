#%%
import datetime
import random
from typing import *

import pandas as pd


def default_labeller(
    data: pd.DataFrame,
    start_date: datetime,
    current_date: datetime,
    latest_date: datetime,
    lookforward: datetime.timedelta,
) -> Union[str, None]:
    """
    Creates label for a given data point.

    Labels are given in this order:
        - 'Buy' if there is 10% profit in [lookforward] period
        - 'Hold' if there 5% at the end of [lookforward] period
        - 'Short' if the stock dips below 10%
        - 'Sell' Otherwise 

    Args:
        - data (pd.DataFrame): The complete data
        - current_date (datetime): the end of the data point's window
        - start_date(datetime): the start of the data point's window
        - latest_date(datetime): the end date of the whole dataset
        - lookforward (datetime.timedelta): the forward looking period to check

    Returns:
        - str: the label for the particular data point, [None] if it isn't possible to assign a label
    """
    base_price = data.loc[current_date, "Close"]
    hold_price = base_price + 0.05 * base_price
    target_price = base_price + 0.1 * base_price
    sell_price = 0.9 * base_price

    end_date = current_date + lookforward
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
        >= 1
    ):
        return "Buy"
    elif (
        target_df.loc[target_df.index[-1], "Open"] > hold_price
        and target_df.loc[target_df.index[-1], "Close"] > hold_price
    ):
        return "Hold"
    elif (
        len(
            target_df[
                (target_df["Close"] <= sell_price) & (target_df["Open"] <= sell_price)
            ]
        )
        >= 1
    ):
        return "Short"
    else:
        return "Sell"


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

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, labels):
        self.__labels = labels

    def gen_data(
        self,
        lookback: datetime.timedelta,
        lookforward: datetime.timedelta,
        labeller: Callable[[pd.DataFrame], Union[str, None]] = default_labeller,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Generates a labels for data base on a sliding window.

        Args:
            - lookback(datetime.timedelta): The duration that constitutes a singe data point
            - lookforward(datetime.timedelta): The duration to consider in determining a label
            - labeller(Callable): refer to [default_labeller] for more
        Returns:
            - pd.DataFrame: with 'data', and 'label' columns.
        """
        self.lookforward, self.lookback = lookforward, lookback
        local_data = []
        label = []
        earliest = self.data.index.min()
        latest = self.data.index.max()
        for date in self.data.index:
            start_date = date - lookback
            if start_date < earliest:
                continue

            current_data = self.data.loc[start_date:date, :]
            current_label = labeller(self.data, start_date, date, latest, lookforward)

            if (latest - date < 2 * lookforward) and verbose:
                print(date - lookback, date, current_label)

            if current_label is None:
                break
            local_data.append(current_data)
            label.append(current_label)
        result = pd.DataFrame({"data": local_data, "label": label})
        assert len(result) == len(
            self.data.loc[earliest + lookback : latest - lookforward, :]
        )

        self.labels = result
        return result

    def view_labels(self, seed: int = None):
        """
        Dummy method to verify data labelling for default_labeller

        Might remove
        """
        print(self.labels["label"].value_counts())
        if seed is not None:
            random.seed(seed)
        index = random.randint(0, len(self.labels))
        data, label = self.labels.iloc[index, :]
        date = data.index.max()
        if label == "Hold":
            to_check = self.data.loc[date + self.lookforward, :]
        elif label == "Short":
            to_check = self.data.loc[date : date + self.lookforward, :]
            target_price = data.loc[date, "Close"] * 0.9
            to_check = to_check[
                (to_check["Close"] <= target_price) & (to_check["Open"] <= target_price)
            ]
        elif label == "Buy":
            to_check = self.data.loc[date : date + self.lookforward, :]
            target_price = data.loc[date, "Close"] * 1.1
            to_check = to_check[
                (to_check["Open"] >= target_price) & (to_check["High"] > target_price)
            ]
        else:
            to_check = self.data.loc[date : date + self.lookforward, :]
        print(label)
        print(data.loc[date, :])
        print(to_check)
