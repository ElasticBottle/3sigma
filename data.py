#%%
from typing import *
from enum import Enum
from pandas_datareader import data as pdr
import pandas as pd
import mplfinance as mpf
import yfinance as yf
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#%%
class YfInterval(Enum):
    ONE_M = "1m"
    TWO_M = "2m"
    FIVE_M = "5m"
    FIFTEEN_M = "15m"
    THIRTY_M = "30m"
    SIXTY_M = "60m"
    NINETY_M = "90m"
    ONE_H = "1h"
    ONE_D = "1d"
    FIVE_D = "5d"
    ONE_WK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTH = "3mo"


def make_list(to_convert) -> list:
    if isinstance(to_convert, list):
        return to_convert
    elif (
        isinstance(to_convert, str)
        or isinstance(to_convert, int)
        or isinstance(to_convert, bool)
    ):
        return [to_convert]


class Data:

    # ! TODO(EB): Add way to control resolution of data, currently only 1 hour
    # ! TODO(EB): Add way to store multiple tick data.
    def __init__(
        self,
        ticker: Union[str, list],
        start: datetime,
        end: datetime,
        interval: Interval = Interval.ONE_D,
    ):
        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        super().__init__()
        yf.pdr_override()
        end += datetime.timedelta(days=2)
        if (
            interval == Interval.ONE_M
            or interval == Interval.TWO_M
            or interval == Interval.FIVE_M
            or interval == Interval.FIFTEEN_M
            or interval == Interval.THIRTY_M
            or interval == Interval.SIXTY_M
            or interval == Interval.NINETY_M
            or interval == Interval.ONE_H
        ):
            start = end - datetime.timedelta(days=60)
        tickers = " ".join(make_list(ticker))
        self.data_df = yf.download(tickers, start, end, interval=interval.value,)
        self._sort_df()
        self.ticker_info = yf.Tickers(tickers).tickers
        self.ticker, self.start, self.end, self.interval = ticker, start, end, interval

    def _sort_df(self):
        # lrn = aapl.data_df.T.unstack(level=0).loc["LRN", :]
        # msft = aapl.data_df.T.unstack(level=0).loc["MSFT", :]
        # lrn = lrn.unstack()
        # lrn.index.value_counts().sort_index()
        pass

    def show(self, start: datetime = None, end: datetime = None) -> None:
        """
        Displays an interactive graph of the underlying dataset

        Args:
            - start (datetime): the starting date to be diplayed. Earliest available data used if not specified
            - end (datetime): the ending date to be diplayed. latest available data used if not specified
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        if start < self.start:
            print(
                f"starting date of {self.ticker}, {start} is before {self.start}, used {self.start} as initial date instead"
            )
            start = self.start
        if end > self.end:
            print(
                f"Ending date fo {self.ticker}, {end} is after {self.end}, used {self.end} as end date instead"
            )
            end = self.end
        date_range = self.data_df.index[
            (self.data_df.index >= start) & (self.data_df.index <= end)
        ]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        fig.add_trace(
            go.Candlestick(
                x=date_range,
                open=self.data_df["Open"],
                high=self.data_df["High"],
                low=self.data_df["Low"],
                close=self.data_df["Close"],
            ),
            row=1,
            col=1,
        )

        # Add overlay for average volume
        fig.add_trace(
            go.Bar(x=date_range, y=self.data_df["Volume"]), row=2, col=1,
        )

        fig.update(layout_xaxis_rangeslider_visible=True)
        # layout_xaxes_rangeslider_visible=True,
        # rangeselector=dict(
        #     buttons=list(
        #         [
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="YTD", step="year", stepmode="todate"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all"),
        #         ]
        #     )
        # ),
        fig.update_layout(height=600, width=1000, title_text=f"Graph of {self.ticker}")
        fig.show()

    def data_stats(self) -> pd.DataFrame:
        """
        General information about the dataset including:
            * Column Names
            * dtpes
            * missing values
            * unique values

        Return :
        - summary_df (pd.DataFrame) : Contains general info about the dataFrame
        """

        # Make summary dataframe
        summary_df = pd.DataFrame()

        # Input the characteristic of the dataset
        summary_df["Cols"] = self.data_df.columns
        summary_df["Dtypes"] = self.data_df.dtypes.values
        summary_df["Total Missing"] = self.data_df.isnull().sum(axis=0).values
        summary_df["Missing%"] = summary_df["Total Missing"] / len(self.data_df) * 100
        summary_df["Total Unique"] = self.data_df.nunique().values
        summary_df["Unique%"] = summary_df["Total Unique"] / len(self.data_df) * 100

        # Dataset dimension
        print("Dataset dimension :", self.data_df.shape)
        print(summary_df)

        return summary_df


#%%
aapl = Data(
    ["lrn"],
    start=datetime.datetime(2018, 1, 1),
    end=datetime.datetime(2019, 12, 18),
    interval=Interval.THIRTY_M,
)
aapl.data_stats()


# aapl.show()
