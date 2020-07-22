from pandas_datareader import data
import yfinance as yf
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Data:
    # ! TODO(EB): Add way to control resolution of data, currently only 1 hour
    # ! TODO(EB): Add way to store multiple tick data.
    def __init__(self, ticker: Union[str, list], start: datetime, end: datetime):
        super().__init__()
        yf.pdr_override()
        self.data_df = data.get_data_yahoo(ticker, start, end)
        self.ticker, self.start, self.end = ticker, start, end

    def show(self, start: datetime = None, end: datetime = None):
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
            (self.data_df.index > start) & (self.data_df.index < end)
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

        fig.add_trace(
            go.Bar(x=date_range, y=self.data_df["Volume"]), row=2, col=1,
        )
        # Add overlay for average volume

        fig.update(layout_xaxis_rangeslider_visible=True)
        fig.update_layout(height=600, width=1000, title_text=f"Graph of {self.ticker}")
        fig.show()

    def data_stats(self):
        print(self.data_df.head())
        print(self.data_df.index)
        print(self.data_df.columns)
        print(self.data_df.describe())
        print(self.data_df.isnull().sum())


aapl = Data("LRN", datetime.datetime(2018, 1, 1), datetime.datetime(2020, 1, 1))
aapl.data_stats()
aapl.show()
