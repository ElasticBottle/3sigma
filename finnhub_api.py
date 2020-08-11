#%%
import datetime
import json
import time
from enum import Enum
from typing import *
import pandas as pd
import numpy as np
import requests

from api_keys import finnhub_api


class FinnHubInterval(Enum):
    ONE_M = "1"
    FIVE_M = "5"
    FIFTEEN_M = "15"
    THIRTY_M = "30"
    H = "60"
    D = "D"
    W = "W"
    M = "M"


class Finnhub:
    """
    Allows user to retrieve bitcoin data from finnhub.io
    """

    def __init__(self, api_key):
        super(Finnhub).__init__()
        self._base_url = "https://finnhub.io/api/v1"
        self.key = api_key
        self.data = {}  # Ticker + interval -> data

    def _to_dateFrame(self, r: Dict[str, int]) -> pd.DataFrame:
        """
        Converts a finnhub response object to a pandas dataframe

        Args:
            - r(dict): contains 'o', 'c', 'h', 'l', 'v', 't', 's' keys from finnHub
        """
        data = pd.DataFrame()
        data["Open"] = r["o"]
        data["Close"] = r["c"]
        data["Low"] = r["l"]
        data["High"] = r["h"]
        data["Volume"] = r["v"]
        data.index = pd.to_datetime(r["t"], unit="s") - datetime.timedelta(hours=4)
        return data

    def _is_valid_time(
        self, res: FinnHubInterval, start: datetime, end: datetime
    ) -> bool:
        if (
            res == FinnHubInterval.ONE_M
            or res == FinnHubInterval.FIVE_M
            or res == FinnHubInterval.THIRTY_M
        ):
            pass

    def get_stock_data(
        self,
        ticker: str,
        resolution: FinnHubInterval,
        start: datetime,
        end: datetime = datetime.datetime.now(),
        delta: int = 12,
        format: str = "json",
        adjusted: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieves stock's historical data from FinnHub's api.

        Note: There is a 1 year limit for free tier users

        Args:
            - ticker (str): the company whose historical data we are interested in.
            resolution (str): 
            - resolution (string): provides the timeframe of the chart that is to be retrieved.
                Available resolutions:
                    - "1_minute"
                    - "5_minute"
                    - "15_minute"
                    - "30_minute"
                    - "hourly"
                    - "daily
            - start (datetime): Provides the start of the timeline to get data for. Timing is based on UTC -4:00
                Format should be datetime.datetime(year, month, day, hour, minute, second)
                Remember to import datetime in the file you're using first
            - end(datetime): Provides the end of the timeline to get data for.
                Format same as from_date.
            - delta (int): The offset from your timezone. The [start] value will get converted from your local machine time to UTC -4:00, use this to override the conversion
            - format (str): either 'json' or 'csv'
            - adjusted (bool): True for adjusted price
        """
        timedelta = datetime.timedelta(hours=delta)
        assert start != end and start < end
        from_date = int(datetime.datetime.timestamp(start + timedelta))
        to_date = int(datetime.datetime.timestamp(end + timedelta))
        r = requests.get(
            self._base_url
            + "/stock/candle?symbol="
            + ticker.upper()
            + "&resolution="
            + str(resolution.value)
            + "&from="
            + str(from_date)
            + "&to="
            + str(to_date)
            + "&token="
            + self.key
        )
        r = json.loads(r.content)
        if r["s"] == "ok":
            self.data[ticker.upper() + str(resolution.value)] = self._to_dateFrame(r)
        else:
            print(json.dumps(r, indent=4))
            raise ValueError(f"Problem fetching {resolution.value} data for {ticker} ")

    def get_cache(self, ticker: str, resolution: FinnHubInterval) -> pd.DataFrame:
        """
        Retrieves previously stored copy of data

        Args:
            - ticker(str): the symbol of the stock
            -resolution(FinnHubInterval): the interval period interested in

        Returns:
            - pd.DataFrame: result of the retrieved data
            - None: if there doesn't exist such data
        """
        return self.data.get(ticker.upper() + str(resolution.value), None)

    def get_crypto_data(self, resolution, from_date, to_date):
        """
        Gets Bitcoin data from the start of from_date to the to_date at the resolution specified.

        Args:
            resolution (string): provides the timeframe of the chart that is to be retrieved.
                Available resolutions:
                    - "1_minute"
                    - "5_minute"
                    - "15_minute"
                    - "30_minute"
                    - "hourly"
                    - "daily
            from_date (datetime): Provides the start of the timeline to get data for.
                Format should be datetime.datetime(year, month, day, hour, minute, second)
                Remember to import datetime in the file you're using first
            to_date(datetime): Provides the end of the timeline to get data for.
                Format same as from_date.
        
        Notes:
            from_date and to_date are 8 hours behind due to the timezone difference between SG and GMT +0 

        """
        from_date = int(datetime.datetime.timestamp(from_date))
        to_date = int(datetime.datetime.timestamp(to_date))
        r = requests.get(
            self._base_url
            + "/crypto/candle?symbol=BINANCE:BTCUSDT&resolution="
            + resolution.value
            + "&from="
            + str(from_date)
            + "&to="
            + str(to_date)
            + "&token="
            + self.key
        ).json()
        if r["s"] == "ok":
            # logging
            print(
                "Gotten data at",
                resolution,
                "resolution\nfrom Date:",
                datetime.datetime.utcfromtimestamp(from_date),
                "to_date: ",
                datetime.datetime.utcfromtimestamp(to_date),
            )
            # removes the status and volume key
            del r["s"]
            del r["v"]
            df = pd.DataFrame(r)
            df.columns = ["close", "high", "low", "open", "time"]
            df[["time"]] = df[["time"]].apply(
                lambda x: datetime.datetime.utcfromtimestamp(x), axis=1
            )
            df.set_index("time", inplace=True)
            return df
        else:
            print(json.dumps(r.json(), indent=4))
            raise ValueError("problem fetching data")


#%%
data = Finnhub(finnhub_api)
data.get_stock_data(
    ticker="aapl",
    resolution=FinnHubInterval.H,
    start=datetime.datetime(2018, 8, 12),
    end=datetime.datetime(2019, 8, 20, 10, 0, 0),
)
#%%
data.get_cache("aapl", FinnHubInterval.H)
