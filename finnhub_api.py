import datetime
import json
import time

import pandas as pd
import requests

import api_keys as key


class Finnhub:
    """
    Allows user to retrieve bitcoin data from finnhub.io
    """

    values = {
        "1_minute": 0,
        "5_minute": 1,
        "15_minute": 2,
        "30_minute": 3,
        "hourly": 4,
        "daily": 5,
    }
    resolutions = ["1", "5", "15", "30", "60", "D"]

    def __init__(self):
        super().__init__()

    def get_data(self, resolution, from_date, to_date):
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
            from_date and to_date are 8 hours ahead due to the timezone difference between SG and GMT +0 I believe

        """
        from_date = int(datetime.datetime.timestamp(from_date))
        to_date = int(datetime.datetime.timestamp(to_date))
        r = requests.get(
            "https://finnhub.io/api/v1/crypto/candle?symbol=BINANCE:BTCUSDT&resolution="
            + self.resolutions[self.values[resolution]]
            + "&from="
            + str(from_date)
            + "&to="
            + str(to_date)
            + "&token="
            + key.finnhub_api
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
