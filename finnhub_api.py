import requests
import api_keys as key
import datetime
import time


class Finnhub:

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
        """
        from_date = str(int(datetime.datetime.timestamp(from_date)))
        to_date = str(int(datetime.datetime.timestamp(to_date)))
        print("from Date:", from_date, "to_date: ", to_date)
        r = requests.get(
            "https://finnhub.io/api/v1/crypto/candle?symbol=BINANCE:BTCUSDT&resolution="
            + self.resolutions[self.values[resolution]]
            + "&from="
            + from_date
            + "&to="
            + to_date
            + "&token="
            + key.finnhub_api
        )
        print(r.json())

