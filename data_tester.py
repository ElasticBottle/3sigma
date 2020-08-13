#%%
from data import *
import api_keys
import datetime

#%%
ticker = "amd"
# get raw data
data = Finnhub(api_keys.finnhub_api)
data.get_stock_data(
    ticker=ticker,
    resolution=FinnhubInterval.H,
    start=datetime.datetime(2019, 7, 11, 9, 0, 0),
    end=datetime.datetime(2020, 8, 14, 10, 0, 0),
)

data.get_cache(ticker, FinnhubInterval.H)
#%%
# generate labelled data
data_gen = StocksDataGen(data.get_cache(ticker, FinnhubInterval.H))
data_gen.gen_data(
    lookback=datetime.timedelta(days=14), lookforward=datetime.timedelta(weeks=1)
)

data_gen.view_labels()

#%%
print(data_gen.labels.loc[: int(0.8 * len(data_gen.labels)), "label"].value_counts())
print(data_gen.labels.loc[int(0.8 * len(data_gen.labels)) :, "label"].value_counts())
print(data_gen.labels.iloc[0]["data"].index.min())
#%%
# Generate plot of labelled data
plotter = OHLCPlot()
index = 0
print(data_gen.labels["label"].iloc[index])
plotter.from_df(data_gen.labels["data"].iloc[index], axis_off=False, save=False)

#%%
