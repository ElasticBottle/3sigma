#%%
import datetime
import time
from typing import *
from functools import partial
import concurrent.futures
import pandas as pd

import api_keys
from data import Finnhub, FinnhubInterval, StocksDataGen


heavy_weight_tickers = [
    "MSFT",
    "AAPL",
    "GOOG",
    "AMZN",
    "FB",
]

silicon_tickers = [
    "AMD",
    "INTC",
    "NVDA",
    "QRVO",
    "AVGO",
    "MU",
]
commerce_tickers = [
    "SHOP",
    "WIX",
    "SQ",
    "PYPL",
    "ETSY",
]

card_tickers = [
    "V",
    "MA",
]

bank_tickers = [
    "AXP",
    "WFC",
    "BAC",
    "JPM",
    "GS",
    "SCHW",
    "AMTD",
    "ETFC",
    "MS",
]

train_tick = heavy_weight_tickers + silicon_tickers + commerce_tickers + card_tickers
valid_tick = bank_tickers


def label_distribution(df: pd.Series):
    total = len(df)
    return df.value_counts() / total * 100


def to_csv(row, ticker: str, train: bool):
    base_dir = (
        "D:\Datasets\stock_data\\1d\\train\\"
        if train
        else "D:\Datasets\stock_data\\1d\\valid\\"
    )
    date_start = row["data"].index.min()
    date_end = row["data"].index.max()
    filename = (
        f"{base_dir}{row['label']}\\{ticker}{date_start.date()} {date_end.date()}.csv"
    )
    row["data"].to_csv(filename)


def save_1d_data(raw_data: Dict[str, pd.DataFrame], train: bool):
    for ticker, data_collection in raw_data.items():
        with concurrent.futures.ThreadPoolExecutor() as ex:
            start = time.time()
            ex.map(
                lambda x: partial(to_csv, ticker=ticker, train=train)(x[1]),
                data_collection.iterrows(),
            )
            end = time.time()
            print(f"{end - start:.2f}")


def generate_data_files(tickers: List[str], train: bool = True):
    data_files = {}
    for ticker in tickers:
        resolution = FinnhubInterval.H
        data = Finnhub(api_keys.finnhub_api)
        data.get_stock_data(
            ticker=ticker,
            resolution=resolution,
            start=datetime.datetime(2019, 7, 11, 9, 0, 0),
            end=datetime.datetime(2020, 8, 14, 10, 0, 0),
        )
        generator = StocksDataGen(data.get_cache(ticker, resolution))
        generator.gen_data(
            lookback=datetime.timedelta(days=14),
            lookforward=datetime.timedelta(weeks=1),
        )
        data_files[ticker] = generator.labels
        print(f"{ticker}, {len(generator.labels)} entries")
        print(
            f"""Distribution of labels: 
{label_distribution(generator.labels.loc[:, 'label'])}"""
        )
    # save_1d_data(data_files, train=train)


def generate_1d(train: List[str], valid: List[str]):
    generate_data_files(train, train=True)
    generate_data_files(valid, train=False)


generate_1d(train_tick, valid_tick)
