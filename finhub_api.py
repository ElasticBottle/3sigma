import requests
import api_keys

r = requests.get(
    "https://finnhub.io/api/v1/crypto/candle?symbol=BINANCE:BTCUSDT&resolution=D&from=1572651390&to=1575243390&token="
)
print(r.json())
