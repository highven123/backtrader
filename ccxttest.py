import requests
print(requests.get("https://api.binance.com/api/v3/exchangeInfo").json())