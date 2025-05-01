import data.fetch_data as fetch_data
df = fetch_data.DataFetcher().fetch_klines()
print(f"{df[-1]['timestamp']}\t{df[-1]['close']}")