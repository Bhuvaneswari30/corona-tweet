import gdown

url = "https://docs.google.com/spreadsheets/d/1-_aprlN7ov57q3GBFiWjhQhFSjHpHRnS/export?format=xlsx"
gdown.download(url, "vaccination_tweets.xlsx", quiet=False)

print("Dataset downloaded successfully!")