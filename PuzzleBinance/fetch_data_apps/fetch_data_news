import requests
import pandas as pd
from textblob import TextBlob
from newsapi import NewsApiClient
import csv

# News API anahtarınızı buraya girin
api_key = '05d4b1407a6349eca1d88951f1722e77'

# NewsApiClient'i başlat
newsapi = NewsApiClient(api_key=api_key)

# Kripto paralarla ilgili haberleri çekmek için API isteği
response = newsapi.get_everything(
    q='cryptocurrency OR bitcoin OR ethereum OR blockchain OR crypto',
    language='en',  # Haberlerin dili
    sort_by='relevancy',  # Haberleri uygunluk sırasına göre getir
    page_size=100  # Çekmek istediğiniz haber sayısı
)

# Haberlerin analiz edilmesi
articles = response['articles']
news_data = []
for article in articles:
    title = article['title']
    description = article['description']
    url = article['url']
    published_at = article['publishedAt']
    content = title + ' ' + (description if description else '')

    # Sentiment analizi
    analysis = TextBlob(content)
    sentiment = analysis.sentiment.polarity
    news_data.append([title, description, url, published_at, sentiment])

# Verileri DataFrame'e kaydetme
df = pd.DataFrame(news_data, columns=["Title", "Description", "URL", "PublishedAt", "Sentiment"])

# DataFrame'i CSV dosyasına kaydetme
with open("csvs_news/crypto_news_sentiments.csv", "w", newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(df.columns)  # Başlık satırını yazma
    writer.writerows(df.values)  # Veri satırlarını yazma

print("Veriler crypto_news_sentiments.csv dosyasına kaydedildi.")
