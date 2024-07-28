import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime

# NLTK VADER sözlüğünü indir
nltk.download('vader_lexicon')

# VADER duygu analiz aracını oluştur
sia = SentimentIntensityAnalyzer()

# CSV verisini oku
data = pd.read_csv('PuzzleBinance/csvs_news/crypto_news_sentiments.csv')

# Her bir haberin duygu puanını hesapla ve yeni bir sütun ekle
data['Sentiment_Score'] = data['Description'].apply(lambda description: sia.polarity_scores(description)['compound'])

# Pozitif, negatif ve nötr duygu puanlarını ayıkla
data['Sentiment_Label'] = data['Sentiment_Score'].apply(lambda score: 'Pozitif' if score > 0 else ('Negatif' if score < 0 else 'Nötr'))

# Analiz sonuçlarını yeni bir CSV dosyasına kaydet
data.to_csv('PuzzleBinance/csvs_news/analyzed_coin_news.csv', index=False)

# Belirli coin isimlerini içeren bir liste oluştur
coins = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Ripple (XRP)', 'Dogecoin', 'Polkadot', 'Uniswap', 'Litecoin', 'Chainlink']

# PDF raporu oluşturma fonksiyonu
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Coin News Sentiment Analysis Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_coin_section(self, coin, positive_count, negative_count, neutral_count, positive_percent, negative_percent, neutral_percent):
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'{coin} Analysis', 0, 1, 'C')
        
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f'Positive news count: {positive_count}', 0, 1)
        self.cell(0, 10, f'Negative news count: {negative_count}', 0, 1)
        self.cell(0, 10, f'Neutral news count: {neutral_count}', 0, 1)
        self.cell(0, 10, f'Positive news percentage: {positive_percent:.2f}%', 0, 1)
        self.cell(0, 10, f'Negative news percentage: {negative_percent:.2f}%', 0, 1)
        self.cell(0, 10, f'Neutral news percentage: {neutral_percent:.2f}%', 0, 1)

        self.image(f'PuzzleBinance/charts/{coin}_sentiment_pie_chart.png', x=10, y=self.get_y(), w=100)

    def add_general_section(self, total_positive, total_negative, total_neutral, total_positive_percent, total_negative_percent, total_neutral_percent):
        self.add_page()
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'General Market Analysis', 0, 1, 'C')
        
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f'Total positive news count: {total_positive}', 0, 1)
        self.cell(0, 10, f'Total negative news count: {total_negative}', 0, 1)
        self.cell(0, 10, f'Total neutral news count: {total_neutral}', 0, 1)
        self.cell(0, 10, f'Total positive news percentage: {total_positive_percent:.2f}%', 0, 1)
        self.cell(0, 10, f'Total negative news percentage: {total_negative_percent:.2f}%', 0, 1)
        self.cell(0, 10, f'Total neutral news percentage: {total_neutral_percent:.2f}%', 0, 1)

        self.image('PuzzleBinance/charts/general_sentiment_pie_chart.png', x=10, y=self.get_y(), w=100)

# PDF raporunu başlat
pdf = PDFReport()

# Genel analiz için toplam sayılar
total_positive = 0
total_negative = 0
total_neutral = 0

# Her coin için analiz yap ve görseller oluştur
for coin in coins:
    coin_news = data[data['Title'].str.contains(coin, case=False)]
    
    if coin_news.empty:
        continue

    positive_news = coin_news[coin_news['Sentiment_Score'] > 0]
    negative_news = coin_news[coin_news['Sentiment_Score'] < 0]
    neutral_news = coin_news[coin_news['Sentiment_Score'] == 0]

    positive_count = len(positive_news)
    negative_count = len(negative_news)
    neutral_count = len(neutral_news)

    total_positive += positive_count
    total_negative += negative_count
    total_neutral += neutral_count

    total_count = positive_count + negative_count + neutral_count
    positive_percent = (positive_count / total_count) * 100 if total_count > 0 else 0
    negative_percent = (negative_count / total_count) * 100 if total_count > 0 else 0
    neutral_percent = (neutral_count / total_count) * 100 if total_count > 0 else 0

    # Duygu dağılımını görselleştir
    labels = ['Pozitif', 'Negatif', 'Nötr']
    sizes = [positive_count, negative_count, neutral_count]
    colors = ['green', 'red', 'gray']
    explode = (0.1, 0, 0)  # Pozitif haber dilimini biraz ayır

    # Geçersiz değerleri kontrol et ve düzelt
    sizes = [size if size > 0 else 0 for size in sizes]
    if sum(sizes) == 0:
        sizes = [1, 1, 1]  # Eğer tüm değerler 0 ise, geçici bir çözüm olarak her dilimi eşit yap

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Eşit oranlı pasta grafiği
    plt.title(f'{coin} Haber Duygu Dağılımı')
    plt.savefig(f'PuzzleBinance/charts/{coin}_sentiment_pie_chart.png')
    plt.close()

    # PDF'e ekle
    pdf.add_coin_section(coin, positive_count, negative_count, neutral_count, positive_percent, negative_percent, neutral_percent)

# Genel analiz görselleştirmesi
total_count = total_positive + total_negative + total_neutral
total_positive_percent = (total_positive / total_count) * 100 if total_count > 0 else 0
total_negative_percent = (total_negative / total_count) * 100 if total_count > 0 else 0
total_neutral_percent = (total_neutral / total_count) * 100 if total_count > 0 else 0

labels = ['Pozitif', 'Negatif', 'Nötr']
sizes = [total_positive, total_negative, total_neutral]
colors = ['green', 'red', 'gray']
explode = (0.1, 0, 0)

# Geçersiz değerleri kontrol et ve düzelt
sizes = [size if size > 0 else 0 for size in sizes]
if sum(sizes) == 0:
    sizes = [1, 1, 1]  # Eğer tüm değerler 0 ise, geçici bir çözüm olarak her dilimi eşit yap

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Eşit oranlı pasta grafiği
plt.title('Genel Haber Duygu Dağılımı')
plt.savefig('PuzzleBinance/charts/general_sentiment_pie_chart.png')
plt.close()

# Genel analiz PDF'e ekle
pdf.add_general_section(total_positive, total_negative, total_neutral, total_positive_percent, total_negative_percent, total_neutral_percent)

# PDF raporunu kaydet
pdf_output = f'PuzzleBinance/news_report_pdf/coin_news_sentiment_report_{datetime.datetime.now().strftime("%Y-%m-%d")}.pdf'
pdf.output(pdf_output)

print(f"Duygu analizi tamamlandı ve sonuçlar '{pdf_output}' ve 'PuzzleBinance/csvs_news/analyzed_coin_news.csv' dosyalarına kaydedildi.")
