import requests
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from flask import Flask, render_template, request
import json
import urllib3
import logging
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# NLTK veritabanlarının indirilmesi
nltk.download('punkt')
nltk.download('wordnet')

# Log ayarları - Hataları göreceğiz
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SSL uyarılarını başa bela kapamamız lazımmış
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# SSL sorunlarını çözmek için yapmak lazum
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = 'DEFAULT:@SECLEVEL=1'

app = Flask(__name__, static_folder='static')

NEWS_API_KEY = '281597f62eb44872b2b643b1e79d866c'  # NewsAPI keyim
USE_SAMPLE_DATA = False  # API hatası olursa test veri yazacuk
RSS_FEEDS = [
    'https://www.hurriyet.com.tr/rss/anasayfa',
    'https://www.milliyet.com.tr/rss/sondakika',
    'https://www.sozcu.com.tr/feed/',
    'https://www.milliyet.com.tr/rss/rssnew/dunyarss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/ekonomi.xml',
    'https://www.milliyet.com.tr/rss/rssnew/siyasetrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/magazinrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/gundem.xml',
    'https://www.milliyet.com.tr/rss/rssnew/otomobilrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/teknolojirss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/egitim.xml',
    'https://www.milliyet.com.tr/rss/rssnew/milliyettatilrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/modarss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/guzellikrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/ailerss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/saglik.xml',
    'https://www.milliyet.com.tr/rss/rssnew/yemekrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/diyet.xml',
    'https://www.milliyet.com.tr/rss/rssnew/iliskiler.xml',
    'https://www.milliyet.com.tr/rss/rssnew/dekorasyonrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/yasamrss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/astrolojirss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/sondakikarss.xml',
    'https://www.milliyet.com.tr/rss/rssnew/yazarlarrss.xml',
    'https://www.ensonhaber.com/rss/ensonhaber.xml',
    'https://www.ensonhaber.com/rss/mansetler.xml',
    'https://www.ensonhaber.com/rss/gundem.xml',
    'https://www.ensonhaber.com/rss/politika.xml',
    'https://www.ensonhaber.com/rss/ekonomi.xml',
    'https://www.ensonhaber.com/rss/dunya.xml',
    'https://www.ensonhaber.com/rss/saglik.xml',
    'https://www.ensonhaber.com/rss/otomobil.xml',
    'https://www.ensonhaber.com/rss/kultur-sanat.xml',
    'https://www.ensonhaber.com/rss/teknoloji.xml',
    'https://www.ensonhaber.com/rss/medya.xml',
    'https://www.ensonhaber.com/rss/yasam.xml',
    'https://www.ensonhaber.com/rss/kralspor.xml',
    'https://www.ensonhaber.com/rss/3-sayfa.xml',
    'https://www.ensonhaber.com/rss/magazin.xml',
    'https://www.ensonhaber.com/rss/kadin.xml',
    'https://www.sozcu.com.tr/feeds-rss-category-resmi-ilanlar',
    'https://www.sozcu.com.tr/feeds-rss-category-2024-paris-olimpiyatlari',
    'https://www.sozcu.com.tr/feeds-haberler',
    'https://www.sozcu.com.tr/feeds-son-dakika',
    'https://www.sozcu.com.tr/feeds-rss-category-dunya',
    'https://www.sozcu.com.tr/feeds-rss-category-egitim',
    'https://www.sozcu.com.tr/feeds-rss-category-sozcu',
    'http://www.bbc.co.uk/turkce/index.xml',
    'http://www.bbc.co.uk/turkce/ekonomi/index.xml',
    'http://www.bbc.co.uk/turkce/izlenim/index.xml',
    'http://www.bbc.co.uk/turkce/ozeldosyalar/index.xml',
    'http://www.bbc.co.uk/turkce/multimedya/index.xml',
    'https://www.ntv.com.tr/son-dakika.rss',
    'https://www.ntv.com.tr/gundem.rss',
    'https://www.ntv.com.tr/turkiye.rss',
    'https://www.ntv.com.tr/egitim.rss',
    'https://www.ntv.com.tr/teknoloji.rss',
    'https://www.ntv.com.tr/otomobil.rss',
    'https://www.ntv.com.tr/yasam.rss',
    'https://www.ahaber.com.tr/rss/gundem.xml',
    'https://www.ahaber.com.tr/rss/ekonomi.xml',
    'https://www.ahaber.com.tr/rss/teknoloji.xml',
    'https://www.ahaber.com.tr/rss/turkiye-kupasi.xml',
    'https://www.ahaber.com.tr/rss/galeri/anasayfa.xml',
    'https://www.ahaber.com.tr/rss/galeri/gundem.xml',
    'https://haberglobal.com.tr/rss',
]

# Lemmatization ve Stemming işlemleri
ps = PorterStemmer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word, wordnet.VERB) for word in tokens])

def stem_text(text):
    tokens = word_tokenize(text)
    return " ".join([ps.stem(word) for word in tokens])

# Metin normalleştirme fonk.
def normalize_text(text, use_stemming=False, use_lemmatization=False):
    if not text:
        return ""
    # Küçük harfe çevircek
    text = text.lower()
    # Noktalama işaretlerini temizlecuk
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # Fazla boşlukları temizlecuk
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization ve Stemming eklenmesi
    if use_lemmatization:
        text = lemmatize_text(text)
    elif use_stemming:
        text = stem_text(text)
    
    return text

# Örnek veriler gelirse ha bu yazacak
def get_sample_news():
    logger.info("Genişletilmiş örnek haber verisi kullanılıyor...")
    sample_news = [
        {
            'title': '!!!Test verisi gardaş API den gelmee',
            'source': {'name': 'TEESTT!'},
            'publishedAt': '2025-05-03T09:30:00Z',
            'url': 'https://example.com/ekonomi-haberleri'
        }
    ]
    return sample_news

# API'den haber başlıklarını alacak
def fetch_news_from_api():
    if USE_SAMPLE_DATA:
        logger.info("Örnek veri kullanılıyor...")
        return get_sample_news()
        
    try:
        url = f'https://newsapi.org/v2/everything?q=turkiye&language=tr&pageSize=100&sortBy=publishedAt&apiKey={NEWS_API_KEY}'
        logger.info(f"NewsAPI'ye istek gönderiliyor: {url}")
        
        response = requests.get(url, timeout=15, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            if 'articles' in data and data['articles']:
                logger.info(f"Başarılı API yanıtı: {len(data['articles'])} makale alındı")
                return data['articles']
            else:
                logger.warning(f"API yanıtında makale bulunamadı. Yanıt: {data}")
                return get_sample_news()
        else:
            logger.error(f"API Hatası: {response.status_code} durum kodu alındı. Yanıt: {response.text}")
            return get_sample_news()
    except Exception as e:
        logger.error(f"API bağlantı hatası: {str(e)}")
        return get_sample_news()

# RSS feed'lerinden haber başlıklarını alacak
def fetch_news_from_rss():
    all_rss_news = []
    try:
        for rss_url in RSS_FEEDS:
            logger.info(f"RSS kaynağına istek gönderiliyor: {rss_url}")
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                all_rss_news.append({
                    'title': entry.title,
                    'source': {'name': feed.feed.title},
                    'publishedAt': entry.published,
                    'url': entry.link
                })
    except Exception as e:
        logger.error(f"RSS feed hatası: {str(e)}")
    
    return all_rss_news

# API ve RSS'yi birleştirerek haberleri almak
def fetch_news():
    api_news = fetch_news_from_api()
    rss_news = fetch_news_from_rss()
    all_news = api_news + rss_news
    return all_news

# İyileştirilmiş benzer haber başlıklarını bulacak
def find_similar_titles(keyword, threshold, use_stemming=False, use_lemmatization=False):
    logger.info(f"Anahtar kelime arama: '{keyword}', Eşik: {threshold}%")
    news_articles = fetch_news()
    similar_titles = []
    
    if not news_articles:
        logger.warning("Haber makaleleri bulunamadı.")
        return []
    
    # Anahtar kelimeyi normalize eder
    normalized_keyword = normalize_text(keyword, use_stemming, use_lemmatization)
    logger.info(f"Normalize edilmiş anahtar kelime: '{normalized_keyword}'")
    
    # Tüm başlıkları normalize et
    titles = [article['title'] for article in news_articles if 'title' in article and article['title']]
    if not titles:
        logger.warning("Başlık bulunamadı.")
        return []
        
    normalized_titles = [normalize_text(title, use_stemming, use_lemmatization) for title in titles]
    logger.info(f"İşlenecek başlık sayısı: {len(titles)}")
    
    # TF-IDF kullanarak benzer başlıkları tespit et
    try:
        vectorizer = TfidfVectorizer()
        all_texts = [normalized_keyword] + normalized_titles
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        keyword_vector = tfidf_matrix[0:1]
        title_vectors = tfidf_matrix[1:]
        cosine_similarities = cosine_similarity(keyword_vector, title_vectors).flatten()
        
        for i, similarity in enumerate(cosine_similarities):
            similarity_percentage = float(similarity * 100)
            if similarity_percentage >= threshold:
                similar_titles.append({
                    'title': titles[i],
                    'source': news_articles[i]['source']['name'],
                    'published_at': news_articles[i]['publishedAt'],
                    'url': news_articles[i]['url'],
                    'similarity': round(similarity_percentage, 2)
                })
        
        similar_titles = sorted(similar_titles, key=lambda x: x['similarity'], reverse=True)
        logger.info(f"TF-IDF ile bulunan benzer başlık sayısı: {len(similar_titles)}")
    except Exception as e:
        logger.error(f"Vektörleştirme hatası: {str(e)}")
    
    return similar_titles

# Başlıklar arası benzerliği tespit etme fonksiyonu
def detect_duplicate_headlines(threshold=75, use_stemming=False, use_lemmatization=False):
    logger.info(f"Çoğaltılmış başlık tespiti başlatıldı. Eşik: {threshold}%")
    news_articles = fetch_news()
    duplicates = []
    
    if not news_articles:
        logger.warning("Haber makaleleri bulunamadı.")
        return []
    
    titles = [article['title'] for article in news_articles if 'title' in article and article['title']]
    if not titles:
        logger.warning("Başlık bulunamadı.")
        return []
        
    normalized_titles = [normalize_text(title, use_stemming, use_lemmatization) for title in titles]
    logger.info(f"İşlenecek başlık sayısı: {len(titles)}")
    
    # Jaccard benzerliği ile karşılaştırma
    for i in range(len(titles)):
        for j in range(i+1, len(titles)):
            words1 = set(normalized_titles[i].split())
            words2 = set(normalized_titles[j].split())
            
            if not words1 or not words2:
                continue
                
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            jaccard_similarity = len(intersection) / len(union) * 100
            
            if jaccard_similarity >= threshold:
                duplicates.append({
                    'title1': titles[i],
                    'source1': news_articles[i]['source']['name'],
                    'title2': titles[j],
                    'source2': news_articles[j]['source']['name'],
                    'similarity': round(jaccard_similarity, 2)
                })
    
    if not duplicates:
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(normalized_titles)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            for i in range(len(titles)):
                for j in range(i+1, len(titles)):
                    similarity_percentage = float(similarity_matrix[i][j] * 100)
                    if similarity_percentage >= threshold:
                        duplicates.append({
                            'title1': titles[i],
                            'source1': news_articles[i]['source']['name'],
                            'title2': titles[j],
                            'source2': news_articles[j]['source']['name'],
                            'similarity': round(similarity_percentage, 2)
                        })
            
            logger.info(f"TF-IDF ile bulunan benzer başlık çifti sayısı: {len(duplicates)}")
        except Exception as e:
            logger.error(f"Benzerlik hesaplama hatası: {str(e)}")
    
    duplicates = sorted(duplicates, key=lambda x: x['similarity'], reverse=True)
    return duplicates

# Ana sayfa rotası
@app.route("/", methods=["GET", "POST"])
def home():
    similar_titles = []
    duplicates = []
    search_performed = False
    
    # API ve RSS'den haberleri çek
    news_articles = fetch_news()
    
    # Sayfaya geldiğinde tüm haberleri görüntüle
    all_headlines = []
    if news_articles:
        for article in news_articles:
            if 'title' in article and article['title']:
                all_headlines.append({
                    'title': article['title'],
                    'source': article['source']['name'] if 'source' in article and 'name' in article['source'] else 'Bilinmeyen Kaynak',
                    'published_at': article['publishedAt'] if 'publishedAt' in article else '',
                    'url': article['url'] if 'url' in article else '#'
                })
    
    if request.method == "POST":
        search_performed = True
        action = request.form.get("action")
        logger.info(f"POST işlemi alındı. Eylem: {action}")
        
        if action == "search":
            keyword = request.form.get("keyword", "")
            threshold = float(request.form.get("threshold", 30))  # Eşik değerini düşürdük
            use_stemming = 'stemming' in request.form
            use_lemmatization = 'lemmatization' in request.form
            
            if keyword:
                similar_titles = find_similar_titles(keyword, threshold, use_stemming, use_lemmatization)
            else:
                logger.warning("Hata: Anahtar kelime sağlanmadı.")
        
        elif action == "detect_duplicates":
            threshold = float(request.form.get("duplicate_threshold", 75))  # Eşik değerini düşürdük
            use_stemming = 'stemming' in request.form
            use_lemmatization = 'lemmatization' in request.form
            
            duplicates = detect_duplicate_headlines(threshold, use_stemming, use_lemmatization)
    
    return render_template("index.html", 
                          similar_titles=similar_titles, 
                          duplicates=duplicates, 
                          search_performed=search_performed,
                          all_headlines=all_headlines)

if __name__ == "__main__":
    app.run(debug=True)

def fetch_news():
    api_news = fetch_news_from_api()
    rss_news = fetch_news_from_rss()
    
    # API ve RSS verilerini birleştir
    all_news = api_news + rss_news
    
    logger.info(f"Toplam {len(all_news)} haber toplandı.")
    return all_news
