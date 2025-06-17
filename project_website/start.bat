@echo off
cls
color b
echo [1/5] Python ortamı kontrol ediliyor...

REM Python yüklü mü kontrol et
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Python yüklü değil. Lütfen Python kurunuz.
    pause
    exit /b
)

echo [2/5] pip ve kütüphaneler kuruluyor...
pip install --upgrade pip
pip install -r requirements.txt

echo [3/5] NLTK veri dosyalari indiriliyor...

REM nltk kaynaklarını indir
python -c "import nltk; \
nltk.download('punkt'); \
nltk.download('stopwords'); \
nltk.download('wordnet'); \
nltk.download('omw-1.4'); \
nltk.download('averaged_perceptron_tagger')"

echo [4/5] Proje baslatiliyor...

REM Buraya çalıştırılacak dosya adını yaz 
python app.py

echo ---
echo Proje tamamlandi. Cikmak icin bir tusa basin...
pause
