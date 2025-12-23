🌟 Lumina AI: Yapay Zeka Destekli Akıllı Öneri Sistemi
"Binlerce seçenek, tek bir ışık." > Lumina AI, adını Latince ışık anlamına gelen Lumen kelimesinden alır. Binlerce film ve kitap arasında kaybolduğunuzda, yapay zeka algoritmalarımız size en uygun yolu aydınlatmak için tasarlandı.

✨ Özellikler
Hibrit Öneri Motoru: Hem filmler hem de kitaplar için içerik tabanlı (Content-Based) filtreleme.

Akıllı Arama: NLP teknikleri kullanarak film özetleri, oyuncu kadroları ve kitap yazarları üzerinden benzerlik kurma.

Modern UI/UX: Toz pembe ve lacivert paletiyle tasarlanmış, Apple tarzı ferah ve profesyonel arayüz.

Lumina Favorilerim: Beğendiğiniz içerikleri anlık olarak kaydedebileceğiniz dinamik favori sistemi ve canlı sayaç.

Görsel Katalog: Kitap kapaklarını ve film afişlerini içeren şık kart tasarımları.

🛠️ Kullanılan Teknolojiler
Dil: Python

Arayüz: Streamlit

Veri Bilimi: Pandas, Scikit-learn (TfidfVectorizer, Cosine Similarity)

Görselleştirme: Streamlit-Lottie, Custom CSS

Veri Setleri: TMDB 5000 Movies & Books Dataset

🚀 Kurulum ve Çalıştırma
Projeyi yerel makinenizde çalıştırmak için şu adımları izleyin:

Depoyu Klonlayın:

Bash

git clone https://github.com/kullaniciadin/lumina-ai.git
cd lumina-ai
Gerekli Kütüphaneleri Yükleyin:

Bash

pip install -r requirements.txt
Uygulamayı Başlatın:

Bash

streamlit run main.py
📂 Dosya Yapısı
Plaintext

lumina-ai/
├── data/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── books.csv
├── main.py              # Uygulamanın ana kodu
├── requirements.txt     # Gerekli Python kütüphaneleri
└── README.md            # Proje dökümantasyonu
🧠 Algoritma Nasıl Çalışır?
Lumina AI, içeriklerin metinsel verilerini (film özetleri, türler, yazarlar) TF-IDF (Term Frequency-Inverse Document Frequency) yöntemiyle sayısal vektörlere dönüştürür. Ardından, bu vektörler arasındaki Cosine Similarity (Kosinüs Benzerliği) değerini hesaplayarak, seçtiğiniz içeriğe matematiksel olarak en yakın olanları önünüze getirir.
