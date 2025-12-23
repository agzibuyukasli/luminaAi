import ast
import json
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import requests
def init_fav_state():
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []
    if "fav_count" not in st.session_state:
        # İlk yüklemede mevcut favoriler varsa sayacı ona göre ayarla
        st.session_state["fav_count"] = len(st.session_state.get("favorites", []))
    if "page_mode" not in st.session_state:
        st.session_state["page_mode"] = "home"
    if "show_favs" not in st.session_state:
        st.session_state["show_favs"] = False
    if "show_favs_panel" not in st.session_state:
        st.session_state["show_favs_panel"] = False


def is_favorite(item_type: str, item_id) -> bool:
    return any((f.get("id") == item_id and f.get("type") == item_type) for f in st.session_state["favorites"])


def toggle_favorite(item_type: str, item_id, title: str, image: Optional[str] = None):
    if is_favorite(item_type, item_id):
        st.session_state["favorites"] = [
            f for f in st.session_state["favorites"] if not (f.get("id") == item_id and f.get("type") == item_type)
        ]
        st.session_state["fav_count"] = max(0, st.session_state["fav_count"] - 1)  # Sayaç 1 azal
        st.toast(f"'{title}' favorilerden çıkarıldı", icon="💔")
    else:
        st.session_state["favorites"].append({"type": item_type, "id": item_id, "title": title, "image": image})
        st.session_state["fav_count"] = st.session_state["fav_count"] + 1  # Sayaç 1 art
        st.toast(f"'{title}' favorilere eklendi", icon="❤️")
    st.rerun()  # Sayacı güncellemek için


def remove_favorite(item_type: str, item_id):
    st.session_state["favorites"] = [
        f for f in st.session_state["favorites"] if not (f.get("id") == item_id and f.get("type") == item_type)
    ]
    st.session_state["fav_count"] = max(0, st.session_state["fav_count"] - 1)  # Sayaç 1 azal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# Yardımcı fonksiyonlar
# -----------------------
def _safe_json_loads(cell: str) -> list:
    """
    JSON görünümlü listeleri güvenle parse eder.
    TMDB alanları bazen string olarak gelir; hata durumunda boş liste döner.
    """
    if pd.isna(cell) or cell == "":
        return []
    # Bazı satırlar tek tırnak içerebiliyor, ast.literal_eval daha toleranslı.
    try:
        return ast.literal_eval(cell)
    except Exception:
        try:
            return json.loads(cell)
        except Exception:
            return []


def _extract_names(records: list, key: str = "name", limit: Optional[int] = None) -> List[str]:
    items = [str(r.get(key, "")).strip() for r in records if isinstance(r, dict) and r.get(key)]
    if limit:
        items = items[:limit]
    return items


def _clean_text_list(items: List[str]) -> str:
    return " ".join(items).lower()


def load_lottieurl(url: str):
    """
    Lottie JSON'ı URL'den indirir; hata durumunda None döner.
    """
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def build_movie_features(
    movies_path: str, credits_path: str, nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    movies = pd.read_csv(movies_path, nrows=nrows)
    credits = pd.read_csv(credits_path, nrows=nrows)

    # credits'teki movie_id ile eşleştir.
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")

    # JSON alanlarını temizle.
    for col in ["genres", "keywords", "cast", "crew"]:
        df[col] = df[col].apply(_safe_json_loads)

    df["genres"] = df["genres"].apply(_extract_names)
    df["keywords"] = df["keywords"].apply(_extract_names)
    df["cast"] = df["cast"].apply(lambda x: _extract_names(x, limit=5))  # ilk 5 oyuncu yeterli.

    def _director(crew_list: list) -> List[str]:
        for member in crew_list:
            if isinstance(member, dict) and member.get("job") == "Director":
                return [member.get("name", "")]
        return []

    df["director"] = df["crew"].apply(_director)

    df["overview"] = df["overview"].fillna("").str.lower()
    df["tags"] = (
        df["overview"]
        + " "
        + df["genres"].apply(_clean_text_list)
        + " "
        + df["keywords"].apply(_clean_text_list)
        + " "
        + df["cast"].apply(_clean_text_list)
        + " "
        + df["director"].apply(_clean_text_list)
    )

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["tags"])
    similarity = cosine_similarity(matrix, matrix)

    result = df[["id", "title", "popularity", "vote_average"]].copy()
    result["tags"] = df["tags"]
    result["popularity"] = result["popularity"].fillna(0)
    result["vote_average"] = result["vote_average"].fillna(0)

    # Min-max normalizasyonu; sabit aralık için sıfır bölme korumalı.
    pop_min, pop_max = result["popularity"].min(), result["popularity"].max()
    vote_min, vote_max = result["vote_average"].min(), result["vote_average"].max()
    pop_denom = pop_max - pop_min if pop_max != pop_min else 1
    vote_denom = vote_max - vote_min if vote_max != vote_min else 1
    result["popularity_norm"] = (result["popularity"] - pop_min) / pop_denom
    result["vote_norm"] = (result["vote_average"] - vote_min) / vote_denom

    return result, similarity


def build_book_features(books_path: str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    books = pd.read_csv(
        books_path,
        sep=";",
        quotechar='"',
        encoding="latin-1",
        nrows=nrows,
        on_bad_lines="skip",
    )
    books = books.rename(
        columns={
            "Book-Title": "title",
            "Book-Author": "author",
            "Publisher": "publisher",
            "Image-URL-L": "image_l",
            "Image-URL-M": "image_m",
        }
    )
    books = books[["title", "author", "publisher", "image_l", "image_m"]].dropna(subset=["title", "author", "publisher"])
    books["title"] = books["title"].astype(str)
    books["author"] = books["author"].astype(str)
    books["publisher"] = books["publisher"].astype(str)
    books["image_l"] = books["image_l"].fillna("")
    books["image_m"] = books["image_m"].fillna("")

    books["tags"] = (
        books["title"].str.lower()
        + " "
        + books["author"].str.lower()
        + " "
        + books["publisher"].str.lower()
    )

    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(books["tags"])
    similarity = cosine_similarity(matrix, matrix)

    return books.reset_index(drop=True), similarity


def top_similar_movies(name: str, data: pd.DataFrame, similarity: np.ndarray, name_col: str = "title", top_k: int = 20) -> pd.DataFrame:
    if name not in data[name_col].values:
        return pd.DataFrame(columns=data.columns)
    idx = data.index[data[name_col] == name][0]
    raw_scores = list(enumerate(similarity[idx]))

    # Ağırlıklı skor: içerik benzerliği + normalize popülerlik + normalize puan.
    WEIGHT_SIM = 0.6
    WEIGHT_VOTE = 0.25
    WEIGHT_POP = 0.15

    weighted = []
    for i, sim_score in raw_scores:
        if i == idx:
            continue  # kendisi hariç
        pop_score = data.iloc[i].get("popularity_norm", 0)
        vote_score = data.iloc[i].get("vote_norm", 0)
        combined = WEIGHT_SIM * sim_score + WEIGHT_VOTE * vote_score + WEIGHT_POP * pop_score
        weighted.append((i, combined))

    if not weighted:
        return pd.DataFrame(columns=data.columns)

    scores = sorted(weighted, key=lambda x: x[1], reverse=True)[:top_k]
    max_score = scores[0][1] if scores else 1
    indices = [i for i, _ in scores]
    result = data.iloc[indices].copy()
    result["score"] = [s for _, s in scores]
    result["match_rate"] = result["score"] / max_score if max_score else 0
    return result


def top_similar_books(name: str, data: pd.DataFrame, similarity: np.ndarray, name_col: str = "title", top_k: int = 20) -> pd.DataFrame:
    if name not in data[name_col].values:
        return pd.DataFrame(columns=data.columns)
    idx = data.index[data[name_col] == name][0]
    raw_scores = list(enumerate(similarity[idx]))

    scored = []
    for i, sim_score in raw_scores:
        if i == idx:
            continue
        scored.append((i, sim_score))
    if not scored:
        return pd.DataFrame(columns=data.columns)

    scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    max_score = scored[0][1] if scored else 1
    indices = [i for i, _ in scored]
    result = data.iloc[indices].copy()
    result["score"] = [s for _, s in scored]
    result["match_rate"] = result["score"] / max_score if max_score else 0
    return result


# -----------------------
# Cache katmanı
# -----------------------
@st.cache_data(show_spinner=False)
def load_movies_cached(sample_size: Optional[int]):
    return build_movie_features(
        movies_path="data/tmdb_5000_movies.csv",
        credits_path="data/tmdb_5000_credits.csv",
        nrows=sample_size,
    )


@st.cache_data(show_spinner=False)
def load_books_cached(sample_size: Optional[int]):
    return build_book_features(books_path="data/books.csv", nrows=sample_size)


# -----------------------
# Streamlit Arayüzü
# -----------------------
def render_movie_tab():
    st.subheader("Film Önerici")
    movies, movie_sim = load_movies_cached(5000)
    selected = st.selectbox("Bir film seçin", movies["title"].values)
    if selected:
        with st.spinner("Processing..."):
            recs = top_similar_movies(selected, movies, movie_sim, name_col="title", top_k=20)
        st.markdown("Benzer Filmler:")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(recs.iterrows()):
            col = cols[idx % 3]
            with col:
                movie_placeholder = "https://images.unsplash.com/photo-1524985069026-dd778a71c7b4?auto=format&fit=crop&w=900&q=80"
                img_tag = (
                    f"<img src='{movie_placeholder}' class='card-cover' "
                    "onerror=\"this.onerror=null;this.style.display='none';this.insertAdjacentHTML('afterend','<div class=\\'card-img placeholder\\'>Görsel Mevcut Değil</div>');\"/>"
                )
                card_html = f"""
                <div class="rec-card movie-card">
                    {img_tag}
                    <div class="card-body">
                        <div class="card-title">{row['title']}</div>
                        <div class="badge">Eşleşme Oranı: {row['match_rate']*100:.0f}%</div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                fav_key = f"fav-movie-{row['id']}-{idx}"
                btn_label = "💔 Favoriden çıkar" if is_favorite("movie", row["id"]) else "❤️ Favorilere ekle"
                if st.button(btn_label, key=fav_key):
                    toggle_favorite("movie", row["id"], row["title"], image=movie_placeholder)


def render_book_tab():
    st.subheader("Kitap Önerici")
    col_anim, col_ui = st.columns([1, 3])
    with col_anim:
        book_lottie = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ai9m80ad.json")
        if book_lottie:
            st_lottie(book_lottie, height=110, key="book-lottie")
    books, book_sim = load_books_cached(8000)
    selected = st.selectbox("Bir kitap seçin", books["title"].values)
    if selected:
        with st.spinner("Processing..."):
            recs = top_similar_books(selected, books, book_sim, name_col="title", top_k=20)
        st.markdown("Benzer Kitaplar:")
        for start in range(0, len(recs), 3):
            cols = st.columns(3)
            for offset, col in enumerate(cols):
                idx = start + offset
                if idx >= len(recs):
                    continue
                row = recs.iloc[idx]
                with col:
                    img_url = row.get("image_l") if isinstance(row.get("image_l"), str) and row.get("image_l") else row.get("image_m")
                    if isinstance(img_url, str) and img_url.strip():
                        img_tag = (
                            f"<img src='{img_url}' class='card-cover' "
                            "onerror=\"this.onerror=null;this.style.display='none';this.insertAdjacentHTML('afterend','<div class=\\'card-img placeholder\\'>Görsel Mevcut Değil</div>');\"/>"
                        )
                    else:
                        img_tag = "<div class='card-img placeholder'>Görsel Mevcut Değil</div>"
                    card_html = f"""
                    <div class="rec-card book-card">
                        {img_tag}
                        <div class="card-body">
                            <div class="card-title">{row['title']}</div>
                            <div class="card-meta">Yazar: {row['author']} | Yayınevi: {row['publisher']}</div>
                            <div class="badge">Eşleşme Oranı: {row.get('match_rate', 0)*100:.0f}%</div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    fav_key = f"fav-book-{idx}-{row.get('title','')}"
                    btn_label = "💔 Favoriden çıkar" if is_favorite("book", row.get("title")) else "❤️ Favorilere ekle"
                    if st.button(btn_label, key=fav_key):
                        toggle_favorite("book", row.get("title"), row.get("title"), image=img_url if isinstance(img_url, str) else None)


def main():
    st.set_page_config(page_title="Yapay Zeka Destekli Öneri Sistemi", layout="wide")
    init_fav_state()
    # Özel tema CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Playfair+Display:wght@700&display=swap');
        /* Global Soft Pink-Navy Theme */
        html, body, [class*="block-container"] {
            background: #FBEFF4 !important;
            color: #1B264F !important;
            font-family: 'Montserrat', sans-serif !important;
        }
        h1, h2, h3, p, div, span {
            font-family: 'Montserrat', sans-serif !important;
        }
        h3 { font-size: 26px; font-weight: 800; }
        /* Sticky Navbar */
        .top-nav-left {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 4px;
            padding: 12px 0;
        }
        .nav-left { display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }
        .logo-row { display: flex; align-items: center; gap: 8px; }
        .logo-text {
            font-family: 'Playfair Display', serif !important;
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -1px;
            color: #1B264F;
            background: #ffffff;
            padding: 6px 12px;
            border-radius: 10px;
        }
        .logo-ai {
            background: #F3C1C6;
            color: #ffffff;
            padding: 6px 10px;
            border-radius: 8px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }
        .nav-slogan {
            font-size: 12px;
            color: #F3C1C6;
            font-style: italic;
            margin-left: 4px;
        }
        .nav-right { display: flex; align-items: center; gap: 10px; }
        .fav-btn {
            border: 1px solid rgba(27,38,79,0.15);
            background: #F3C1C6;
            color: #1B264F;
            padding: 10px 14px;
            border-radius: 20px;
            font-weight: 800;
            box-shadow: 0 8px 20px rgba(243,193,198,0.35);
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .fav-btn:hover { color: #1B264F; filter: brightness(0.95); box-shadow: 0 10px 24px rgba(27,38,79,0.2); }
        .fav-heart { font-size: 14px; }
        .fav-count {
            background: #1B264F;
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
        }
        /* Cards */
        .rec-card {
            background: #ffffff;
            border-radius: 15px;
            padding: 12px;
            margin-bottom: 14px;
            box-shadow: 0 10px 26px rgba(243,193,198,0.25);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            border: 1px solid rgba(243,193,198,0.5);
        }
        .rec-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 40px rgba(242,138,178,0.35);
        }
        .card-img, .card-cover {
            width: 100%;
            border-radius: 12px;
            height: 320px;
            object-fit: cover;
            background: #f7e7ec;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 42px;
            color: #1B264F;
        }
        .movie-gradient { background: linear-gradient(135deg, #f6c7d8 0%, #f2cbe0 100%); }
        .book-gradient { background: linear-gradient(135deg, #f2cbe0 0%, #f6c7d8 100%); }
        .card-body { margin-top: 10px; }
        .card-title {
            font-weight: 800;
            font-size: 16px;
            color: #1B264F;
        }
        .card-meta {
            font-size: 12px;
            color: #4b5563;
            margin-top: 4px;
        }
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #1B264F 0%, #F3C1C6 100%);
            color: #ffffff;
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            margin-top: 10px;
            letter-spacing: 0.2px;
        }
        .placeholder {
            color: #6b7280;
            font-weight: 600;
            font-size: 13px;
            background: #f7e7ec;
        }
        .fav-card-btn {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(27,38,79,0.2);
            background: #1B264F;
            color: #F3C1C6;
            font-weight: 700;
            padding: 8px 10px;
            box-shadow: 0 8px 20px rgba(27,38,79,0.14);
        }
        /* Tabs */
        .stTabs [role="tablist"] {
            gap: 8px;
            border-bottom: 1px solid rgba(0,0,0,0.04);
        }
        .stTabs [role="tab"] {
            border-radius: 12px 12px 0 0;
            padding: 10px 16px;
            border: 1px solid transparent;
            color: #1B264F;
            font-weight: 800;
            font-size: 20px;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            border: 1px solid rgba(243,193,198,0.45);
            border-bottom: 2px solid #F3C1C6;
            background: #fff5f8;
            color: #1B264F;
            box-shadow: 0 8px 24px rgba(243,193,198,0.18);
        }
        /* Force buttons and headings */
        .stButton>button { 
            background-color: #F3C1C6 !important; 
            color: #1B264F !important; 
            border-radius: 20px !important; 
            font-weight: bold !important; 
            border: none !important;
            padding: 10px 16px !important;
            box-shadow: 0 8px 20px rgba(243,193,198,0.35) !important;
        }
        .stButton>button:hover { 
            filter: brightness(0.95) !important; 
            box-shadow: 0 10px 24px rgba(27,38,79,0.2) !important; 
        }
        .stMarkdown h1 { color: #1B264F !important; font-size: 45px !important; font-family: 'Montserrat', sans-serif !important; }
        /* Header wrapper - Full width lacivert header */
        .header-wrapper {
            background: #1B264F !important;
            border-radius: 0 !important;
            padding: 16px 3.5rem !important;
            box-shadow: 0 4px 12px rgba(27,38,79,0.3) !important;
            margin-left: calc(-50vw + 50%) !important;
            margin-right: calc(-50vw + 50%) !important;
            margin-top: -1rem !important;
            margin-bottom: 24px !important;
            width: 100vw !important;
            max-width: 100vw !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            position: relative !important;
            z-index: 100 !important;
            min-height: 70px !important;
        }
        /* Override Streamlit container to allow full-width header but keep padding */
        .main .block-container {
            padding-left: 3.5rem !important;
            padding-right: 3.5rem !important;
            max-width: 100% !important;
        }
        /* Ensure header content is properly aligned */
        .header-wrapper .stColumn {
            padding: 0 !important;
        }
        .header-left {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 4px;
        }
        .header-right {
            display: flex;
            align-items: center;
            justify-content: flex-end;
        }
        /* Streamlit button inside header - aynı görünüm */
        .header-wrapper .stButton {
            margin: 0 !important;
            padding: 0 !important;
        }
        .header-wrapper .stButton > button {
            background: #F3C1C6 !important;
            color: #1B264F !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 16px !important;
            font-weight: bold !important;
            font-size: 14px !important;
            cursor: pointer !important;
            box-shadow: 0 8px 20px rgba(243,193,198,0.35) !important;
            font-family: 'Montserrat', sans-serif !important;
        }
        .header-wrapper .stButton > button:hover {
            filter: brightness(0.95) !important;
            box-shadow: 0 10px 24px rgba(27,38,79,0.2) !important;
        }
        .fav-btn-in-header {
            background: #F3C1C6 !important;
            color: #1B264F !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 16px !important;
            font-weight: bold !important;
            font-size: 14px !important;
            cursor: pointer !important;
            box-shadow: 0 8px 20px rgba(243,193,198,0.35) !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 8px !important;
            font-family: 'Montserrat', sans-serif !important;
        }
        .fav-btn-in-header:hover {
            filter: brightness(0.95) !important;
            box-shadow: 0 10px 24px rgba(27,38,79,0.2) !important;
        }
        /* Streamlit button için header favorilerim butonu stili */
        button[data-testid="baseButton-header_fav_button"] {
            background: #F3C1C6 !important;
            color: #1B264F !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 16px !important;
            font-weight: bold !important;
            font-size: 14px !important;
            cursor: pointer !important;
            box-shadow: 0 8px 20px rgba(243,193,198,0.35) !important;
            font-family: 'Montserrat', sans-serif !important;
            width: auto !important;
        }
        button[data-testid="baseButton-header_fav_button"]:hover {
            filter: brightness(0.95) !important;
            box-shadow: 0 10px 24px rgba(27,38,79,0.2) !important;
        }
        .fav-count-badge {
            background: #1B264F !important;
            color: #ffffff !important;
            padding: 2px 8px !important;
            border-radius: 999px !important;
            font-size: 11px !important;
            font-weight: 800 !important;
        }
        /* Selectbox labels - lacivert yap */
        .stSelectbox label, .stSelectbox > label {
            color: #1B264F !important;
            font-weight: 600 !important;
        }
        /* Selectbox input - header rengiyle aynı lacivert (#1B264F) */
        .stSelectbox select, 
        .stSelectbox > div > div > select,
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] > div {
            background-color: #1B264F !important;
            background: #1B264F !important;
            color: #ffffff !important;
            border: 1px solid #1B264F !important;
            border-radius: 12px !important;
        }
        .stSelectbox select option,
        .stSelectbox [data-baseweb="select"] option {
            background-color: #1B264F !important;
            color: #ffffff !important;
        }
        /* Streamlit'in selectbox wrapper'ı */
        .stSelectbox [data-baseweb="select"] {
            background-color: #1B264F !important;
        }
        /* Favoriler Paneli Stili */
        .fav-panel-container {
            background: #FDFCFD !important;
            border: 2px solid #F3C1C6 !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 20px 0 !important;
            box-shadow: 0 8px 24px rgba(243,193,198,0.3) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header & nav - TÜM İÇERİK LACİVERT HEADER İÇİNDE
    # Sayaç direkt session_state'den alınıyor
    fav_count = st.session_state.get("fav_count", 0)
    
    # TÜM HEADER İÇERİĞİ TEK BİR HTML DIV İÇİNDE - YERLER AYNI KALDI
    st.markdown(
        f"""
        <div class="header-wrapper">
            <div class="header-left">
                <div class="logo-row">
                    <div class="logo-text">Lumina</div>
                    <div class="logo-ai">AI</div>
                </div>
                <div class="nav-slogan">Zekanın ışığında yeni keşifler.</div>
            </div>
            <div class="header-right">
        """,
        unsafe_allow_html=True,
    )
    
    # Streamlit button - panel açmak için
    if st.button(f"❤️ Lumina Favorilerim ({fav_count})", key="header_fav_button"):
        st.session_state["show_favs_panel"] = not st.session_state.get("show_favs_panel", False)
        st.rerun()
    
    st.markdown(
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-separator"></div>', unsafe_allow_html=True)
    
    # Favoriler Paneli - Header'ın altında
    if st.session_state.get("show_favs_panel", False):
        st.markdown('<div class="fav-panel-container">', unsafe_allow_html=True)
        col_title, col_close = st.columns([5, 1])
        with col_title:
            st.markdown('<h2 style="color: #1B264F; font-family: Montserrat, sans-serif; margin-bottom: 20px;">❤️ Favorilerim</h2>', unsafe_allow_html=True)
        with col_close:
            if st.button("✕ Kapat", key="close_favs_panel"):
                st.session_state["show_favs_panel"] = False
                st.rerun()
        
        favs = st.session_state.get("favorites", [])
        if not favs:
            st.info("Henüz favori eklemediniz.")
        else:
                # Verileri yükle
                movies, movie_sim = load_movies_cached(5000)
                books, book_sim = load_books_cached(8000)
                
                def find_movie(mid):
                    m = movies[movies["id"] == mid]
                    return m.iloc[0] if not m.empty else None
                def find_book(title):
                    b = books[books["title"] == title]
                    return b.iloc[0] if not b.empty else None
                
                # Favorileri göster
                for start in range(0, len(favs), 3):
                    cols = st.columns(3)
                    for offset, col in enumerate(cols):
                        idx = start + offset
                        if idx >= len(favs):
                            continue
                        fav = favs[idx]
                        with col:
                            if fav.get("type") == "movie":
                                row = find_movie(fav.get("id"))
                                if row is not None:
                                    img_url = fav.get("image") or "https://via.placeholder.com/300x450?text=Film"
                                    if isinstance(img_url, str) and img_url:
                                        img_tag = (
                                            f"<img src='{img_url}' class='card-cover' "
                                            "onerror=\"this.onerror=null;this.style.display='none';this.insertAdjacentHTML('afterend','<div class=\\'card-img placeholder\\'>Görsel Mevcut Değil</div>');\"/>"
                                        )
                                    else:
                                        img_tag = "<div class='card-img placeholder'>Görsel Mevcut Değil</div>"
                                    card_html = f"""
                                    <div class="rec-card">
                                        {img_tag}
                                        <div class="card-body">
                                            <div class="card-title">{row['title']}</div>
                                            <div class="badge">Film</div>
                                        </div>
                                    </div>
                                    """
                                    st.markdown(card_html, unsafe_allow_html=True)
                                    if st.button("💔 Favoriden çıkar", key=f"rm-fav-panel-movie-{fav.get('id')}"):
                                        remove_favorite("movie", fav.get("id"))
                                        st.rerun()
                            elif fav.get("type") == "book":
                                row = find_book(fav.get("id"))
                                if row is not None:
                                    img_url = row.get("image_l") if isinstance(row.get("image_l"), str) and row.get("image_l") else row.get("image_m")
                                    if isinstance(img_url, str) and img_url:
                                        img_tag = (
                                            f"<img src='{img_url}' class='card-cover' "
                                            "onerror=\"this.onerror=null;this.style.display='none';this.insertAdjacentHTML('afterend','<div class=\\'card-img placeholder\\'>Görsel Mevcut Değil</div>');\"/>"
                                        )
                                    else:
                                        img_tag = "<div class='card-img placeholder'>Görsel Mevcut Değil</div>"
                                    card_html = f"""
                                    <div class="rec-card">
                                        {img_tag}
                                        <div class="card-body">
                                            <div class="card-title">{row['title']}</div>
                                            <div class="badge">Kitap</div>
                                        </div>
                                    </div>
                                    """
                                    st.markdown(card_html, unsafe_allow_html=True)
                                    if st.button("💔 Favoriden çıkar", key=f"rm-fav-panel-book-{fav.get('id')}"):
                                        remove_favorite("book", fav.get("id"))
                                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Normal view - Film/Kitap sekmeleri (Panel açık olsa bile göster)
    tab_film, tab_book = st.tabs(["Film Öner", "Kitap Öner"])
    with tab_film:
        render_movie_tab()
    with tab_book:
        render_book_tab()

    # About / İsim Hikayesi
    st.info(
        'Lumina, adını Latince ışık anlamına gelen "Lumen" kelimesinden alır. '
        'Binlerce film ve kitap arasında kaybolduğunuzda, yapay zeka algoritmalarımız size en uygun yolu aydınlatmak için tasarlandı.'
    )


if __name__ == "__main__":
    main()
