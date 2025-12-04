import re
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from src.hybrid_recommender import (
    load_data,
    build_popularity_model,
    build_content_model,
    build_item_similarity_matrix,
    hybrid_recommend,
)

# ---------------------------------------------------
# TMDB SETTINGS
# ---------------------------------------------------
TMDB_API_KEY = "5b4e3bd733d52d9f78f9c892f98bd3ad"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750.png?text=No+Image"


# ---------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Hybrid Movie Recommender ")
st.write("Fast, personalized movie recommendations powered by ML + TMDB.")


# ---------------------------------------------------
# GLOBAL CSS (for page width only)
# ---------------------------------------------------
st.markdown("""
<style>
.main .block-container { max-width: 1300px; padding: 2rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def clean_title(raw_title: str):
    """Remove (1999) etc."""
    return re.sub(r"\s*\(\d{4}\)$", "", raw_title).strip()


@st.cache_data(show_spinner=False)
def fetch_tmdb_data(movie_id: int, title: str):
    """FAST TMDB fetch with caching + timeout + fallback."""
    try:
        clean = clean_title(title)
        params = {"api_key": TMDB_API_KEY, "query": clean}
        r = requests.get(TMDB_SEARCH_URL, params=params, timeout=1)

        data = r.json()
        results = data.get("results", [])

        if not results:
            raise Exception("No results")

        best = results[0]
        poster_path = best.get("poster_path")
        poster = TMDB_POSTER_BASE + poster_path if poster_path else PLACEHOLDER_POSTER

        return {
            "poster_url": poster,
            "rating": best.get("vote_average", None),
            "overview": best.get("overview", "No description."),
            "year": (best.get("release_date", "")[:4] if best.get("release_date") else "—"),
        }

    except:
        return {
            "poster_url": PLACEHOLDER_POSTER,
            "rating": None,
            "overview": "No description.",
            "year": "—",
        }


# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_data(show_spinner=True)
def load_all_models():
    ratings, movies = load_data("data/ratings.csv", "data/movies.csv")
    popularity_df = build_popularity_model(ratings)
    tfidf_matrix, tfidf_ids, _ = build_content_model(movies)
    item_similarity, user_item_matrix = build_item_similarity_matrix(ratings)
    return ratings, movies, popularity_df, tfidf_matrix, tfidf_ids, item_similarity, user_item_matrix


with st.spinner("Loading models…"):
    ratings, movies, popularity_df, tfidf_matrix, tfidf_ids, item_similarity, user_item_matrix = load_all_models()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("🔧 Controls")
user_id = st.sidebar.selectbox("Select User ID", sorted(ratings["userId"].unique()))
top_n = st.sidebar.slider("Number of Movies", 5, 25, 10)

st.sidebar.subheader("Model Weights")
w_pop = st.sidebar.slider("Popularity", 0.0, 1.0, 0.2)
w_cb = st.sidebar.slider("Content-Based", 0.0, 1.0, 0.4)
w_cf = st.sidebar.slider("Collaborative", 0.0, 1.0, 0.4)


# ---------------------------------------------------
# RECOMMENDATION ENGINE (NO HTML ANYMORE)
# ---------------------------------------------------
if st.sidebar.button("🎯 Recommend Now"):

    # Generate recommendations
    with st.spinner("Calculating recommendations…"):
        recs = hybrid_recommend(
            user_id=user_id,
            ratings=ratings,
            movies=movies,
            popularity_df=popularity_df,
            tfidf_matrix=tfidf_matrix,
            tfidf_movie_ids=tfidf_ids,
            item_similarity=item_similarity,
            user_item_matrix=user_item_matrix,
            top_n=top_n,
            w_pop=w_pop,
            w_cb=w_cb,
            w_cf=w_cf,
        )

    st.subheader(f"🎥 Top {top_n} Movies for User {user_id}")

    # -------------------- PARALLEL TMDB FETCH --------------------
    def load_tmdb(movie_row):
        return fetch_tmdb_data(movie_row["movieId"], movie_row["title"])

    with ThreadPoolExecutor(max_workers=10) as executor:
        tmdb_results = list(executor.map(load_tmdb, [row for _, row in recs.iterrows()]))

    # -------------------- STREAMLIT GRID --------------------
    num_cols = 3
    cols = st.columns(num_cols)

    for idx, ((_, row), tmdb) in enumerate(zip(recs.iterrows(), tmdb_results)):
        col = cols[idx % num_cols]
        with col:
            poster = tmdb["poster_url"]
            rating = tmdb["rating"]
            overview = tmdb["overview"]
            year = tmdb["year"]
            rating_str = f"{rating:.1f}/10" if rating else "N/A"
            score = row["hybrid_score"]

            st.image(poster, use_container_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(f"📅 {year} • ⭐ {rating_str} • Hybrid Score: {score:.3f}")
            st.write(overview[:200] + ("..." if len(overview) > 200 else ""))


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.write("Built with ❤️ using Streamlit · TMDB API · Hybrid AI Model")
