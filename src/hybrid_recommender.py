# hybrid_recommender.py

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(
    ratings_path: str = "ratings.csv",
    movies_path: str = "movies.csv"
):
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Basic sanity checks
    required_r_cols = {"userId", "movieId", "rating"}
    required_m_cols = {"movieId", "title", "genres"}

    if not required_r_cols.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain columns: {required_r_cols}")
    if not required_m_cols.issubset(movies.columns):
        raise ValueError(f"movies.csv must contain columns: {required_m_cols}")

    return ratings, movies


def build_popularity_model(ratings: pd.DataFrame) -> pd.DataFrame:
   
    pop = (
        ratings
        .groupby("movieId")["rating"]
        .agg(["count", "mean"])
        .rename(columns={"count": "rating_count", "mean": "rating_mean"})
        .reset_index()
    )

    pop["popularity_score"] = pop["rating_count"] * pop["rating_mean"]

    max_score = pop["popularity_score"].max()
    if max_score > 0:
        pop["popularity_score_norm"] = pop["popularity_score"] / max_score
    else:
        pop["popularity_score_norm"] = 0.0

    return pop[["movieId", "popularity_score_norm"]]


def build_content_model(movies: pd.DataFrame):
    
    movies["genres"] = movies["genres"].fillna("")

    corpus = movies["genres"]

    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf_matrix = tfidf.fit_transform(corpus)

    movie_ids = movies["movieId"].values

    return tfidf_matrix, movie_ids, tfidf


def get_content_scores_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    movie_ids: np.ndarray
) -> pd.Series:
    
    user_ratings = ratings[ratings["userId"] == user_id]
    
    liked = user_ratings[user_ratings["rating"] >= 4.0]

    if liked.empty:
        return pd.Series(0.0, index=movie_ids)

    liked_movie_ids = liked["movieId"].values
    movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

    liked_indices = [movie_id_to_index[mid] for mid in liked_movie_ids if mid in movie_id_to_index]

    if not liked_indices:
        return pd.Series(0.0, index=movie_ids)

    liked_tfidf = tfidf_matrix[liked_indices]
    sim_matrix = cosine_similarity(liked_tfidf, tfidf_matrix)

    content_scores = sim_matrix.mean(axis=0)

    max_val = content_scores.max()
    if max_val > 0:
        content_scores = content_scores / max_val

    return pd.Series(content_scores, index=movie_ids)


def build_item_similarity_matrix(ratings: pd.DataFrame, min_ratings_per_movie: int = 10):
    
    movie_counts = ratings["movieId"].value_counts()
    valid_movies = movie_counts[movie_counts >= min_ratings_per_movie].index

    filtered = ratings[ratings["movieId"].isin(valid_movies)]

    user_item = filtered.pivot_table(
        index="userId", columns="movieId", values="rating"
    )

    user_item_filled = user_item.fillna(0)

    item_sim = cosine_similarity(user_item_filled.T)
    item_sim_df = pd.DataFrame(
        item_sim,
        index=user_item_filled.columns,
        columns=user_item_filled.columns
    )

    return item_sim_df, user_item


def get_cf_scores_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    item_similarity: pd.DataFrame,
    user_item_matrix: pd.DataFrame
) -> pd.Series:
   
    if user_id not in user_item_matrix.index:
        return pd.Series(0.0, index=item_similarity.index)

    user_ratings = user_item_matrix.loc[user_id]
    user_ratings = user_ratings.dropna()

    if user_ratings.empty:
        return pd.Series(0.0, index=item_similarity.index)

    scores = {}
    for target_movie in item_similarity.index:
        if target_movie in user_ratings.index:
            continue

        sim_vector = item_similarity.loc[target_movie, user_ratings.index]
        rating_vector = user_ratings.values

        numerator = np.dot(sim_vector.values, rating_vector)
        denominator = sim_vector.values.sum()

        if denominator > 0:
            pred = numerator / denominator
            scores[target_movie] = pred

    if not scores:
        return pd.Series(0.0, index=item_similarity.index)

    cf_scores = pd.Series(scores)

    max_val = cf_scores.max()
    if max_val > 0:
        cf_scores = cf_scores / max_val

    cf_scores = cf_scores.reindex(item_similarity.index, fill_value=0.0)

    return cf_scores


def hybrid_recommend(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    popularity_df: pd.DataFrame,
    tfidf_matrix,
    tfidf_movie_ids: np.ndarray,
    item_similarity: pd.DataFrame,
    user_item_matrix: pd.DataFrame,
    top_n: int = 10,
    w_pop: float = 0.2,
    w_cb: float = 0.4,
    w_cf: float = 0.4
) -> pd.DataFrame:
    

    pop_series = popularity_df.set_index("movieId")["popularity_score_norm"]

    cb_scores = get_content_scores_for_user(
        user_id, ratings, movies, tfidf_matrix, tfidf_movie_ids
    )

    cf_scores = get_cf_scores_for_user(
        user_id, ratings, item_similarity, user_item_matrix
    )


    # -------------------------------
    # NEW LOGIC → Popularity-only gives SAME output for everyone
    # -------------------------------
    if w_pop == 1 and w_cb == 0 and w_cf == 0:
        rated_movies = set()  # Do NOT exclude any movies
    else:
        rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"].unique())
    # -------------------------------


    all_movie_ids = set(pop_series.index) | set(cb_scores.index) | set(cf_scores.index)
    all_movie_ids = sorted(all_movie_ids)

    pop_aligned = pop_series.reindex(all_movie_ids, fill_value=0.0)
    cb_aligned = cb_scores.reindex(all_movie_ids, fill_value=0.0)
    cf_aligned = cf_scores.reindex(all_movie_ids, fill_value=0.0)

    final_scores = []
    for mid in all_movie_ids:
        if mid in rated_movies:
            continue

        s_pop = pop_aligned.loc[mid]
        s_cb = cb_aligned.loc[mid]
        s_cf = cf_aligned.loc[mid]
        score = w_pop * s_pop + w_cb * s_cb + w_cf * s_cf

        final_scores.append((mid, s_pop, s_cb, s_cf, score))

    if not final_scores:
        return pd.DataFrame(columns=["movieId", "title", "pop", "cb", "cf", "hybrid_score"])

    rec_df = pd.DataFrame(
        final_scores,
        columns=["movieId", "pop", "cb", "cf", "hybrid_score"]
    )

    rec_df = rec_df.merge(movies[["movieId", "title"]], on="movieId", how="left")

    rec_df = rec_df.sort_values("hybrid_score", ascending=False).head(top_n)

    rec_df = rec_df[["movieId", "title", "pop", "cb", "cf", "hybrid_score"]]

    return rec_df.reset_index(drop=True)



if __name__ == "__main__":

    ratings, movies = load_data(
        ratings_path=r"data/ratings.csv",
        movies_path=r"data/movies.csv"
    )

    popularity_df = build_popularity_model(ratings)

    tfidf_matrix, tfidf_movie_ids, tfidf_vectorizer = build_content_model(movies)

    item_similarity, user_item_matrix = build_item_similarity_matrix(ratings)

    sample_user_id = int(ratings["userId"].sample(1, random_state=42).values[0])
    print(f"Generating recommendations for user {sample_user_id}...\n")

    recs = hybrid_recommend(
        user_id=sample_user_id,
        ratings=ratings,
        movies=movies,
        popularity_df=popularity_df,
        tfidf_matrix=tfidf_matrix,
        tfidf_movie_ids=tfidf_movie_ids,
        item_similarity=item_similarity,
        user_item_matrix=user_item_matrix,
        top_n=10,
        w_pop=1.0,   # TESTING POPULARITY ONLY
        w_cb=0.0,
        w_cf=0.0
    )

    print(recs)
