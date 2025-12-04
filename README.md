🎬 Hybrid Movie Recommendation System

This project is a mini recommendation system that suggests movies using a hybrid approach. It combines:

Popularity-Based Filtering

Content-Based Filtering (TF-IDF + Cosine Similarity)

Item-Based Collaborative Filtering

The system is built using the MovieLens dataset and uses the TMDB API to display posters and movie details.
A Streamlit interface is included for easy interaction.

🔧 Features

Hybrid scoring using 3 models

Adjustable model weights

Personalized user profiles

TMDB posters, ratings, and overviews

Clean, responsive Streamlit UI

🧠 How It Works

Popularity score = rating count × rating mean

Content-based score from TF-IDF genre vectors

Collaborative filtering using item-item similarity

Hybrid Score = weighted combination of all three

📁 Project Structure
app.py               # Streamlit UI
src/hybrid_recommender.py  # ML backend
data/ratings.csv
data/movies.csv

▶️ Run the App
pip install -r requirements.txt
streamlit run app.py

📚 Dataset

MovieLens dataset containing movie titles, genres, and user ratings.

