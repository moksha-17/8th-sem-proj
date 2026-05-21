import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optimize Streamlit presentation view layout
st.set_page_config(page_title="Course Recommender Engine", layout="wide")
st.title("🎓 Machine Learning Course Recommender")
st.caption("High-Performance Single-Process In-Memory Computing Paradigm")

# Cache calculations to ensure sub-millisecond mathematical performance
@st.cache_resource
def initialize_and_compute_matrix():
    # Load dataset directly into operational RAM from the local asset tier
    try:
        catalog_df = pd.read_csv("courses.csv")
    except FileNotFoundError:
        # Fallback dictionary matrix if the local file isn't populated
        catalog_df = pd.DataFrame({
            'id': [201, 202, 203, 204, 205, 206],
            'title': ['Data Science Intro', 'Advanced ML', 'Software Eng', 'Cloud Architecture', 'Database Systems', 'Deep Neural Networks'],
            'meta': [
                'python data pandas stats machine learning foundational',
                'ml ai python clustering optimization neural networks advanced',
                'software engineering java architecture design patterns agile',
                'cloud computing aws architecture distributed systems devops',
                'database SQL indexing query optimization relational storage',
                'deep learning neural networks computer vision ai python'
            ]
        })
    
    # NLP Engine: Syntactic filtering & high-dimensional feature space mapping
    vec = TfidfVectorizer(stop_words='english')
    matrix = vec.fit_transform(catalog_df['meta'].fillna(''))
    
    # Generate the comprehensive non-Euclidean angular similarity grid
    sim_grid = cosine_similarity(matrix, matrix)
    return catalog_df, sim_grid

# Load data assets directly into persistent memory addresses
courses_df, proximity_matrix = initialize_and_compute_matrix()

# Phase 3: Presentation Layer Interactive Selection
st.subheader("Select an Anchor Module")
selected_title = st.selectbox("Choose a course from the platform catalog:", courses_df['title'].tolist())

if selected_title:
    # Capture event and extract the matching row index directly from RAM
    idx = courses_df[courses_df['title'] == selected_title].index[0]
    
    # Benchmark computational processing latency
    start_time = time.perf_counter()
    similarity_scores = list(enumerate(proximity_matrix[idx]))
    
    # Sort the indices based on alignment, stripping out the self-matched core item
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:4]
    execution_latency = (time.perf_counter() - start_time) * 1000 # Convert to ms

    st.markdown("---")
    st.subheader("Recommended Next-Sequence Educational Modules")
    
    # Render interactive layout columns reactively
    cols = st.columns(len(sorted_scores))
    for col, (recommend_idx, score) in zip(cols, sorted_scores):
        with col:
            st.info(f"**{courses_df.iloc[recommend_idx]['title']}**")
            st.metric(label="Thematic Alignment", value=f"{score*100:.1f}%")
            st.caption(f"Database Node ID: {courses_df.iloc[recommend_idx]['id']}")
            
    # System Status Performance Footer
    st.markdown("---")
    st.caption(f"⚡ In-Memory Vector Compute Latency: {execution_latency:.4f} ms | UI Status: Optimal")
