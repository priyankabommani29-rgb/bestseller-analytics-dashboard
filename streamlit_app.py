# streamlit_app.py

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
import requests

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="📚 Bestseller Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Custom Header
# ---------------------------
st.markdown("""
<style>
.main-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #262730;
    padding: 15px 30px;
    border-bottom: 2px solid #4CAF50;
}
.main-header .logo { font-size: 22px; font-weight: bold; color: #4CAF50; }
.main-header .nav { font-size: 16px; }
.main-header .nav a { margin-left: 20px; text-decoration: none; color: #FAFAFA; font-weight: 500; }
.main-header .nav a:hover { color: #4CAF50; }
</style>
<div class="main-header">
    <div class="logo">📚 Bestseller Analytics</div>
    <div class="nav">
        <a href="#overview">Overview</a>
        <a href="#analysis">Analysis</a>
        <a href="#trends">Trends</a>
        <a href="#predictor">Predictor</a>
        <a href="#dataset">Dataset</a>
        <a href="https://github.com/yourusername" target="_blank">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSTa4mpDseg3bYEZ2jETc6onjpeARl-Hlt4Bsba8i3R8W4TvqmTp8jxfH3cnWLL-qOne4eFh0KRqnBM/pub?output=csv"
    df = pd.read_csv(sheet_url)
    df = df.drop_duplicates()
    
    # Rename columns if they exist
    rename_map = {}
    if "Name" in df.columns: rename_map["Name"] = "Title"
    if "Year" in df.columns: rename_map["Year"] = "Publication Year"
    if "User Rating" in df.columns: rename_map["User Rating"] = "Rating"
    df = df.rename(columns=rename_map)
    
    # Create default columns if missing
    if "Genre" not in df.columns: df["Genre"] = "Unknown"
    if "Publication Year" not in df.columns: df["Publication Year"] = 2000
    if "Price" not in df.columns: df["Price"] = 0.0
    if "Rating" not in df.columns: df["Rating"] = 0.0
    if "Author" not in df.columns: df["Author"] = "Unknown"
    
    # Ensure correct types
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    df["Publication Year"] = pd.to_numeric(df["Publication Year"], errors="coerce").fillna(2000).astype(int)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0.0)
    
    return df

df = load_data()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.header("🔍 Filters")

selected_genre = st.sidebar.multiselect(
    "Select Genre(s)",
    options=df["Genre"].unique(),
    default=df["Genre"].unique()
)

selected_year = st.sidebar.slider(
    "Publication Year Range",
    int(df["Publication Year"].min()),
    int(df["Publication Year"].max()),
    (int(df["Publication Year"].min()), int(df["Publication Year"].max()))
)

filtered_df = df[
    (df["Genre"].isin(selected_genre)) &
    (df["Publication Year"].between(selected_year[0], selected_year[1]))
]

# ---------------------------
# Overview Section
# ---------------------------
st.markdown('<a name="overview"></a>', unsafe_allow_html=True)
st.title("📊 Bestseller Analytics Dashboard")
st.markdown("Explore trends in bestselling books: **authors, ratings, prices, genres**, and enriched info!")

col1, col2, col3 = st.columns(3)
col1.metric("📘 Total Books", filtered_df.shape[0])
col2.metric("⭐ Avg. Rating", round(filtered_df["Rating"].mean(), 2) if not filtered_df.empty else 0)
col3.metric("💰 Avg. Price ($)", round(filtered_df["Price"].mean(), 2) if not filtered_df.empty else 0)

# ---------------------------
# Analysis Section
# ---------------------------
st.markdown('<a name="analysis"></a>', unsafe_allow_html=True)
st.header("📈 Analysis")

if not filtered_df.empty:
    st.subheader("🏆 Top 10 Authors")
    author_counts = filtered_df["Author"].value_counts().head(10)
    fig_authors = px.bar(
        x=author_counts.values, y=author_counts.index, orientation="h",
        title="Top 10 Authors", labels={"x":"Number of Books","y":"Author"},
        color=author_counts.values, color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_authors, use_container_width=True)
    
    st.subheader("⭐ Average Rating by Genre")
    avg_rating_by_genre = filtered_df.groupby("Genre")["Rating"].mean().reset_index()
    fig_genre = px.bar(avg_rating_by_genre, x="Genre", y="Rating",
                       color="Rating", color_continuous_scale="viridis",
                       title="Average Rating by Genre")
    st.plotly_chart(fig_genre, use_container_width=True)
else:
    st.info("No data available for selected filters.")

# ---------------------------
# Trends Section
# ---------------------------
st.markdown('<a name="trends"></a>', unsafe_allow_html=True)
st.header("📅 Trends Over Time")

if not filtered_df.empty:
    rating_trend = filtered_df.groupby("Publication Year")["Rating"].mean().reset_index().sort_values("Publication Year")
    fig_rating_trend = px.line(rating_trend, x="Publication Year", y="Rating", markers=True,
                               title="Average Rating Over Years", line_shape="spline",
                               color_discrete_sequence=["#00CC96"])
    st.plotly_chart(fig_rating_trend, use_container_width=True)

    price_trend = filtered_df.groupby("Publication Year")["Price"].mean().reset_index().sort_values("Publication Year")
    fig_price_trend = px.line(price_trend, x="Publication Year", y="Price", markers=True,
                              title="Average Price Over Years", line_shape="spline",
                              color_discrete_sequence=["#AB63FA"])
    st.plotly_chart(fig_price_trend, use_container_width=True)

# ---------------------------
# ML Predictor Section
# ---------------------------
st.markdown('<a name="predictor"></a>', unsafe_allow_html=True)
st.header("🤖 AI-Powered Rating Predictor")

if not df.empty:
    X = df[["Price", "Publication Year"]]
    y = df["Rating"]
    model = LinearRegression().fit(X, y)

    col1, col2 = st.columns(2)
    with col1: price_input = st.slider("Select Price ($)", int(df["Price"].min()), int(df["Price"].max()), 20)
    with col2: year_input = st.slider("Select Year", int(df["Publication Year"].min()), int(df["Publication Year"].max()), 2015)

    predicted_rating = model.predict([[price_input, year_input]])[0]
    st.metric("Predicted Rating", f"{predicted_rating:.2f} ⭐")
    st.caption("Based on Linear Regression model trained on bestseller dataset.")
else:
    st.info("Dataset is empty, cannot train predictor.")

# ---------------------------
# Dataset Section
# ---------------------------
st.markdown('<a name="dataset"></a>', unsafe_allow_html=True)
st.header("📄 Dataset Preview")
with st.expander("🔎 Show first 20 rows of filtered dataset"):
    st.dataframe(filtered_df.head(20))

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: #FAFAFA;
    text-align: center;
    padding: 12px;
    font-size: 14px;
    border-top: 1px solid #4CAF50;
}
.footer a { color:#4CAF50; text-decoration:none; margin:0 8px; }
.footer a:hover { text-decoration:underline; }
</style>
<div class="footer">
    📊 Project by <b>Priyanka B</b> | 
    <a href="https://github.com/yourusername" target="_blank">GitHub</a> · 
    <a href="https://linkedin.com/in/priyanka-b-350b672a6" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
