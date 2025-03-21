import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

# Set page config
st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")

# Sidebar - Product selection
st.sidebar.header("Select a Product")
product = st.sidebar.selectbox("Choose a product to analyze:", [
    "Samsung Galaxy Z Flip 6", "Samsung Galaxy Z Fold 6", "Apple iPhone 16 Pro Max", "OnePlus 13", "Apple Watch Series 9",
    "Apple MacBook Air Laptop", "Apple AirPods Pro (2nd Generation)", "NIKE Mens Jordan Stay Loyal 3 Running Shoes",
    "Premium Aquatic Eau De Cologne", "Samsung S24 Ultra", "Redmi Note 13 Pro", "L'Oreal Paris Hyaluron Moisture 72HR Moisture Filling Shampoo",
    "Lakmé 9 to 5 Kajal Twin Pack", "Puma Women's Pacific Maze Sneaker"
])

# Title
st.title("E-Commerce Competitor Strategy Dashboard")

# Load price data from CSV
price_data = pd.read_csv("prices_dataset.csv")
price_data = price_data[price_data['product_name'] == product]

# Extract competitor names from URLs
def extract_competitor(url):
    if "amazon" in url.lower():
        return "Amazon"
    elif "flipkart" in url.lower():
        return "Flipkart"
    else:
        return "Other"

price_data["competitor"] = price_data["source"].apply(extract_competitor)

# Competitor Analysis Table
st.subheader(f"Competitor Analysis for {product}")
st.dataframe(price_data[['competitor', 'price', 'discount', 'date', 'source']], use_container_width=True)

# Plot price trends over time
price_data['date'] = pd.to_datetime(price_data['date'], format="%d-%m-%Y", errors='coerce')  # Ensure 'date' is in datetime format
price_data_sorted = price_data.sort_values(by='date')

fig, ax = plt.subplots(figsize=(10, 5))
for competitor in price_data['competitor'].unique():
    competitor_data = price_data_sorted[price_data_sorted['competitor'] == competitor]
    ax.plot(competitor_data['date'], competitor_data['price'], label=competitor)
    
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.set_title(f"Price Trend for {product}")
ax.legend()
st.pyplot(fig)

# Load reviews data from CSV
reviews_data = pd.read_csv("reviews_dataset.csv")
reviews_data = reviews_data[reviews_data['product_name'] == product]

# Sentiment Analysis
positive_keywords = ["good", "great", "excellent", "amazing", "love", "best", "awesome"]
negative_keywords = ["bad", "worst", "poor", "terrible", "awful", "disappointed", "hate"]

def analyze_sentiment(review):
    review = str(review).lower()
    if any(word in review for word in positive_keywords):
        return "Positive"
    elif any(word in review for word in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

reviews_data["sentiment"] = reviews_data["reviews"].apply(analyze_sentiment)

# Customer Sentiment Analysis (Bar Chart)
sentiment_counts = reviews_data['sentiment'].value_counts()

fig, ax = plt.subplots()
ax.bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')
ax.set_ylabel("Count")
ax.set_xlabel("Sentiment")
ax.set_title("Sentiment Analysis Results")
st.subheader("Customer Sentiment Analysis")
st.pyplot(fig)

# Strategic Recommendations
st.subheader("Strategic Recommendations")
st.write(
    f"To develop competitive strategies for {product} based on the Competitor Data and Sentiment Analysis, "
    "we should focus on three major aspects: pricing, promotions, and customer satisfaction."
)
