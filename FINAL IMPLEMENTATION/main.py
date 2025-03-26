import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import time
from statsmodels.tsa.arima.model import ARIMA

# API key
API_KEY = "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"

# API Configuration
API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

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

# Load price data
price_data = pd.read_csv("prices_dataset.csv")
price_data = price_data[price_data['product_name'] == product]

# Extract competitor names from URLs
def extract_competitor(url):
    if "amazon" in url.lower():
        return "Amazon"
    elif "flipkart" in url.lower():
        return "Flipkart"
    else:
        return None

price_data["competitor"] = price_data["source"].apply(extract_competitor)

# Filter only latest price
latest_prices = price_data.sort_values(by="date", ascending=False).drop_duplicates(subset=["competitor"])

# Reset index for proper serial numbering
latest_prices.reset_index(drop=True, inplace=True)
latest_prices.index += 1  

# Competitor Analysis Table
st.subheader(f"Competitor Analysis for {product}")
if latest_prices.empty:
    st.warning("No competitor price and discount data available!")
else:
    st.dataframe(latest_prices[['competitor', 'price', 'discount']], use_container_width=True)

# Price & Discount Trends Graph (Toggle)
st.subheader("Price & Discount Trends")

# Datetime format
price_data["date"] = pd.to_datetime(price_data["date"], format="%d-%m-%Y", errors="coerce")

# Select graph type
graph_type = st.radio("Select Trend to Display:", ["Price Trend", "Discount Trend"], horizontal=True)

fig, ax = plt.subplots(figsize=(10, 5))
for competitor in price_data["competitor"].unique():
    competitor_data = price_data[price_data["competitor"] == competitor].sort_values("date")
    if graph_type == "Price Trend":
        ax.plot(competitor_data["date"], competitor_data["price"], label=competitor)
    else:
        ax.plot(competitor_data["date"], competitor_data["discount"], label=competitor)

ax.set_xlabel("Date")
ax.set_ylabel("Price" if graph_type == "Price Trend" else "Discount")
ax.set_title(f"{graph_type} for {product}")
ax.legend()
st.pyplot(fig)

# Load reviews data
reviews_data = pd.read_csv("reviews_dataset.csv")
reviews_data = reviews_data[reviews_data['product_name'] == product]

# Sentiment Analysis (Using LLM)
st.subheader("Customer Sentiment Analysis")

batch_size = 5
reviews_list = reviews_data["reviews"].tolist()
batched_reviews = [reviews_list[i:i + batch_size] for i in range(0, len(reviews_list), batch_size)]

sentiments = {"Positive": 0, "Negative": 0}

for batch in batched_reviews:
    try:
        messages = [{"role": "system", "content": "Classify each review as positive or negative. Return JSON: {\"sentiments\": [\"positive\", \"negative\", ...]}"}]
        for review in batch:
            messages.append({"role": "user", "content": review})

        data = {
            "model": "qwen-2.5-32b",
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))

        if response.status_code == 200:
            response_json = response.json()
            sentiments_list = json.loads(response_json["choices"][0]["message"]["content"])["sentiments"]

            for sentiment in sentiments_list:
                if sentiment.lower() == "positive":
                    sentiments["Positive"] += 1
                elif sentiment.lower() == "negative":
                    sentiments["Negative"] += 1
        else:
            st.error(f"LLM API Error ({response.status_code}): {response.text}")

        time.sleep(2)  

    except Exception as e:
        st.error(f"Error fetching sentiment analysis: {str(e)}")

fig, ax = plt.subplots()
ax.bar(sentiments.keys(), sentiments.values(), color='skyblue')
ax.set_ylabel("Count")
ax.set_xlabel("Sentiment")
ax.set_title("Sentiment Analysis Results")
st.pyplot(fig)

# Price & Discount Prediction Table (ARIMA)
st.subheader("Price & Discount Prediction for Next 3 Days")

p, d, q = 5, 1, 0
predicted_prices = {}
predicted_discounts = {}

for competitor in latest_prices["competitor"].unique():
    comp_data = price_data[price_data["competitor"] == competitor].sort_values("date")

    if not comp_data.empty:
        try:
            comp_data = comp_data.set_index("date")
            comp_data.index = pd.DatetimeIndex(comp_data.index).to_period("D")

            model_price = ARIMA(comp_data["price"], order=(p, d, q)).fit()
            model_discount = ARIMA(comp_data["discount"], order=(p, d, q)).fit()

            predicted_prices[competitor] = model_price.forecast(steps=3).tolist()
            predicted_discounts[competitor] = model_discount.forecast(steps=3).tolist()
        except Exception as e:
            st.error(f"Prediction failed for {competitor}: {e}")

future_dates = ["Day 1", "Day 2", "Day 3"]
prediction_table = []

for competitor in predicted_prices.keys():
    for i in range(3):
        prediction_table.append([competitor, future_dates[i], round(predicted_prices[competitor][i], 2), round(predicted_discounts[competitor][i], 2)])

prediction_df = pd.DataFrame(prediction_table, columns=["Competitor", "Day", "Predicted Price", "Predicted Discount"])
prediction_df.index += 1

st.dataframe(prediction_df, use_container_width=True)

# Strategic Recommendations 
st.subheader("Strategic Recommendations")

# Prepare data for LLM
product_name = product
competitor_data = latest_prices.to_dict(orient="records")
sentiment_analysis = {"positive": sentiments.get("Positive", 0), "negative": sentiments.get("Negative", 0)}
date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Prompt with structure + JSON requirement
prompt = f"""
You are a top e-commerce strategist. Analyze this data and provide detailed, actionable recommendations in JSON format.

Product: {product_name}
Current Date: {date}

Competitor Analysis:
{json.dumps(competitor_data, indent=2)}

Sentiment Analysis:
Positive: {sentiment_analysis['positive']}
Negative: {sentiment_analysis['negative']}

Provide extremely detailed recommendations with these sections:

1. Pricing Strategy:
   - Current market position
   - Specific price adjustment recommendations
   - Discount strategy
   - Implementation timeline

2. Promotional Campaigns:
   - Campaign names and details
   - Target audience
   - Duration and execution plan
   - Expected outcomes

3. Customer Satisfaction:
   - Key pain points to address
   - Specific improvement programs
   - Customer retention strategies
   - Measurement metrics

Format the response as JSON with this structure:
{{
    "pricing_strategy": {{
        "analysis": "text",
        "recommendations": [
            {{"action": "text", "details": "text", "timeline": "text"}}
        ]
    }},
    "promotions": {{
        "campaigns": [
            {{"name": "text", "details": "text", "duration": "text", "target": "text"}}
        ]
    }},
    "customer_satisfaction": {{
        "improvements": [
            {{"area": "text", "action": "text", "metrics": "text"}}
        ]
    }}
}}
"""

data = {
    "model": "qwen-2.5-32b",
    "messages": [{"role": "system", "content": prompt}],
    "temperature": 0.7,
    "response_format": {"type": "json_object"}
}

response = requests.post(API_URL, headers=HEADERS, json=data)
if response.status_code == 200:
    try:
        response_json = response.json()
        strategy_output = json.loads(response_json["choices"][0]["message"]["content"])
        
        # Formatted recommendations
        st.markdown("### 📈 Pricing Strategy")
        st.write(strategy_output["pricing_strategy"]["analysis"])
        for rec in strategy_output["pricing_strategy"]["recommendations"]:
            st.markdown(f"- **{rec['action']}**: {rec['details']} (Timeline: {rec.get('timeline', 'TBD')})")
        
        st.markdown("### 🎯 Promotional Campaigns")
        for campaign in strategy_output["promotions"]["campaigns"]:
            st.markdown(f"#### {campaign['name']}")
            st.write(f"**Details**: {campaign['details']}")
            st.write(f"**Duration**: {campaign['duration']} | **Target**: {campaign['target']}")
        
        st.markdown("### 😊 Customer Satisfaction")
        for improvement in strategy_output["customer_satisfaction"]["improvements"]:
            st.markdown(f"- **{improvement['area']}**: {improvement['action']}")
            st.write(f"Success metrics: {improvement['metrics']}")
            
    except Exception as e:
        st.error(f"Error processing recommendations: {str(e)}")
        st.json(response_json)  
else:
    error_message = response.json().get("error", {}).get("message", "Unknown API Error")
    st.error(f"API Error: {error_message}")