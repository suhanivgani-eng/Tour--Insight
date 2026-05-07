import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np

# -----------------------------
# SAFE SEABORN IMPORT (FIX)
# -----------------------------
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except:
    SEABORN_AVAILABLE = False

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Tour Insight AI", layout="wide")

st.title("🌍 Tour Insight - Smart Tourism Analytics System")
st.markdown("AI-based sentiment analysis with insights and heatmap dashboard")

# -----------------------------
# UPLOAD FILE
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Tourist Reviews CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("CSV must contain 'review' column")
        st.stop()

    # -----------------------------
    # SENTIMENT ANALYSIS
    # -----------------------------
    def get_sentiment(text):
        score = TextBlob(str(text)).sentiment.polarity
        if score > 0:
            return "Positive"
        elif score == 0:
            return "Neutral"
        else:
            return "Negative"

    # -----------------------------
    # TOPIC DETECTION (INCLUDING PLACE)
    # -----------------------------
    def get_topic(text):
        text = str(text).lower()

        if any(x in text for x in ["palace", "hill", "zoo", "garden", "temple", "beach"]):
            return "Place"
        elif "food" in text:
            return "Food"
        elif "transport" in text or "bus" in text or "auto" in text:
            return "Transport"
        elif "clean" in text or "toilet" in text:
            return "Cleanliness"
        elif "price" in text or "cost" in text or "expensive" in text:
            return "Pricing"
        elif "guide" in text or "map" in text:
            return "Guidance"
        else:
            return "General"

    df["Sentiment"] = df["review"].apply(get_sentiment)
    df["Topic"] = df["review"].apply(get_topic)

    # -----------------------------
    # NUMERIC SCORE
    # -----------------------------
    def sentiment_score(text):
        return TextBlob(str(text)).sentiment.polarity

    df["Score"] = df["review"].apply(sentiment_score)

    # -----------------------------
    # FILTERS
    # -----------------------------
    st.sidebar.title("🔍 Filters")

    topic_filter = st.sidebar.selectbox(
        "Select Topic",
        ["All", "Food", "Transport", "Cleanliness", "Pricing", "Guidance", "Place", "General"]
    )

    sentiment_filter = st.sidebar.selectbox(
        "Select Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    filtered_df = df.copy()

    if topic_filter != "All":
        filtered_df = filtered_df[filtered_df["Topic"] == topic_filter]

    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", len(df))
    col2.metric("Filtered Reviews", len(filtered_df))
    col3.metric("Avg Sentiment Score", round(df["Score"].mean() * 100, 2))

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Analytics", "🔥 Heatmap", "💡 Insights"])

    # -----------------------------
    # TAB 1
    # -----------------------------
    with tab1:
        st.subheader("Dataset")
        st.dataframe(filtered_df)

    # -----------------------------
    # TAB 2 - ANALYTICS
    # -----------------------------
    with tab2:

        st.subheader("Sentiment Distribution")
        fig, ax = plt.subplots()
        filtered_df["Sentiment"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        st.subheader("Word Cloud")
        text = " ".join(filtered_df["review"].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array())

        st.subheader("Topic Distribution")
        fig2, ax2 = plt.subplots()
        filtered_df["Topic"].value_counts().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    # -----------------------------
    # TAB 3 - HEATMAP (SAFE FIX)
    # -----------------------------
    with tab3:

        st.subheader("🔥 Tourist Satisfaction Heatmap")

        heat_data = df.pivot_table(
            index="Topic",
            values="Score",
            aggfunc="mean"
        )

        if SEABORN_AVAILABLE:
            fig3, ax3 = plt.subplots()
            sns.heatmap(heat_data, annot=True, cmap="RdYlGn", ax=ax3)
            st.pyplot(fig3)
        else:
            st.warning("Seaborn not installed, showing simple chart instead")

            st.bar_chart(heat_data)

        st.info("Green = High Satisfaction | Red = Low Satisfaction")

    # -----------------------------
    # SMART INSIGHTS
    # -----------------------------
    def recommendations(topic):

        topic = str(topic).lower()

        if topic == "cleanliness":
            return ["Improve sanitation", "Smart bins", "Frequent cleaning"]
        elif topic == "transport":
            return ["Better buses", "Fix fares", "Improve routes"]
        elif topic == "pricing":
            return ["Control pricing", "Display boards", "Monitoring system"]
        elif topic == "food":
            return ["Hygiene checks", "Clean food zones", "Regular inspections"]
        elif topic == "guidance":
            return ["QR guides", "AI chatbot", "Better maps"]
        elif topic == "place":
            return ["Improve attractions", "Crowd control", "Better facilities"]
        else:
            return ["Improve tourism experience"]

    # -----------------------------
    # TAB 4
    # -----------------------------
    with tab4:

        st.subheader("🧠 Smart Insights")

        top_topic = df["Topic"].value_counts().idxmax()

        st.success(f"Main Issue: {top_topic}")

        for r in recommendations(top_topic):
            st.info("👉 " + r)

        st.subheader("❌ Negative Reviews")
        st.write(filtered_df[filtered_df["Sentiment"] == "Negative"]["review"].head(5))

else:
    st.info("📂 Upload CSV file to start analysis")
