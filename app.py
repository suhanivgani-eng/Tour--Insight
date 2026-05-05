import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import numpy as np
from wordcloud import WordCloud

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Tour Insight AI", layout="wide")

st.title("🌍 Tour Insight - AI Tourism Analytics System")
st.markdown("Smart analysis with sentiment scoring, place insights & heatmaps")

# -----------------------------
# FILE UPLOAD
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
    # TOPIC DETECTION (WITH PLACE ADDED)
    # -----------------------------
    def get_topic(text):
        text = str(text).lower()

        if any(x in text for x in ["palace", "hill", "zoo", "garden", "temple"]):
            return "Place"

        elif "food" in text:
            return "Food"

        elif "transport" in text or "bus" in text or "auto" in text:
            return "Transport"

        elif "clean" in text or "toilet" in text or "waste" in text:
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
    # SENTIMENT SCORE (NUMERIC)
    # -----------------------------
    def sentiment_score(text):
        return TextBlob(str(text)).sentiment.polarity

    df["Score"] = df["review"].apply(sentiment_score)

    # -----------------------------
    # SIDEBAR FILTERS
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
    # PLACE SCORE ENGINE
    # -----------------------------
    def place_score(data):
        place_df = data[data["Topic"] == "Place"]

        if len(place_df) == 0:
            return None

        return round(np.mean(place_df["Score"]) * 100, 2)

    # -----------------------------
    # DASHBOARD METRICS
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", len(df))
    col2.metric("Filtered Reviews", len(filtered_df))
    col3.metric("Avg Sentiment Score", round(df["Score"].mean() * 100, 2))
    col4.metric("Place Score", place_score(df) if place_score(df) else "N/A")

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Analytics", "🔥 Heatmap", "💡 Insights"])

    # -----------------------------
    # TAB 1 - DATA
    # -----------------------------
    with tab1:
        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df)

        st.download_button(
            "⬇ Download Data",
            filtered_df.to_csv(index=False).encode('utf-8'),
            "tour_insight_data.csv",
            "text/csv"
        )

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
    # TAB 3 - HEATMAP (NEW)
    # -----------------------------
    with tab3:

        st.subheader("🔥 Tourism Satisfaction Heatmap")

        heat_data = df.pivot_table(
            index="Topic",
            values="Score",
            aggfunc="mean"
        )

        fig3, ax3 = plt.subplots()
        sns.heatmap(heat_data, annot=True, cmap="RdYlGn", ax=ax3)

        st.pyplot(fig3)

        st.info("Green = High Satisfaction | Red = Low Satisfaction")

    # -----------------------------
    # SMART INSIGHTS
    # -----------------------------
    def get_recommendations(topic):

        topic = str(topic).lower()

        if topic == "cleanliness":
            return ["Increase cleaning frequency", "Smart dustbins", "Toilet monitoring"]

        elif topic == "transport":
            return ["Shuttle buses", "Fixed fare system", "Better routes"]

        elif topic == "pricing":
            return ["Control overpricing", "Price boards", "Monitoring system"]

        elif topic == "food":
            return ["Improve hygiene", "Food inspections", "Clean zones"]

        elif topic == "guidance":
            return ["QR guides", "AI chatbot", "Better maps"]

        elif topic == "place":
            return ["Improve tourist attraction maintenance", "Crowd management", "Better facilities"]

        else:
            return ["Improve overall tourism experience"]

    # -----------------------------
    # TAB 4 - INSIGHTS
    # -----------------------------
    with tab4:

        st.subheader("🧠 Smart Tourism Insights")

        top_topic = df["Topic"].value_counts().idxmax()

        st.success(f"Main Issue Area: {top_topic}")

        st.subheader("🚀 Recommendations")

        for r in get_recommendations(top_topic):
            st.info("👉 " + r)

        st.subheader("❌ Negative Reviews")
        st.write(filtered_df[filtered_df["Sentiment"] == "Negative"]["review"].head(5))

else:
    st.info("📂 Upload CSV file to start analysis")
