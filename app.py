import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🌍 Tour Insight - Smart Tourism Feedback Analysis")
st.subheader("AI-powered Sentiment & Feedback Analysis System")

# -----------------------------
# FUNCTION: SENTIMENT ANALYSIS
# -----------------------------
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# -----------------------------
# INPUT SECTION
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Tourist Reviews CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    # Assume column name is 'review'
    if 'review' not in df.columns:
        st.error("CSV must contain a column named 'review'")
    else:

        # -----------------------------
        # SENTIMENT ANALYSIS
        # -----------------------------
        df["Sentiment"] = df["review"].apply(get_sentiment)

        st.write("### 🧠 Analyzed Data")
        st.dataframe(df)

        # -----------------------------
        # SENTIMENT COUNTS
        # -----------------------------
        sentiment_counts = df["Sentiment"].value_counts()

        # -----------------------------
        # PIE CHART
        # -----------------------------
        st.write("### 📊 Sentiment Distribution")

        fig1, ax1 = plt.subplots()
        ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        ax1.axis("equal")
        st.pyplot(fig1)

        # -----------------------------
        # BAR GRAPH
        # -----------------------------
        st.write("### 📈 Sentiment Count Graph")

        fig2, ax2 = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax2)
        st.pyplot(fig2)

        # -----------------------------
        # INSIGHTS
        # -----------------------------
        st.write("### 💡 Key Insights")

        st.success(f"Most reviews are: {sentiment_counts.idxmax()}")
        st.info("This helps tourism authorities understand overall visitor satisfaction.")

        negative_reviews = df[df["Sentiment"] == "Negative"]["review"]

        st.write("### ⚠️ Sample Negative Feedback")
        st.write(negative_reviews.head(5))
else:
    st.info("Please upload a CSV file with tourist reviews.")