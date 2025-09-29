import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime

# Use caching to avoid re-downloading stock data on every page rerun
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Downloads daily stock data and returns a DataFrame."""
    try:
        stock_df = yf.download(ticker, start=start_date, end=end_date, progress=False, interval='1mo')
        return stock_df
    except Exception as e:
        st.error(f"Failed to download stock data for {ticker}. Error: {e}")
        return None

def extract_quarter_and_year(filename):
    """Extracts Qx and YYYY from a filename using regex."""
    # Look for Q1-Q4
    quarter_match = re.search(r'Q([1-4])', filename, re.IGNORECASE)
    # Look for FYxx
    fy_match = re.search(r'FY(\d{2})', filename, re.IGNORECASE)

    if quarter_match and fy_match:
        quarter = f"Q{quarter_match.group(1)}"
        year_suffix = int(fy_match.group(1))
        # Convert FY18 -> 2018, FY23 -> 2023
        year = 2000 + year_suffix if year_suffix < 50 else 1900 + year_suffix
        return f"{year}-{quarter}"
    
    return None

def create_comparison_chart(merged_df, ticker, column):
    """Creates a dual-axis chart comparing stock price and negative sentiment."""
    if merged_df.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Line chart for Sentiment Score (Right Y-axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=merged_df[column],
            name=column,
            mode='lines+markers',
            line=dict(color='lightcoral', width=3),
            connectgaps=True 
        ),
        secondary_y=True,
    )

    # Add Line chart for Quarterly Average Stock Price (Left Y-axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['Date'],
            y=merged_df['Close'],
            name='Avg. Quarterly Stock Price',
            mode='lines+markers',
            line=dict(color='royalblue', width=3)
        ),
        secondary_y=False,
    )

    # Update layout and axis titles
    fig.update_layout(
        title_text=f"<b>{ticker}: Avg. Quarterly Stock Price vs. {column}</b>",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, borderwidth=1),
        height=500
    )
    fig.update_xaxes(title_text="Quarter")
    fig.update_yaxes(title_text="<b>Avg. Quarterly Stock Price (USD)</b>", secondary_y=False, color='royalblue')
    fig.update_yaxes(title_text=column, secondary_y=True, color='lightcoral', range=[0, max(0.1, merged_df[column].max() * 1.1)])

    return fig

def main():
    st.set_page_config(
        page_title="Stock & Sentiment Correlation",
        page_icon="🔗",
        layout="wide"
    )

    st.title("🔗 Stock Price & Sentiment Correlation")

    # --- Load Data from Session State ---
    analysis_results = st.session_state.get("analysis_results")
    ticker = st.session_state.get("ticker")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")
    summary_df = st.session_state.get("summary_df")
    
    # --- Validate that necessary data exists ---
    if not all([analysis_results, ticker, start_date, end_date]):
        st.warning("⚠️ Analysis data not found. Please go to the Home page, upload files, and run an analysis first.")
        st.page_link("home.py", label="⬅️ Go to Home Page", icon="🏠")
        return

    # --- 1. Process Sentiment Data ---
    sentiment_data = []
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            quarter_year = extract_quarter_and_year(row["File"])
            if quarter_year:
                sentiment_data.append({
                    "QuarterYear": quarter_year,
                    "Positive_Score": row["Positive_Score"],
                    "Negative_Score": row["Negative_Score"],
                    "Compound_Score": row["Compound_Score"],
                    "risk_score": row["Risk_Score"],
                    "word_count": row["Word_Count"],
                    "File": row["File"]
                })

    if not sentiment_data:
        st.error("Could not extract any valid quarters (e.g., 'Q1-2023') from the uploaded filenames. Please rename your files and try again.")
        return

    sentiment_df = pd.DataFrame(sentiment_data)
    # Convert QuarterYear string to a proper timestamp for plotting (end of quarter)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['QuarterYear'].str.replace(r'(Q\d)-(\d{4})', r'\2-\1', regex=True))
    sentiment_df = sentiment_df.sort_values('Date').set_index('Date')


    # --- 2. Process Stock Data ---
    stock_df = get_stock_data(ticker, start_date, end_date)

    if stock_df is None:
        st.error(f"Could not retrieve stock data for '{ticker}'. Please check the ticker symbol and date range.")
        return

    if isinstance(stock_df.columns, pd.MultiIndex):
        stock_df.columns = [col[0] for col in stock_df.columns]

    price_df = stock_df.reset_index()[['Date', 'Close']]


    merged_df = pd.merge(price_df, sentiment_df, on="Date", how="left").sort_values("Date")
   
    # --- 3. Create and Display Charts ---
    st.subheader(f"📊 {ticker}: Stock Price vs. Sentiment Scores")
    Negative_Sentiment  = create_comparison_chart(merged_df, ticker, "Negative_Score")
    st.plotly_chart(Negative_Sentiment, use_container_width=True)

    Positive_Sentiment  = create_comparison_chart(merged_df, ticker, "Positive_Score")
    st.plotly_chart(Positive_Sentiment, use_container_width=True)

    Risk = create_comparison_chart(merged_df, ticker, "risk_score")
    st.plotly_chart(Risk, use_container_width=True)

    word = create_comparison_chart(merged_df, ticker, "word_count")
    st.plotly_chart(word, use_container_width=True)

    st.markdown("---")
    st.header("📋 Data Tables")
    st.dataframe(stock_df, use_container_width=True)
    st.dataframe(sentiment_df, use_container_width=True)
    
if __name__ == "__main__":
    main()