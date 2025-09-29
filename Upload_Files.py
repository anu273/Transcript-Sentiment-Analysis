import streamlit as st
from datetime import date
import datetime as dt
import pandas as pd
from file_text_processor import (
    extract_text_from_pdf,  
    get_sentiment_score,
    extract_forward_looking_statements,
    calculate_sentiment_volatility,
    analyze_sentiment_trends
)

st.set_page_config(
    page_title="Transcript Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

def clear_and_prepare_files():

    # Clear any previous analysis results to ensure a fresh start
    for key in ['analysis_results', 'summary_df', 'analysis_complete', 'file_data']:
        if key in st.session_state:
            del st.session_state[key]
    
    # Pre-process and store the content of the newly uploaded files
    if st.session_state.get("file_uploader_key"):
        uploaded_files = st.session_state["file_uploader_key"]
        file_data = []
        for uploaded_file in uploaded_files:
            # Read the file content ONCE and store the bytes
            file_data.append({
                'name': uploaded_file.name,
                'content': uploaded_file.getvalue() # Use getvalue() to get bytes
            })
        # Store the prepared data in session state for the analysis page
        st.session_state["file_data"] = file_data


st.title("📊 Transcript Call Sentiment Analysis")
st.markdown("Upload financial documents (PDF) to analyze sentiment trends and extract insights.")

# Use a key for the file uploader to access its state in the callback
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files for sentiment analysis",
    on_change=clear_and_prepare_files,
    key="file_uploader_key" 
)

file_data = st.session_state.get("file_data", [])
# Use the pre-processed file_data from session_state
if st.session_state.get("file_data"):
    ticker = st.text_input("Enter stock ticker (e.g., AAPL)", key="ticker_input")
    
    # Set calendar bounds
    min_date = dt.date(1980, 1, 1)   # earliest selectable
    max_date = dt.date.today()       # latest selectable

    start_date = st.date_input(
        "Start Date",
        value=dt.date(2020, 1, 1),              # default shown
        min_value=min_date,          # earliest selectable date
        max_value=max_date,          # latest selectable date
        key="start_date_input"
    )

    end_date = st.date_input(
        "End Date",
        value=max_date,              # default shown
        min_value=min_date,
        max_value=max_date,
        key="end_date_input"
    )
    #start_date = st.date_input("Start Date", key="start_date_input")
    #end_date = st.date_input("End Date", key="end_date_input")

    if ticker and st.button("Analyze 🚀"):
        # Save ticker and dates to session_state
        st.session_state["ticker"] = ticker
        st.session_state["start_date"] = start_date
        st.session_state["end_date"] = end_date
        
        # Set a flag to indicate that a new analysis should be run
        st.session_state["analysis_complete"] = False
        
        results = []
        if not st.session_state.get("analysis_complete", False):
            progress_bar = st.progress(0, text="Initializing analysis...")
            
            for i, file_info in enumerate(file_data):
                file_name = file_info['name']
                pdf_bytes = file_info['content']
                
                progress_bar.text(f"Processing {file_name}...")
                text = extract_text_from_pdf(pdf_bytes)
                
                if text:
                    # --- Pre-calculate EVERYTHING needed for display ---
                    overall_sentiment = get_sentiment_score(text, use_finbert=True)
                    trends_df = analyze_sentiment_trends(text)
                    sentiment_volatility = calculate_sentiment_volatility(text)
                    forward_statements = extract_forward_looking_statements(text)

                    
                    results.append({
                        'file_name': file_name,
                        'text': text,
                        'sentiment': overall_sentiment,
                        'trends': trends_df,
                        'word_count': len(text.split()),
                        'sentiment_volatility': sentiment_volatility,
                        'forward_looking_statements': forward_statements,
                    })
                
                progress_bar.progress((i + 1) / len(file_data))
            
            st.session_state['analysis_results'] = results
            st.session_state['analysis_complete'] = True
            progress_bar.empty()

        if results:
            summary_data = []
            for idx, result in enumerate(results):
                summary_data.append({
                        'File': result['file_name'],
                        'Overall_Sentiment': result['sentiment']['label'],
                        'Compound_Score': result['sentiment']['compound'],
                        'Positive_Score': result['sentiment']['positive'],
                        'Neutral_Score': result['sentiment']['neutral'],
                        'Negative_Score': result['sentiment']['negative'],
                        'Risk_Score': result['sentiment']['risk_score'],
                        'Volatility_Score': result['sentiment_volatility'],
                        'Word_Count': result['word_count']
                    })
                
            summary_df = pd.DataFrame(summary_data)
            st.session_state['summary_df'] = summary_df
        # Switch to the analysis page
        st.switch_page("pages/Transcript_Analysis.py")