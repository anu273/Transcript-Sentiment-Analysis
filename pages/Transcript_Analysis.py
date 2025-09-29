import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from file_text_processor import (
    extract_text_from_pdf,  
    get_sentiment_score,
    extract_forward_looking_statements,
    calculate_sentiment_volatility
)


def extract_key_phrases(text, sentiment_type="positive"):
    """Extract key phrases based on sentiment type"""
    try:
        sentences = sent_tokenize(text)
        key_sentences = []
        
        for sentence in sentences:
            sentiment = get_sentiment_score(sentence)
            if sentiment_type == "positive" and sentiment['compound'] > 0.3:
                key_sentences.append(sentence)
            elif sentiment_type == "negative" and sentiment['compound'] < -0.3:
                key_sentences.append(sentence)
            elif sentiment_type == "neutral" and abs(sentiment['compound']) <= 0.1:
                key_sentences.append(sentence)
        
        return sorted(key_sentences, key=lambda x: abs(get_sentiment_score(x)['compound']), reverse=True)[:5]
    except:
        return []

def create_sentiment_comparison_chart(results_df):
    """Create comparison chart for multiple files"""
    fig = go.Figure()
    
    # Add bars for each sentiment
    fig.add_trace(go.Bar(
        name='Positive',
        x=results_df['file_name'],
        y=results_df['positive'],
        marker_color='lightgreen'
    ))
    
    fig.add_trace(go.Bar(
        name='Neutral',
        x=results_df['file_name'],
        y=results_df['neutral'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative',
        x=results_df['file_name'],
        y=results_df['negative'],
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Sentiment Distribution Across Files',
        xaxis_title='Files',
        yaxis_title='Sentiment Score',
        barmode='group',
        height=500
    )
    
    return fig

def create_trend_chart(trends_df, file_name):
    """Create sentiment trend chart for a single file"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trends_df['position'],
        y=trends_df['positive'],
        mode='lines+markers',
        name='Positive',
        line=dict(color='green', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=trends_df['position'],
        y=trends_df['neutral'],
        mode='lines+markers',
        name='Neutral',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=trends_df['position'],
        y=trends_df['negative'],
        mode='lines+markers',
        name='Negative',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=f'Sentiment Trends - {file_name}',
        xaxis_title='Document Progress (%)',
        yaxis_title='Sentiment Score',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_risk_gauge(risk_score):
    """Create a risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': 'lightgreen'},
                {'range': [33, 66], 'color': 'yellow'},
                {'range': [66, 100], 'color': 'lightcoral'}
            ],
        }
    ))
    
    fig.update_layout(height=170, margin=dict(t=10, b=5, l=2, r=2))
    return fig

# Main Dashboard
def main():
    st.set_page_config(
        page_title="Call Sentiment Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    file_data = st.session_state.get("file_data", [])
    ticker = st.session_state.get("ticker", "")

    st.title("📊 Transcript Call Sentiment Analysis")
    st.markdown(f"**Ticker:** `{ticker}` | **Date Range:** `{st.session_state.get('start_date')}` to `{st.session_state.get('end_date')}`")
    st.markdown("---")

    if not file_data or not ticker:
        st.warning("⚠️ No files to analyze. Please go back to the home page to upload files.")
        st.page_link("home.py", label="⬅️ Go to Home Page", icon="🏠")
        return

    

    # --- Display logic now uses the fully pre-computed results ---
    
    results = st.session_state.get('analysis_results', [])

    if results:
        # Overview Section
        st.header("📈 Analysis Overview")
        # (This section is already efficient, no changes needed)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Analyzed", len(results))
        with col2:
            avg_sentiment = np.mean([r['sentiment']['compound'] for r in results])
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        with col3:
            bullish_count = sum(1 for r in results if r['sentiment']['label'] == 'Bullish')
            st.metric("Bullish Documents", bullish_count)
        with col4:
            total_words = sum(r['word_count'] for r in results)
            st.metric("Total Words Analyzed", f"{total_words:,}")

        # Comparison Charts (already efficient)
        if len(results) > 1:
            st.header("📊 Cross-Document Comparison")
            
            comparison_data = []
            for result in results:
                comparison_data.append({
                    'file_name': result['file_name'][:20] + '...' if len(result['file_name']) > 20 else result['file_name'],
                    'positive': result['sentiment']['positive'],
                    'neutral': result['sentiment']['neutral'],
                    'negative': result['sentiment']['negative'],
                    'compound': result['sentiment']['compound'],
                    'risk_score': result['sentiment']['risk_score'],
                    'label': result['sentiment']['label']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            fig_comparison = create_sentiment_comparison_chart(comparison_df)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            fig_scatter = px.scatter(
                comparison_df, x='file_name', y='compound', color='label', size='risk_score',
                title='Overall Sentiment Score by Document',
                color_discrete_map={'Bullish': 'green', 'Bearish': 'red', 'Neutral': 'blue'}
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Individual File Analysis
        st.header("📄 Individual File Analysis")

        for idx, result in enumerate(results):
            with st.expander(f"📋 {result['file_name']}", expanded=len(results) == 1):
                sentiment = result['sentiment']
                
                st.subheader("🔍 Key Metrics")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Overall Sentiment", sentiment['label'])
                with m_col2:
                    st.metric("Compound Score", f"{sentiment['compound']:.3f}")
                with m_col3:
                    st.metric("Risk Level", "High" if sentiment['risk_score'] > 0.66 else "Medium" if sentiment['risk_score'] > 0.33 else "Low")
                with m_col4:
                    st.metric("Sentiment Volatility", f"{result['sentiment_volatility']:.3f}")
                
                st.markdown("---")

                st.subheader("📈 Sentiment Trend Over Time")
                fig_trend = create_trend_chart(result['trends'], result['file_name'])
                st.plotly_chart(fig_trend, use_container_width=True, key=f"trends_{idx}")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("⚠️ Risk Score")
                    fig_gauge = create_risk_gauge(sentiment['risk_score'])
                    st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_chart_{idx}")
                with col2:
                    st.subheader("📊 Sentiment Breakdown")
                    breakdown_data = {'Sentiment': ['Positive', 'Neutral', 'Negative'], 'Score': [sentiment['positive'], sentiment['neutral'], sentiment['negative']]}
                    fig_pie = px.pie(breakdown_data, values='Score', names='Sentiment', color_discrete_map={'Positive': 'green', 'Neutral': 'lightblue', 'Negative': 'red'}, hole=.4)
                    fig_pie.update_layout(height=250, showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{idx}")

                st.markdown("---")

                # Forward-Looking Statements
                st.subheader("🔮 Forward-Looking Statements")
                forward_statements = result.get('forward_looking_statements', [])
                if forward_statements:
                    for i, stmt in enumerate(forward_statements[:5]):
                        st.write(f"{i+1}. {stmt}")
                else:
                    st.warning("No significant forward-looking statements were identified.")

  
        
        summary_df = st.session_state.get('summary_df')
        st.subheader("📊 Summary Table")
        st.dataframe(summary_df, use_container_width=True)
        
        #csv = summary_df.to_csv(index=False)
        #st.download_button(
        #    label="Download CSV Report",
        #    data=csv,
        #    file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        #    mime="text/csv"
        #)
    else:
        # This part remains the same
        st.info("👆 Please upload one or more PDF files to begin the analysis.")
        st.subheader("🎯 What You'll Get:")
        st.markdown("""
        - **Sentiment Trends**: Track how sentiment changes throughout each document
        - **Cross-Document Comparison**: Compare sentiment across multiple files
        - **Risk Assessment**: Identify potential risk indicators in the text
        - **Key Phrase Extraction**: Find the most positive, neutral, and negative statements
        - **Interactive Visualizations**: Explore data with interactive charts
        - **Export Capabilities**: Download analysis results as CSV
        """)

if __name__ == "__main__":
    main()