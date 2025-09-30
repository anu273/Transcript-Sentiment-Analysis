# 📊 Transcript Sentiment Analysis Dashboard

A comprehensive interactive web application built with Streamlit for analyzing sentiment in financial documents, particularly earnings call transcripts. This tool leverages advanced Natural Language Processing (NLP) and machine learning to extract valuable insights from unstructured text and correlate them with stock market performance.

## 🎯 Overview

This application employs a sophisticated hybrid sentiment analysis approach, combining VADER's rule-based methodology (60% weight) with FinBERT's domain-specific financial language understanding (40% weight). This dual-model approach provides nuanced, accurate sentiment scores tailored specifically for financial documents.

## 🏗️ Application Architecture & User Flow

The application follows a three-page architecture with seamless navigation and persistent state management:

### Page 1: Upload & Configuration (`1_Upload_Files.py`)

**Purpose**: Initial data ingestion and preprocessing.

**Key Components**:
- **File Upload Interface**: 
  - Multi-file PDF uploader with drag-and-drop support.
  - Real-time file validation.
  - On-change callback triggers automatic preprocessing.
 
    ![Image](https://github.com/user-attachments/assets/f7bf40fc-2212-4e5f-917c-770c43287698)

- **Stock Data Retrieval**:
  - The user enters a ticker symbol and selects a date range.
  - Ensure that the same company ticker and date range are used as in the uploaded files.
 
    ![Image](https://github.com/user-attachments/assets/78a54d4a-e061-4f6b-87bc-408195bd3ef4)
    
- **Click the "Analyze 🚀" button.**



### Page 2: Transcript Analysis (`pages/2_Transcript_Analysis.py`)

**Purpose**: In-depth analysis of individual and multiple documents.

**Key Components**:
- **Analysis Overview**: High-level metrics (files analyzed, average sentiment, etc.).

  ![Image](https://github.com/user-attachments/assets/4ea118d1-f61c-4e97-9efa-0903deb23f04)
  
- **Cross-Document Comparison**: Bar charts and scatter plots comparing sentiment scores across all uploaded files.
  
  <img width="1512" height="637" alt="Image" src="https://github.com/user-attachments/assets/b24f8922-e433-4aed-8a89-4bc5beb19de8" />
  
  <img width="1365" height="400" alt="Image" src="https://github.com/user-attachments/assets/b748ad21-fe04-450b-a3a2-b5e1d6833aa4" />
  
- **Individual File Analysis**: Expandable sections for each document, featuring:
  - Key metrics (overall sentiment, compound score, risk level).

    ![Image](https://github.com/user-attachments/assets/c68befa2-5693-4b06-b1e4-a8bf3ea4c53f)

  - Sentiment trend over time.
  - Risk gauge and sentiment breakdown pie chart.

    ![Image](https://github.com/user-attachments/assets/fe7ed80d-821e-4dc4-8ad5-3b364b66a12b)

 
  - Extracted forward-looking statements.
 
    ![Image](https://github.com/user-attachments/assets/1ec31aa8-d805-4db4-ac84-cb6319807421)

  - Overall scores and results

    ![Image](https://github.com/user-attachments/assets/68a84776-c98b-42d7-a091-947e97f11a36)

    

### Page 3: Correlation Analysis (`pages/3_Correlation.py`)

**Purpose**: Correlating sentiment data with stock market performance.

**Key Components**:
- **Data Synchronization**: Automatically fetches historical stock data and merges it with sentiment scores based on the quarter and year extracted from filenames.
- **Interactive Charts**: Dual-axis line charts to visualize the relationship between:
  - Stock Price vs. Negative Sentiment
    <img width="1109" height="500" alt="Image" src="https://github.com/user-attachments/assets/0b697c7c-d3e6-4d0a-81e7-34bf0550f937" />
    
  - Stock Price vs. Positive Sentiment
    <img width="1109" height="500" alt="Image" src="https://github.com/user-attachments/assets/7cb207ff-86e2-4b3b-9def-4e25268773b9" />
    
  - Stock Price vs. Risk Score
    <img width="1109" height="500" alt="Image" src="https://github.com/user-attachments/assets/9b551fe8-f24e-4cca-bbed-26eb06c3f531" />
- **Data Tables**: Displays the raw stock data and sentiment dataframes.


## ✨ Key Features

### 📁 Document Processing & Analysis
- **Multi-File Upload**: Process multiple PDF documents simultaneously with batch analysis capabilities
- **Automated Text Extraction**: Robust PDF text extraction with cleanup of common formatting issues
- **Session State Management**: Efficient file handling that prevents redundant processing and maintains analysis state across page navigation

### 🧠 Advanced Sentiment Analysis
- **Hybrid Sentiment Scoring**: Combines VADER (60%) and FinBERT (40%) for optimal financial sentiment detection
- **Multi-Dimensional Metrics**: 
  - Overall sentiment classification (Bullish/Bearish/Neutral)
  - Compound, positive, neutral, and negative scores
  - Individual VADER and FinBERT component scores
- **Risk Assessment**: Automated identification and quantification of risk-related terminology
- **Sentiment Volatility**: Measures consistency of sentiment throughout documents using sliding window analysis

### 📈 Trend & Comparitive Analysis
- **Cross-Document Comparison**: Side-by-side sentiment metrics across multiple documents
- **Aggregated Statistics**: Average sentiment, bullish document count, and total word analysis
- **Visual Comparisons**: Interactive bar charts and scatter plots for multi-file analysis
- **Temporal Sentiment Tracking**: Visualize sentiment evolution from document start to finish
- **Forward-Looking Statement Extraction**: Automatically identifies and extracts key future-oriented statements using 20+ indicator keywords

### 💹 Stock Market Correlation
- **Historical Price Integration**: Fetches quarterly stock data via yfinance API
- **Sentiment-Price Visualization**: Dual-axis charts comparing:
  - Negative sentiment vs. stock price
  - Positive sentiment vs. stock price
  - Risk scores vs. stock price
  - Document length (word count) vs. stock price
- **Quarterly Mapping**: Intelligent extraction of quarter/year information from filenames (e.g., Q1-FY17)
- **Date-Aligned Merging**: Automatically synchronizes sentiment data with corresponding quarterly stock performance

### 🎨 Interactive Visualizations
- **Sentiment Trend Lines**: Multi-series line charts tracking positive, neutral, and negative sentiment
- **Risk Gauges**: Visual indicators with color-coded risk levels (Low/Medium/High)
- **Sentiment Breakdown**: Pie charts showing sentiment distribution
- **Comparative Bar Charts**: Group comparisons across documents
- **Time Series Analysis**: Stock price and sentiment correlation over time

### 📋 Data Export & Reporting
- **Summary Tables**: Comprehensive dataframes with all computed metrics
- **CSV Export Ready**: All analysis results structured for easy export
- **File Metadata**: Tracking of word counts, file names, and processing timestamps


### Core Module

**file_text_processor.py**
- `extract_text_from_pdf()`: PDF parsing with PyPDF2
- `clean_text()`: Text preprocessing with stopword removal and stemming
- `get_sentiment_score()`: Hybrid VADER + FinBERT sentiment analysis
- `extract_forward_looking_statements()`: Forward-looking statement identification
- `calculate_sentiment_volatility()`: Volatility calculation using sliding windows
- `analyze_sentiment_trends()`: Window-based sentiment trend analysis
- `cumulative_sentiment()`: Sentence-level cumulative scoring

## 🛠️ Technology Stack

### Core Framework
- **Streamlit**: Web application framework with session state management

### Data Processing
- **Pandas**: Data manipulation and DataFrame operations
- **NumPy**: Numerical computations and array operations

### Visualization
- **Plotly**: Interactive charts (scatter, line, bar, pie, gauge)
- **Plotly Subplots**: Multi-axis and complex visualizations

### NLP & Machine Learning
- **NLTK**: Tokenization, sentence splitting, stopword removal, stemming
- **VADER Sentiment**: Rule-based sentiment analysis
- **Transformers (Hugging Face)**: FinBERT model integration
- **PyTorch**: Deep learning backend for FinBERT

### Financial Data
- **yfinance**: Historical stock market data retrieval

### Document Processing
- **PyPDF2**: PDF text extraction


## 🙏 Acknowledgments
- FinBERT Model: ProsusAI/finbert for the domain-specific financial sentiment model
- VADER: Hutto & Gilbert for the VADER sentiment analysis tool
- Streamlit: For the excellent Python web framework
- Hugging Face: For the transformers library and model hosting
- Community: All contributors and users providing valuable feedback

## 📄 License
- This project is licensed under the MIT License - see the LICENSE file for details.
