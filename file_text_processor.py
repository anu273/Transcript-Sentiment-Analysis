import re
import io
import PyPDF2
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Ensure necessary NLTK resources are downloaded
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

#pdf_file = r"C:\Users\anami\Documents\Projects\FinTech\2 Neuland-Q1-FY17.pdf"


# Load FinBERT (HuggingFace model)
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

FORWARD_LOOKING_INDICATORS = {
    'expect', 'anticipate', 'forecast', 'guidance', 'outlook', 'future', 'will',
    'plan', 'intend', 'project', 'estimate', 'target', 'goal', 'objective',
    'strategy', 'initiative', 'upcoming', 'next quarter', 'next year', 'going forward'
}


def extract_text_from_pdf(pdf_bytes):
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        page_text = page.extract_text()
        # Clean up common PDF extraction issues
        page_text = re.sub(r'\n+', ' ', page_text)
        page_text = re.sub(r'\s+', ' ', page_text)
        text += page_text + " "
    
    return text.strip()

def clean_text(text):
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())

    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return ' '.join(stemmed_tokens)

def get_sentiment_score(text, use_finbert=True):
    text = clean_text(text)
    analyzer = SentimentIntensityAnalyzer()
    base_scores = analyzer.polarity_scores(text)

    vader_compound = base_scores['compound']
    adjusted_compound = vader_compound  # default to VADER only

    finbert_compound = 0.0
    if use_finbert:
        # Tokenize and run through FinBERT
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # FinBERT labels: 0 = negative, 1 = neutral, 2 = positive
        finbert_compound = probs[2] - probs[0]  # Positive - Negative

        # Blend: 60% VADER + 40% FinBERT
        adjusted_compound = 0.6 * vader_compound + 0.4 * finbert_compound

    # Final sentiment label
    if adjusted_compound >= 0.1:
        label = "Bullish"
    elif adjusted_compound <= -0.1:
        label = "Bearish"
    else:
        label = "Neutral"

    # Risk score (using same logic as before)
    risk_words = ['risk', 'uncertainty', 'volatile', 'challenge', 'concern', 'pressure']
    risk_count = sum(text.lower().count(word) for word in risk_words)
    risk_score = min(risk_count / 100.0, 1.0)

    return {
        "positive": base_scores['pos'],
        "negative": base_scores['neg'],
        "neutral": base_scores['neu'],
        "compound": adjusted_compound,
        "label": label,
        "risk_score": risk_score,
        "finbert_score": finbert_compound,
        "vader_score": vader_compound
    }


def extract_forward_looking_statements(text):
    """Extract and analyze forward-looking statements."""
    sentences = sent_tokenize(text)
    forward_looking_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in FORWARD_LOOKING_INDICATORS):
            forward_looking_sentences.append(sentence)
    
    if forward_looking_sentences:
        return forward_looking_sentences[:5]  # Top 5 statements
    
    return []

def calculate_sentiment_volatility(text, window_size=100):
    """Calculate sentiment volatility across the transcript."""
    words = text.split()
    if len(words) < window_size * 2:
        return 0
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    # Calculate sentiment for sliding windows
    for i in range(0, len(words) - window_size + 1, window_size // 2):
        window_text = ' '.join(words[i:i + window_size])
        score = analyzer.polarity_scores(window_text)['compound']
        sentiment_scores.append(score)
    
    # Calculate standard deviation as volatility measure
    if len(sentiment_scores) > 1:
        return float(np.std(sentiment_scores))
    else:
        return 0
    
def analyze_sentiment_trends(text, window_size=200):
    """Analyze sentiment trends across the document"""
    words = text.split()
    if len(words) < window_size:
        window_size = max(len(words) // 4, 50)
    
    trends = []
    step_size = max(window_size // 4, 50)
    
    for i in range(0, len(words) - window_size + 1, step_size):
        window_text = ' '.join(words[i:i + window_size])
        sentiment = get_sentiment_score(window_text)
        
        trends.append({
            'position': i / len(words) * 100,  # Position as percentage
            'positive': sentiment['positive'],
            'negative': sentiment['negative'],
            'neutral': sentiment['neutral'],
            'compound': sentiment['compound'],
            'label': sentiment['label']
        })
    
    return pd.DataFrame(trends)

def cumulative_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    cumulative = []
    total = 0
    for i, sentence in enumerate(sentences):
        score = analyzer.polarity_scores(sentence)['compound']
        total += score
        cumulative.append({'index': i, 'cumulative_score': total})
    return cumulative

    
#if __name__ == "__main__":
    #text = extract_text_from_pdf(pdf_file)
    #text = clean_text(text)
    #dic = extract_forward_looking_statements(text)

    #print(dic)  
