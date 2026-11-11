# 📈 NewsPulse: Financial News Sentiment & Market Reaction Predictor

> **Built for HackSphere — Build and Learn (Aayam XI)**  
> **Organised by:** DSAI Club, NIT Agartala  
> **Track:** Open Innovation (AI & Financial Data Science)

---

## 1. Problem Statement
Everyday retail investors and students read breaking financial headlines about companies like **Tata, Reliance, or Apple**, but struggle to quantify how that news impacts short-term stock movement. Raw news text is unstructured, emotionally charged, and full of domain-specific jargon that general sentiment tools often misclassify.

## 2. Proposed Solution
**NewsPulse** is a dual-stage Machine Learning web application that translates raw financial news headlines into actionable market intelligence in real time:
1. **Stage 1 — Hybrid Financial Sentiment Model:** Combines a domain-augmented **VADER Sentiment Analyzer** (enhanced with financial triggers like *bullish, surge, dividend, regulatory, fine, lawsuit*) with a **TF-IDF + Logistic Regression** classifier trained on financial headlines to output a calibrated Sentiment Score from `-1.0` (Bearish) to `+1.0` (Bullish).
2. **Stage 2 — Short-Term Market Impact Predictor:** A secondary **Random Forest Classifier** that maps sentiment polarity, trigger intensity, and historical sentiment-vs-return correlations into an expected short-term directional movement (`↑ UPWARD`, `↓ DOWNWARD`, or `↔ NEUTRAL`).

---

## 3. Key Features
- **Interactive Search & Test Bar:** Evaluators can test pre-loaded benchmark financial headlines or paste custom live news articles.
- **Real-Time Sentiment Dial Gauge:** Interactive Plotly dial displaying exact polarity (`-1.0` to `+1.0`) alongside separate VADER and Logistic Regression sub-scores.
- **Market Reaction & Return Estimator:** Predicts short-term stock direction (`↑` or `↓`) and plots the headline against historical 1-day return correlations.
- **Explainable AI Trigger Word Cloud:** Automatically extracts and highlights key **Bullish (Green)** and **Bearish (Red)** financial trigger words driving the model's decision.

---

## 4. Technologies Used
- **Language:** Python 3.10+
- **Machine Learning & NLP:** `scikit-learn` (TF-IDF Vectorizer, Logistic Regression, Random Forest), `vaderSentiment` (Financial Lexicon Augmented)
- **Frontend & Visualization:** `Streamlit`, `Plotly` (Dial Gauge & Correlation Scatter), `WordCloud`, `Matplotlib`
- **Data Manipulation:** `Pandas`, `NumPy`

---

## 5. Instructions to Set Up and Run Locally

### Step 1: Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/NewsPulse-HackSphere.git
cd NewsPulse-HackSphere