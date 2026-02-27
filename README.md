# 🏠 Real Estate Machine+ Valuation

An integrated AI-powered real estate valuation system for King County, Washington. This project bridges the gap between traditional Machine Learning (ML) and Generative AI (LLM) to provide a comprehensive property analysis.

## 🚀 Key Features
- Price Prediction: Accurately estimates property value using a trained Linear Regression model.
- Smart Classification: Categorizes properties into (Budget, Mid-Range, or Luxury) using a K-Means clustering algorithm.
- AI-Driven Insights: Generates real-time investment advice and market analysis using Llama 3.1 via the Groq API.
- Interactive Dashboard: A professional, user-friendly interface built with Streamlit for seamless data input and visualization.

## 🛠️ Tech Stack
- Python: Core programming language.
- Scikit-learn: Used for model development and preprocessing.
- Pandas & Numpy: Data manipulation and numerical computation.
- Groq API: Integration of Large Language Models (LLM) for natural language reasoning.
- Streamlit: Framework for the web application interface.

## 📋 Setup & Installation
1. Ensure all model files (`.pkl`) are in the root directory:
   - house_price_model.pkl
   - house_category_model.pkl
   - scaler.pkl
   - encoder.pkl
   - kmeans_model.pkl

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt