## Policy Summarization & Scenario Generator
AI-powered system that summarises real-world policy documents and generates 
scenario-based policy drafts using Generative AI.

## Project Overview
This project presents an AI-assisted web application built to analyse and adapt 
real-world policy documents for different scenarios and audiences.

The analysis focuses on:
* Policy summarisation using NLP techniques
* Scenario-based policy draft generation
* Interactive web interface for policy exploration

## Dataset Source
* Source: Sri Lanka Government Policy Document
* Type: Real-world published policy
* Format: PDF / Text

## Technologies Used
* Python — Core programming language
* Hugging Face — AI model for text generation (Llama 3.1)
* Streamlit — Web application framework
* NLP (NLTK / spaCy) — Text preprocessing and summarisation
* GitHub — Version control and project documentation

## Project Structure
policy-summarization-app/
|-- summarizer.py
|-- scenario_generator.py
|-- streamlit_app.py
|-- README.md

## How to Run

### Step 1 – Get a Hugging Face API Key
1. Go to **huggingface.co** and create a free account
2. Click your profile → **Settings** → **Access Tokens**
3. Click **New Token** and copy it

### Step 2 – Set your API Key
On **Windows:**
set HF_TOKEN=your-token-here
On **Mac/Linux:**
export HF_TOKEN=your-token-here

### Step 3 – Install dependencies
pip install -r requirements.txt

### Step 4 – Run the app
streamlit run streamlit_app.py

## Key Features
* NLP-based summarisation of complex policy documents
* Minimum 2 scenario-based adapted policy drafts
* Interactive left/right panel web interface
* Supports multiple scenarios from the same summary


