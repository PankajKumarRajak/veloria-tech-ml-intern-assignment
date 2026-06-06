# 🏏 IPL Cricket Match Analysis & RAG Search System

## Project Overview

This project was developed as part of the Veloria Tech ML Internship Assignment.

The project consists of three components:

1. Data Scraping
2. Machine Learning Model
3. RAG (Retrieval Augmented Generation) Search System

---

## Features

### Data Scraping

- Scrapes IPL match data from Howstat.
- Extracts:
  - Match Date
  - Team 1
  - Team 2
  - Venue
  - Match Result
- Stores data in CSV format.

### Machine Learning

- Uses Random Forest Classifier.
- Encodes categorical features using LabelEncoder.
- Predicts match winners.
- Evaluates model using:
  - Accuracy Score
  - F1 Score
  - Confusion Matrix

### RAG Search System

- Converts match records into text documents.
- Generates embeddings using Sentence Transformers.
- Stores embeddings in ChromaDB.
- Performs semantic search on IPL match data.
- Supports natural language queries.

---

## Technologies Used

- Python
- Pandas
- BeautifulSoup
- Scikit-Learn
- Sentence Transformers
- ChromaDB
- Hugging Face Transformers

---

## Project Structure

```text
veloria-tech-ml-intern-assignment/
│
├── scraper.py
├── model.py
├── rag.py
├── match_data.csv
├── cricket_documents.txt
├── chroma_db/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Scrape Data

```bash
python scraper.py
```

### 2. Train ML Model

```bash
python model.py
```

### 3. Run RAG Search

```bash
python rag.py
```

---

## Example Query

```text
Which matches did Royal Challengers Bengaluru win?
```

```text
Who won matches at Wankhede Stadium?
```

---

## Future Improvements

- Integrate Gemini API
- Add LLM-generated answers
- Deploy using Streamlit
- Add Vector Database Optimization

---

## Author

Pankaj
