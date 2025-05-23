# News Summarization Project

## Overview
This project implements and compares different text summarization techniques for news articles, specifically focusing on tennis-related news. It leverages both extractive (spaCy) and abstractive (BART) summarization methods, evaluates their performance using ROUGE metrics, and includes a basic Named Entity Recognition (NER) system.

## Features
- Automated text summarization using two different approaches:
  - Extractive summarization with spaCy
  - Abstractive summarization with BART (facebook/bart-large-cnn)
- Performance comparison using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- Basic Named Entity Recognition to identify persons, locations, and organizations
- Data preprocessing and cleaning
- Visualization of results

## Requirements
- Python 3.6+
- Google Colab (for running the notebook)
- Kaggle API credentials (to access the dataset)

## Dependencies
```
transformers
pandas
torch
spacy
nltk
rouge-score
matplotlib
kaggle
```

## Setup Instructions

### 1. Kaggle API Setup
1. Create a Kaggle account if you don't have one at [kaggle.com](https://www.kaggle.com/)
2. Generate your API token from Account -> API -> Create New API Token
3. Download the `kaggle.json` file
4. Upload this file when prompted in the notebook

### 2. Running the Notebook
1. Open the notebook in Google Colab
2. Run the cells in sequence
3. When prompted, upload your `kaggle.json` file

## Project Structure

### Data Acquisition
- Uses Kaggle API to download the "tennis-articles" dataset
- Unzips and loads the data into a pandas DataFrame

### Data Preprocessing
- Cleans and normalizes the text data
- Removes newlines and extra spaces
- Prepares the data for summarization

### Summarization Methods

#### 1. BART Summarization (Abstractive)
- Utilizes Hugging Face's transformers library
- Implements facebook/bart-large-cnn pre-trained model
- Generates concise abstractive summaries

#### 2. spaCy Summarization (Extractive)
- Uses frequency-based sentence scoring
- Extracts the most important sentences based on word importance
- Implements a customized extractive algorithm

### Evaluation
- Compares both summarization methods using ROUGE metrics:
  - ROUGE-1: Overlap of unigrams
  - ROUGE-2: Overlap of bigrams
  - ROUGE-L: Longest common subsequence
- Visualizes performance using bar charts

### Named Entity Recognition
- Implements a rule-based NER system using POS tagging
- Identifies and categorizes named entities:
  - PERSON: Athletes, coaches, etc.
  - LOCATION: Countries, cities, venues
  - ORGANIZATION: Tennis associations, tournaments, etc.
- Uses predefined dictionaries and NLTK's POS tagger

## Usage

### Generating Summaries
```python
# Using BART for abstractive summarization
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(article_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

# Using spaCy for extractive summarization
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def text_summarizer(raw_docx):
    # Implementation details in the notebook
    return summary
```

### Evaluating Summaries
```python
from rouge_score import rouge_scorer

def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure
```

### Named Entity Recognition
```python
def extract_named_entities(text):
    tokens = preprocess_text(text)
    entities = identify_entities(tokens)
    categorized_entities = {}
    # Further processing as shown in the notebook
    return categorized_entities
```

## Output Files
- `summarized_articles.csv`: Contains original articles and BART summaries
- `tennis_articles_spacy_model_summarized.csv`: Contains original articles and spaCy summaries
- `summarized_articles_with_entities.csv`: Contains summaries with extracted named entities

## Results and Findings
- BART (abstractive) generally produces more cohesive and readable summaries
- spaCy (extractive) preserves more exact information from the original text
- ROUGE scores comparison provides quantitative evaluation
- NER system successfully identifies key entities in tennis articles

![image](https://github.com/user-attachments/assets/b2fb3c46-f669-4680-8159-18e7152f415c)

Based on the graph, the Spacy model (extractive summarization) significantly outperforms the BART model (abstractive summarization) across all ROUGE metrics. Spacy shows particularly strong performance in ROUGE-1 and ROUGE-2 scores, suggesting it better preserves the original text's words and phrases. This is expected since extractive summarization directly selects sentences from the source text, while abstractive summarization generates new text that may use different wording than the original.RetryClaude can make mistakes. Please double-check responses.


## Limitations
- Dataset limited to tennis articles
- Basic NER system relies on predefined dictionaries
- Processing very long articles may require text truncation

## Future Improvements
- Implement more advanced NER using transformer-based models
- Explore fine-tuning BART on sports-specific content
- Add support for multi-language summarization
- Implement a more sophisticated evaluation framework
- Create a web interface for real-time article summarization

## Acknowledgments
- Tennis articles dataset from Kaggle
- Hugging Face Transformers library
- spaCy NLP library
- NLTK for basic NLP tasks

# Contributing to News Summarization Project

## Clone the Repository
```bash
git clone https://github.com/yourusername/news-summarization-project.git
cd news-summarization-project
```

## Fork the Repository
1. Click "Fork" on the GitHub repository page
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/news-summarization-project.git
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/yourusername/news-summarization-project.git
   ```
4. Keep your fork updated:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

## Submit a Pull Request
1. Create a branch:
   ```bash
   git checkout -b feature/your-feature
   ```
2. Make and commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```
4. Go to the original repository and click "New Pull Request"
5. Select "compare across forks" and select your branch
6. Add a clear title and description
7. Submit the pull request

## PR Requirements
- Follow code style (PEP 8)
- Include tests for new features
- Update documentation as needed
- Use clear commit messages

For questions, open an issue or contact: jai.shree.dam@gmail.com

## License
This project is available under the MIT License.
