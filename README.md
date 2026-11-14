# IMDB Sentiment Analysis using PyTorch

A Deep Learning project for binary sentiment classification of movie reviews using **LSTM (Long Short-Term Memory)** networks and pre-trained **GloVe word embeddings**.

## Project Overview

This project aims to classify movie reviews as either Positive or Negative based on their textual content.

The model uses an LSTM architecture to capture sequential information and understand how word order contributes to sentiment.

## Dataset

- **Dataset:** IMDB Movie Reviews Dataset
- **Total Reviews:** 50,000
- **Classes:** Positive and Negative
- **Class Distribution:** 25,000 reviews per class

## Model Architecture

The model consists of:

1. **Embedding Layer:** Maps token IDs to dense word vectors using pre-trained GloVe embeddings.
2. **LSTM Layer:** Processes word sequences and captures sequential information.
3. **Fully Connected Layer:** Maps the final hidden representation to a sentiment score.

The model uses `BCEWithLogitsLoss` for binary classification.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- GloVe Word Embeddings

## Project Structure

```text
Sentiment-Analysis/
│
├── data/                  # Dataset (not included)
├── embeddings/            # GloVe vectors (not included)
├── models/                # Saved model weights (not included)
│
├── main.py                # Training and evaluation
├── requirements.txt       # Project dependencies
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/manoemam555/Sentiment-Analysis.git
cd Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Before running the project, place the required files in the following locations:

- IMDB dataset: `data/IMDB_Dataset.csv`
- GloVe vectors: `embeddings/glove_vectors.txt`

These files are not included in this repository.

## Training

Run the training script:

```bash
python main.py
```

The script handles:

- Data preprocessing
- Vocabulary construction
- GloVe embedding initialization
- Model training
- Validation and testing

The best model checkpoint is saved to `models/best_lstm.pth`.

## Future Improvements

- Create a separate prediction script.
- Add inference for custom movie reviews.
- Experiment with GRU and Bidirectional LSTM architectures.
- Improve model performance through hyperparameter tuning.

## Author

**Abdelrahman Emam**

GitHub: [@manoemam555](https://github.com/manoemam555)

Project Repository: [Sentiment-Analysis](https://github.com/manoemam555/Sentiment-Analysis)