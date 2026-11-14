
import re
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle 
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.model_selection import train_test_split


# ==========================================
# 1. Configuration
# ==========================================

SEED = 42

BATCH_SIZE = 64
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 1

LEARNING_RATE = 0.001
EPOCHS = 10
MIN_FREQ = 2

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
UNK_IDX = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_PATH = "data/IMDB_Dataset.csv"
GLOVE_PATH = "embeddings/glove_vectors.txt"
MODEL_PATH = "models/best_lstm.pth"


# ==========================================
# 2. Reproducibility
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# 3. Load Dataset
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

X = df["review"].to_numpy()
y = df["sentiment"].to_numpy()


# ==========================================
# 4. Split Dataset
# ==========================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.125,
    random_state=SEED,
    stratify=y_train_val
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ==========================================
# 5. Text Cleaning
# ==========================================

def clean_text(text):

    text = re.sub(r"<.*?>", " ", text)

    text = text.lower()

    text = re.sub(r"[^a-z0-9']", " ", text)

    tokens = text.split()

    return tokens


# ==========================================
# 6. Build Vocabulary
# ==========================================

def build_vocabulary(texts, min_freq=2):

    counter = Counter()

    for text in texts:

        tokens = clean_text(text)

        counter.update(tokens)

    vocab = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX
    }

    # Deterministic vocabulary ordering
    valid_words = sorted([
        word
        for word, count in counter.items()
        if count >= min_freq
    ])

    for word in valid_words:

        vocab[word] = len(vocab)

    return vocab


vocab = build_vocabulary(
    X_train,
    min_freq=MIN_FREQ
)

print("Vocabulary size:", len(vocab))



# ==========================================
# 7. Encode Reviews
# ==========================================

def encode_review(text, vocab):

    tokens = clean_text(text)

    encoded = [
        vocab.get(token, UNK_IDX)
        for token in tokens
    ]

    if len(encoded) == 0:
        encoded = [UNK_IDX]

    return encoded


X_train_encoded = [
    encode_review(text, vocab)
    for text in X_train
]

X_val_encoded = [
    encode_review(text, vocab)
    for text in X_val
]

X_test_encoded = [
    encode_review(text, vocab)
    for text in X_test
]


# ==========================================
# 8. Load GloVe Embeddings
# ==========================================

def load_glove_embeddings(
    glove_path,
    vocab,
    embedding_dim
):

    embedding_matrix = np.random.normal(
        0,
        0.01,
        (len(vocab), embedding_dim)
    ).astype(np.float32)

    embedding_matrix[PAD_IDX] = np.zeros(
        embedding_dim,
        dtype=np.float32
    )

    found_words = 0

    with open(glove_path, "r", encoding="utf-8") as file:

        for line in file:

            values = line.rstrip().split()

            if len(values) != embedding_dim + 1:
                continue

            word = values[0]

            if word in vocab:

                vector = np.asarray(
                    values[1:],
                    dtype=np.float32
                )

                embedding_matrix[vocab[word]] = vector

                found_words += 1

    print("GloVe words found:", found_words)

    return embedding_matrix


embedding_matrix = load_glove_embeddings(
    GLOVE_PATH,
    vocab,
    EMBEDDING_DIM
)


# ==========================================
# 9. Dataset Class
# ==========================================

class SentimentDataset(Dataset):

    def __init__(self, reviews, labels):

        self.reviews = reviews
        self.labels = labels

    def __len__(self):

        return len(self.reviews)

    def __getitem__(self, index):

        review = torch.tensor(
            self.reviews[index],
            dtype=torch.long
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return review, label


# ==========================================
# 10. Collate Function
# ==========================================

def collate_fn(batch):

    reviews, labels = zip(*batch)

    # Save original sequence lengths before padding
    lengths = torch.tensor(
        [len(review) for review in reviews],
        dtype=torch.long
    )

    reviews_padded = pad_sequence(
        reviews,
        batch_first=True,
        padding_value=PAD_IDX
    )

    labels = torch.stack(labels)

    return reviews_padded, labels, lengths


# ==========================================
# 11. Create DataLoaders
# ==========================================

train_dataset = SentimentDataset(
    X_train_encoded,
    y_train
)

val_dataset = SentimentDataset(
    X_val_encoded,
    y_val
)

test_dataset = SentimentDataset(
    X_test_encoded,
    y_test
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ==========================================
# 12. LSTM Model
# ==========================================

class SentimentLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        embedding_matrix
    ):

        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=False,
            padding_idx=PAD_IDX
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (h_n, c_n) = self.lstm(
            packed
        )

        last_hidden = h_n[-1]

        logits = self.fc(last_hidden)

        return logits.squeeze(1)


# ==========================================
# 13. Initialize Model
# ==========================================

model = SentimentLSTM(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    embedding_matrix=embedding_matrix
).to(DEVICE)


# ==========================================
# 14. Loss and Optimizer
# ==========================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 15. Training Function
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for reviews, labels, lengths in loader:

        reviews = reviews.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(reviews, lengths)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = (
            logits >= 0
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / len(loader)

    accuracy = correct / total

    return avg_loss, accuracy


# ==========================================
# 16. Evaluation Function
# ==========================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for reviews, labels, lengths in loader:

            reviews = reviews.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(reviews, lengths)

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = (
                logits >= 0
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / len(loader)

    accuracy = correct / total

    return avg_loss, accuracy


# ==========================================
# 17. Training Loop
# ==========================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

best_val_loss = float("inf")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    # Save the best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best model saved!")


# ==========================================
# 18. Load Best Model
# ==========================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)


# ==========================================
# 19. Final Test Evaluation
# ==========================================

test_loss, test_acc = evaluate(
    model,
    test_loader,
    criterion
)

print("\nFinal Test Results")
print("------------------")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
