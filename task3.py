
import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from gensim.downloader import load
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("sentiment_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df["sentiment"].value_counts())


# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    return tokens


df["tokens"] = df["text"].apply(clean_text)

# Join tokens back into cleaned sentences
df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

print("\nText preprocessing completed!")
print("Original text:", df["text"].iloc[0])
print("Cleaned tokens:", df["tokens"].iloc[0])


# ==========================================
# 3. TRAIN AND TEST SPLIT
# ==========================================

X = df["clean_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. TF-IDF + LOGISTIC REGRESSION
# ==========================================

tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

tfidf_model.fit(X_train, y_train)

tfidf_predictions = tfidf_model.predict(X_test)

tfidf_accuracy = accuracy_score(y_test, tfidf_predictions)

tfidf_f1 = f1_score(
    y_test,
    tfidf_predictions,
    average="macro"
)

print("\n========== TF-IDF BASELINE ==========")
print("Accuracy:", round(tfidf_accuracy, 4))
print("Macro F1:", round(tfidf_f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    tfidf_predictions,
    labels=["Negative", "Neutral", "Positive"]
))

print("\nClassification Report:")
print(classification_report(y_test, tfidf_predictions))


# ==========================================
# 5. TOP 10 IMPORTANT WORDS FOR EACH CLASS
# ==========================================

vectorizer = tfidf_model.named_steps["tfidf"]
classifier = tfidf_model.named_steps["classifier"]

feature_names = vectorizer.get_feature_names_out()

print("\n========== TOP 10 IMPORTANT WORDS ==========")

for index, class_name in enumerate(classifier.classes_):

    top_indices = np.argsort(
        classifier.coef_[index]
    )[-10:][::-1]

    top_words = feature_names[top_indices]

    print("\n", class_name, ":")
    print(top_words)


# ==========================================
# 6. LOAD GLOVE WORD EMBEDDINGS
# ==========================================

print("\n========== LOADING GLOVE EMBEDDINGS ==========")

print("Downloading/loading GloVe vectors...")
glove = load("glove-wiki-gigaword-50")

print("GloVe embeddings loaded successfully!")
print("Embedding size:", glove.vector_size)


# ==========================================
# 7. FIND 5 NEAREST WORDS
# ==========================================

chosen_words = ["king", "computer"]

print("\n========== NEAREST WORDS ==========")

for word in chosen_words:

    print("\nNearest words to", word, ":")

    similar_words = glove.most_similar(word, topn=5)

    for similar_word, similarity in similar_words:
        print(similar_word, ":", round(similarity, 4))


# ==========================================
# 8. WORD ANALOGY
# king - man + woman = queen
# ==========================================

print("\n========== WORD ANALOGY ==========")

result = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)

print("king - man + woman ->", result[0][0])


# ==========================================
# 9. CREATE SENTENCE VECTORS
# ==========================================

def sentence_vector(sentence):

    words = sentence.split()

    word_vectors = [
        glove[word]
        for word in words
        if word in glove
    ]

    if len(word_vectors) == 0:
        return np.zeros(glove.vector_size)

    return np.mean(word_vectors, axis=0)


X_train_vectors = np.array([
    sentence_vector(sentence)
    for sentence in X_train
])

X_test_vectors = np.array([
    sentence_vector(sentence)
    for sentence in X_test
])

print("\nSentence vectors created successfully!")
print("Training vector shape:", X_train_vectors.shape)
print("Testing vector shape:", X_test_vectors.shape)


# ==========================================
# 10. LOGISTIC REGRESSION ON GLOVE VECTORS
# ==========================================

glove_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

glove_model.fit(X_train_vectors, y_train)

glove_predictions = glove_model.predict(X_test_vectors)

glove_accuracy = accuracy_score(y_test, glove_predictions)

glove_f1 = f1_score(
    y_test,
    glove_predictions,
    average="macro"
)

print("\n========== GLOVE SENTENCE EMBEDDINGS ==========")

print("Accuracy:", round(glove_accuracy, 4))
print("Macro F1:", round(glove_f1, 4))


# ==========================================
# 11. COMPARE BOTH MODELS
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print(
    "TF-IDF Accuracy:",
    round(tfidf_accuracy, 4)
)

print(
    "GloVe Sentence Vector Accuracy:",
    round(glove_accuracy, 4)
)

print(
    "TF-IDF Macro F1:",
    round(tfidf_f1, 4)
)

print(
    "GloVe Macro F1:",
    round(glove_f1, 4)
)

print("\nTask 3 completed successfully!")
