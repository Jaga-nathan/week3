
import csv
import time
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "distilbert-base-uncased"

LABELS = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ==========================================
# 2. LOAD DATASET
# ==========================================

texts = []
labels = []

with open("sentiment_dataset.csv", "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:
        texts.append(row["text"])
        labels.append(LABELS[row["sentiment"]])

print("\nDataset loaded successfully!")
print("Total samples:", len(texts))


# ==========================================
# 3. LOAD DISTILBERT TOKENIZER
# ==========================================

print("\n========== TOKENIZER OUTPUT ==========")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sentences = [
    "I love this product",
    "The customer service was very poor"
]

for sentence in sentences:

    encoded = tokenizer(sentence)

    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("Input IDs:", encoded["input_ids"])
    print("Attention Mask:", encoded["attention_mask"])


# ==========================================
# 4. SENTIMENT ANALYSIS PIPELINE
# ==========================================

print("\n========== SENTIMENT ANALYSIS PIPELINE ==========")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment_sentences = [
    "I really love this product",
    "The service was terrible and disappointing",
    "The package arrived yesterday"
]

for sentence in sentiment_sentences:

    result = sentiment_pipeline(sentence)

    print("\nSentence:", sentence)
    print("Result:", result)


# ==========================================
# 5. NAMED ENTITY RECOGNITION PIPELINE
# ==========================================

print("\n========== NER PIPELINE ==========")

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

ner_sentences = [
    "Elon Musk works at Tesla in the United States",
    "Microsoft is located in Redmond, Washington",
    "Sachin Tendulkar was born in Mumbai, India"
]

for sentence in ner_sentences:

    result = ner_pipeline(sentence)

    print("\nSentence:", sentence)
    print("Entities:", result)


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. CREATE PYTORCH DATASET
# ==========================================

class SentimentDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=64
        )

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


train_dataset = SentimentDataset(
    X_train, y_train, tokenizer
)

test_dataset = SentimentDataset(
    X_test, y_test, tokenizer
)

print("\nPyTorch datasets created successfully!")


# ==========================================
# 8. LOAD DISTILBERT MODEL
# ==========================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID_TO_LABEL,
    label2id=LABELS
)

print("\nDistilBERT model loaded successfully!")


# ==========================================
# 9. EVALUATION METRICS
# ==========================================

def compute_metrics(eval_pred):

    logits, true_labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(
        true_labels, predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1": macro_f1
    }


# ==========================================
# 10. TRAINING ARGUMENTS
# ==========================================

training_args = TrainingArguments(
    output_dir="./distilbert_results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    use_cpu=True
)


# ==========================================
# 11. TRAIN DISTILBERT USING TRAINER
# ==========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("\n========== DISTILBERT TRAINING ==========")

start_train = time.time()

trainer.train()

distilbert_training_time = time.time() - start_train

print("\nTraining completed!")
print(
    "DistilBERT training time:",
    round(distilbert_training_time, 2),
    "seconds"
)


# ==========================================
# 12. DISTILBERT EVALUATION
# ==========================================

print("\n========== DISTILBERT EVALUATION ==========")

eval_results = trainer.evaluate()

distilbert_accuracy = eval_results["eval_accuracy"]
distilbert_f1 = eval_results["eval_f1"]

print("Accuracy:", round(distilbert_accuracy, 4))
print("Macro F1:", round(distilbert_f1, 4))


# ==========================================
# 13. INFERENCE TIME
# ==========================================

print("\n========== INFERENCE TIME ==========")

start_inference = time.time()

predictions = trainer.predict(test_dataset)

distilbert_inference_time = (
    time.time() - start_inference
)

print(
    "DistilBERT inference time:",
    round(distilbert_inference_time, 4),
    "seconds"
)


# ==========================================
# 14. SAVE FINE-TUNED MODEL
# ==========================================

trainer.save_model("./finetuned_model")

tokenizer.save_pretrained("./finetuned_model")

print("\nFine-tuned model saved successfully!")


# ==========================================
# 15. TF-IDF BASELINE
# ==========================================

print("\n========== TF-IDF BASELINE ==========")

tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

start_train = time.time()

tfidf_model.fit(X_train, y_train)

tfidf_training_time = time.time() - start_train

start_inference = time.time()

tfidf_predictions = tfidf_model.predict(X_test)

tfidf_inference_time = time.time() - start_inference

tfidf_accuracy = accuracy_score(
    y_test, tfidf_predictions
)

tfidf_f1 = f1_score(
    y_test,
    tfidf_predictions,
    average="macro"
)

print("Accuracy:", round(tfidf_accuracy, 4))
print("Macro F1:", round(tfidf_f1, 4))


# ==========================================
# 16. MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

print(
    f"{'Metric':<25}"
    f"{'TF-IDF':<20}"
    f"{'DistilBERT':<20}"
)

print("-" * 65)

print(
    f"{'Accuracy':<25}"
    f"{tfidf_accuracy:<20.4f}"
    f"{distilbert_accuracy:<20.4f}"
)

print(
    f"{'Macro F1':<25}"
    f"{tfidf_f1:<20.4f}"
    f"{distilbert_f1:<20.4f}"
)

print(
    f"{'Training time (sec)':<25}"
    f"{tfidf_training_time:<20.2f}"
    f"{distilbert_training_time:<20.2f}"
)

print(
    f"{'Inference time (sec)':<25}"
    f"{tfidf_inference_time:<20.4f}"
    f"{distilbert_inference_time:<20.4f}"
)


# ==========================================
# 17. SAVE COMPARISON RESULTS
# ==========================================

with open(
    "comparison_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("TF-IDF vs DistilBERT Comparison\n")
    file.write("--------------------------------\n")

    file.write(
        f"TF-IDF Accuracy: {tfidf_accuracy:.4f}\n"
    )

    file.write(
        f"DistilBERT Accuracy: {distilbert_accuracy:.4f}\n"
    )

    file.write(
        f"TF-IDF Macro F1: {tfidf_f1:.4f}\n"
    )

    file.write(
        f"DistilBERT Macro F1: {distilbert_f1:.4f}\n"
    )

    file.write(
        f"TF-IDF Training Time: {tfidf_training_time:.2f} seconds\n"
    )

    file.write(
        f"DistilBERT Training Time: {distilbert_training_time:.2f} seconds\n"
    )

    file.write(
        f"TF-IDF Inference Time: {tfidf_inference_time:.4f} seconds\n"
    )

    file.write(
        f"DistilBERT Inference Time: {distilbert_inference_time:.4f} seconds\n"
    )


print("\nComparison saved successfully!")
print("\nTask 4 training completed successfully!")

