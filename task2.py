import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# STEP 1: SET RANDOM SEED
# ==========================================

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# STEP 2: LOAD BREAST CANCER DATASET
# ==========================================

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

df.to_csv("breast_cancer.csv", index=False)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# ==========================================
# STEP 3: SPLIT FEATURES AND TARGET
# ==========================================

X = df.drop("target", axis=1).values
y = df["target"].values

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# ==========================================
# STEP 4: FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ==========================================
# STEP 5: CREATE CSV DATASET CLASS
# ==========================================

class CsvDataset(Dataset):

    def __init__(self, features, labels):

        self.features = torch.tensor(
            features, dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels, dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):

        return self.features[index], self.labels[index]


train_dataset = CsvDataset(X_train, y_train)
val_dataset = CsvDataset(X_val, y_val)

# ==========================================
# STEP 6: CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print("DataLoaders created successfully!")

# ==========================================
# STEP 7: BUILD MLP NEURAL NETWORK
# ==========================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(30, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        return self.network(x)


model = MLP()

# Count model parameters

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Params:", total_params)

# ==========================================
# STEP 8: LOSS FUNCTION AND OPTIMIZER
# ==========================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# STEP 9: VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            outputs = model(features).squeeze(1)

            predictions = (outputs >= 0).float()

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy


# ==========================================
# STEP 10: TRAINING LOOP
# ==========================================

epochs = 40
patience = 7

best_accuracy = 0
best_epoch = 0
counter = 0

best_model_state = copy.deepcopy(model.state_dict())

train_losses = []

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, labels in train_loader:

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward propagation
        outputs = model(features).squeeze(1)

        # 3. Calculate loss
        loss = criterion(outputs, labels)

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    # Validation accuracy
    val_accuracy = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Training Loss: {epoch_loss:.4f} "
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    # Early stopping

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        counter = 0

    else:

        counter += 1

    if counter >= patience:

        print("Early stopping triggered!")

        break


# ==========================================
# STEP 11: LOAD BEST MODEL
# ==========================================

model.load_state_dict(best_model_state)

print("\nBest val accuracy:", round(best_accuracy, 4))
print("Best epoch:", best_epoch)

# ==========================================
# STEP 12: SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "breast_cancer_model.pth"
)

print("Model saved successfully!")

# ==========================================
# STEP 13: LOAD MODEL INTO NEW MODEL
# ==========================================

new_model = MLP()

new_model.load_state_dict(
    torch.load(
        "breast_cancer_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

new_model.eval()

# ==========================================
# STEP 14: CHECK RELOADED ACCURACY
# ==========================================

reloaded_accuracy = evaluate(
    new_model,
    val_loader
)

print(
    "Reloaded model accuracy:",
    round(reloaded_accuracy, 4)
)

print(
    "Accuracy identical:",
    best_accuracy == reloaded_accuracy
)

# ==========================================
# STEP 15: LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_accuracy = accuracy_score(
    y_val,
    logistic_predictions
)

print("\n========== MODEL COMPARISON ==========")

print(
    "PyTorch MLP Accuracy:",
    round(best_accuracy, 4)
)

print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy, 4)
)

print("Training pipeline completed successfully!")
