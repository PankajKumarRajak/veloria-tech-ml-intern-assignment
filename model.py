import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load CSV
df = pd.read_csv("match_data.csv")

# Create Winner column from Result column
df["Winner"] = df["Result"].apply(
    lambda x: "Royal Challengers Bengaluru"
    if "Royal Challengers" in str(x)
    else "Chennai Super Kings"
)

# Encode categorical columns
le_team1 = LabelEncoder()
le_team2 = LabelEncoder()
le_venue = LabelEncoder()
le_winner = LabelEncoder()

df["Team1"] = le_team1.fit_transform(df["Team1"])
df["Team2"] = le_team2.fit_transform(df["Team2"])
df["Venue"] = le_venue.fit_transform(df["Venue"])
df["Winner"] = le_winner.fit_transform(df["Winner"])

# Features and Target
X = df[["Team1", "Team2", "Venue"]]
y = df["Winner"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions, average="weighted"))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))