import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/cleaned_dataset.csv")

X = df['clean_text]
y = df['label']

vectorizer = TfidVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)
X_vec = vectorizer.fit_transform(X)

X_train, X_text, y_train, y_test = train_test_split(X,vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X.test)

print(classification_report(y_test, predictions))
