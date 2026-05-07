import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

df = pd.read_csv("data/sample_dataset.csv")

def clean_text(text):
  text = text.lower()
  text = re.sub(r"http\S+", "", text)
  text = re.sub(r"[^a-zA-Z\s]", "", text)
  return text

df['clean_text'] = df['text'].apply(clean_text)
df.to_csv("data/cleaned_dataset.csv", index=False)

print("preprocessing is complete")

