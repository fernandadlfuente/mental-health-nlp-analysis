from transformers import pipeline
import pandas as pd

classifier = pipeline("sentiment-analysis")

df = pd.read_csv("data/cleaned_dataset.csv")

sample_texts = df['clean_text'].head(10)

for text in sample_texts:
  result = classifier(text[:512])
  print(result)
