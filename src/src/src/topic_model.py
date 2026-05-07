import pandas as pd
from bertopic import BERTopic

df = pd.read_csv("data/cleaned_dataset.csv")

docs = df['clean_text'].tolist()

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

print(topic_model.get_topic_info())
