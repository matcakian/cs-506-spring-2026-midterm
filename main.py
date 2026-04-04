import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


raw_df = pd.read_csv("train.csv")
df = raw_df.dropna().copy()

df["TextLength"] = df["Text"].str.len()
df["WordCount"] = df["Text"].str.split().str.len()
df["SentenceCount"] = df["Text"].str.count(r"[.!?]")
df["ExclamationCount"] = df["Text"].str.count("!")
df["HelpfulnessRatio"] = df["HelpfulnessNumerator"] / (df["HelpfulnessDenominator"] + 1e-8)
df["SummaryLength"] = df["Summary"].str.len()
df["SummaryWordCount"] = df["Summary"].str.split().str.len()
df["SummarySentenceCount"] = df["Summary"].str.count(r"[.!?]")
df["SummaryExclamationCount"] = df["Summary"].str.count("!")


# print(df.info())

df_features = df.columns.drop(["Id", "ProductId", "UserId", "Score"])
numeric_features = df_features.drop(["Summary", "Text"])

X_train, X_test, y_train, y_test = train_test_split(df[df_features], df["Score"], train_size=0.8, random_state=73)


preprocess = ColumnTransformer(
	transformers=[
		("tfidf", TfidfVectorizer(max_features=10000,
			ngram_range=(1, 2), min_df=5, max_df=.8, stop_words="english"), "Text"),
		("numeric", StandardScaler(with_mean=False), numeric_features)
	]
)


pipeline = Pipeline([
	("preprocess", preprocess),
	("regression", Ridge(alpha=7))
], verbose=True)


pipeline.fit(X_train, y_train)

# print(np.sqrt(mean_squared_error(pipeline.predict(X_test), y_test)))
# print(np.sqrt(mean_squared_error(pipeline.predict(X_train), y_train)))

joblib.dump(pipeline, "model_pipeline.pkl")

