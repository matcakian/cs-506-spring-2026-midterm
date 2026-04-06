import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.impute import SimpleImputer



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

texts = df["Text"].astype(str).tolist()
summaries = df["Summary"].astype(str).tolist()

text_sentiment = [analyzer.polarity_scores(t)["compound"] for t in tqdm(texts)]
summary_sentiment = [analyzer.polarity_scores(t)["compound"] for t in tqdm(summaries)]

df["ReviewSentiment"] = text_sentiment
df["SummarySentiment"] = summary_sentiment


X_train, X_test, y_train, y_test = train_test_split(df, df["Score"], train_size=0.8, random_state=58)


def exclusiveStats(df, old_avg_name, old_cnt_name, new_avg_name, target_name):
	df = df.copy()
	tmp = df[old_cnt_name] > 1

	df.loc[~tmp, new_avg_name] = pd.NA
	df.loc[tmp, new_avg_name] = (df.loc[tmp, old_avg_name] * df.loc[tmp, old_cnt_name] - 
		df.loc[tmp, target_name]) / (df.loc[tmp, old_cnt_name] - 1)

	df = df.drop(old_avg_name, axis=1)
	df[old_cnt_name] = df[old_cnt_name] - 1

	return df


product_df = X_train.groupby("ProductId").agg(AverageProductScore=("Score", "mean"),
	ProductReviewCount=("Score", "count"))

X_train = exclusiveStats(X_train.merge(product_df, on="ProductId", how="left"),
	"AverageProductScore", "ProductReviewCount", "ExclusiveProductAverage", "Score")

X_test = X_test.merge(product_df, on="ProductId", how="left").rename(
	columns={"AverageProductScore": "ExclusiveProductAverage"})

X_test["ProductReviewCount"].fillna(0, inplace=True)


user_df = X_train.groupby("UserId").agg(AverageReviewScore=("Score", "mean"),
	ReviewCount=("Score", "count"))

X_train = exclusiveStats(X_train.merge(user_df, on="UserId", how="left"),
	"AverageReviewScore", "ReviewCount", "ExclusiveReviewAverage", "Score")

X_test = X_test.merge(user_df, on="UserId", how="left").rename(
	columns={"AverageReviewScore": "ExclusiveReviewAverage"})

X_test["ReviewCount"].fillna(0, inplace=True)



df_features = X_train.columns.drop(["Id", "ProductId", "UserId", "Score"])
numeric_features = df_features.drop(["Summary", "Text"])

with open("vocabulary.txt") as f:
    vocabulary = [line.strip() for line in f]


preprocess = ColumnTransformer(
	transformers=[
		("tfidf_text", TfidfVectorizer(vocabulary=vocabulary), "Text"),
		("tfidf_summary", TfidfVectorizer(vocabulary=vocabulary), "Summary"),
		("numeric", Pipeline([
			("impute", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)),
			("scale", StandardScaler(with_mean=False))], verbose=True), numeric_features)
	], verbose=True)


pipeline = Pipeline([
	("preprocess", preprocess),
	("regression", Ridge(alpha=0))
], verbose=True)



pipeline.fit(X_train, y_train)

# print(np.sqrt(mean_squared_error(pipeline.predict(X_test), y_test)))
# print(np.sqrt(mean_squared_error(pipeline.predict(X_train), y_train)))

joblib.dump(pipeline, "model_pipeline.pkl")

