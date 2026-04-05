import joblib
import pandas as pd


raw_df = pd.read_csv("train.csv")
raw_test_df = pd.read_csv("test.csv")


df = raw_test_df.merge(raw_df, on="Id", how="inner").dropna(axis=1)

df["TextLength"] = df["Text"].str.len()
df["WordCount"] = df["Text"].str.split().str.len()
df["SentenceCount"] = df["Text"].str.count(r"[.!?]")
df["ExclamationCount"] = df["Text"].str.count("!")
df["HelpfulnessRatio"] = df["HelpfulnessNumerator"] / (df["HelpfulnessDenominator"] + 1e-8)
df["SummaryLength"] = df["Summary"].str.len()
df["SummaryWordCount"] = df["Summary"].str.split().str.len()
df["SummarySentenceCount"] = df["Summary"].str.count(r"[.!?]")
df["SummaryExclamationCount"] = df["Summary"].str.count("!")

product_df = raw_df.groupby("ProductId").agg(ExclusiveProductAverage=("Score", "mean"),
	ProductReviewCount=("Score", "count"))

df = df.merge(product_df, on="ProductId", how="left")
df["ProductReviewCount"].fillna(0, inplace=True)


user_df = raw_df.groupby("UserId").agg(ExclusiveReviewAverage=("Score", "mean"),
	ReviewCount=("Score", "count"))

df = df.merge(user_df, on="UserId", how="left")
df["ReviewCount"].fillna(0, inplace=True)


df_features = df.columns.drop(["Id", "ProductId", "UserId"])

# print(1)

pipeline = joblib.load("model_pipeline.pkl")

# print(2)


result = raw_test_df.copy()
result["Score"] = pipeline.predict(df[df_features])

result.to_csv("result.csv", index=False)
