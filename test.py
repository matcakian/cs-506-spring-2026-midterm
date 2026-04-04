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


df_features = df.columns.drop(["Id", "ProductId", "UserId"])

# print(1)

pipeline = joblib.load("model_pipeline.pkl")

# print(2)


result = raw_test_df.copy()
result["Score"] = pipeline.predict(df[df_features])

result.to_csv("result.csv", index=False)
