# Review Score Prediction

The model uses TF-IDF features from review text and summaries combined with engineered numeric features (e.g., text statistics, sentiment scores, and product/user aggregates). These features are processed through a sklearn pipeline and fed into a Ridge regression model to predict review scores.

## How to Run

1. Train the model:
python main.py

2. Generate predictions:
python test.py

- Running main.py creates: model_pipeline.pkl
- Running test.py creates: result.csv

## Files
main.py          # training
test.py          # inference
vocabulary.txt   # TF-IDF vocab
train.csv        # training data
test.csv         # test data

## Output
result.csv       # final predictions (Id, Score)