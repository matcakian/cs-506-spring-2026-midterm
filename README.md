# Review Score Prediction

The model uses TF-IDF features from review text and summaries combined with engineered numeric features (e.g., text statistics, and product/user aggregates). These features are processed through a sklearn pipeline and fed into a Ridge regression model to predict review scores.

## How to Run

1. Train the model:
python main.py

2. Generate predictions:
python test.py

- Running main.py creates: model_pipeline.pkl
- Running test.py creates: result.csv

## Files

- `main.py` — training pipeline  
- `test.py` — inference / submission script  
- `vocabulary.txt` — TF-IDF vocabulary

### Data (not included in repo)
- `train.csv` — training data  
- `test.csv` — test data  

## Output
result.csv       # final predictions (Id, Score)


## Approach

### Feature Engineering
Key features were derived from both text and metadata. TF-IDF representations were extracted from the review text and summary using a fixed vocabulary that were produced by running a separate logistic regression on all possible TF-IDF feautures and then selecting those with the highest coefficients across all classes. Additional numeric features were engineered, including text length, word count, sentence count, exclamation counts, and helpfulness ratios. Product- and user-level aggregate features were also included using leave-one-out averages to avoid leakage.

---

### Model Selection
A Ridge regression model was chosen due to its ability to provide a regularization term. Compared to unregularized linear regression, Ridge provided better generalization and reduced overfitting.

---

### Assumptions
- Review text and summaries contain meaningful signals for predicting scores.
- Product and user historical averages are predictive of future reviews.
- Missing values (e.g., unseen products/users) can be safely handled via imputation.
- The fixed vocabulary captures sufficiently important terms.

---

### Hyperparameter Tuning
The main parameter tuned was the Ridge regularization strength (`alpha`). TF-IDF parameters such as `max_features`, `min_df`, `max_df`, and `ngram_range` were also adjusted to balance model complexity and performance, but proved to be unne

---

### Validation Strategy
Model performance was evaluated using a train/test split and Root Mean Squared Error (RMSE). Care was taken to avoid data leakage by computing aggregate features using only training data. Improvements were validated based on both training and test performance to monitor overfitting.