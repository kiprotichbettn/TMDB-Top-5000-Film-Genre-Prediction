cat > README.md <<EOF
# TMDB Top 5000 Film Genre Prediction

Used Python and Scikit-learn to implement classification models to predict the primary genre (`genres_1`) of movies from the TMDB dataset. The workflow includes data cleaning, preprocessing, feature scaling, and model evaluation using both **K-Nearest Neighbors (KNN)** and **Logistic Regression**.


## Run
https://colab.research.google.com/github/kiprotichbettn/TMDB-Top-5000-Film-Genre-Prediction/blob/main/tmdb_genre_classification.ipynb



## Workflow

1. **Data Cleaning & Preprocessing**  
   - Fill missing values for numeric features (`budget`, `revenue`, etc.) with `0`.  
   - Fill missing values for categorical features (`keywords`, `production_companies`) with `'Unknown'`.  
   - Remove classes with less than 200 samples to reduce imbalance.  

2. **Feature Encoding & Scaling**  
   - Encode categorical columns using `LabelEncoder`.  
   - Scale numeric features to range [0, 1] using `MinMaxScaler`.  

3. **Data Splitting**  
   - 70% Train, 15% Development, 15% Test.  

4. **KNN Classification**  
   - Distance metric: Manhattan distance  
   - Optimal `k` chosen via development set accuracy.  

5. **Logistic Regression**  
   - Multi-class classification with `multinomial` option.  
   - Evaluated with accuracy, weighted recall, and weighted F1-score.

---

## Results

- Development Set Accuracy (KNN): *varies based on `k`*  
- Test Set Accuracy (KNN): *reported*  
- MAE & MSE (KNN): *reported*  
- Logistic Regression Accuracy: *reported*  
- Weighted Avg Recall & F1 Score (Logistic Regression): *reported*

---

## Requirements

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn

