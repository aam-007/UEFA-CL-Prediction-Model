# UEFA Champions League Prediction Model

## Project Overview

This project implements a machine learning model to predict UEFA Champions League match outcomes using historical data from 2004-2021. The model uses a Categorical Naive Bayes classifier to predict match winners and simulate elimination tournament outcomes.

---

## Dataset

**Source**: UEFA Champions League matches from 2004-2021  
**Location**: `data/raw/UEFA Champions League 2004-2021.csv`

### Dataset Features:
- **homeTeam**: Home team name
- **awayteam**: Away team name
- **homeScore**: Goals scored by home team
- **awayscore**: Goals scored by away team
- Match metadata (date, round, etc.)

---

## Data Preprocessing

### 1. Score Cleaning
```python
def clean_score(x):
    if pd.isna(x): return 0
    m = re.search(r'(\d+)', str(x))
    return int(m.group(1)) if m else 0
```
- Handles missing values
- Extracts numeric scores using regex
- Converts to integer format

### 2. Winner Determination
```python
df['Winner'] = df.apply(lambda r: 
    'Home' if r['homeScore'] > r['awayscore'] 
    else ('Away' if r['homeScore'] < r['awayscore'] 
    else 'Draw'), axis=1)
```

### 3. Label Encoding
- All unique teams extracted from both home and away columns
- **LabelEncoder** used to convert team names to numeric identifiers
- Encoded columns: `h_enc` (home team), `a_enc` (away team)

### 4. Binary Classification Setup
- **Draws excluded** from training data
- **Target variable**: Binary (0 = Home Win, 1 = Away Win)
- **Train/Test Split**: 80/20 with stratification

---

## Model Architecture

### Categorical Naive Bayes Classifier

**Algorithm**: `CategoricalNB`  
**Hyperparameters**:
- `alpha=0.1` (Laplace smoothing parameter)

**Why Naive Bayes?**
- Effective for categorical features
- Probabilistic predictions
- Fast training and inference
- Works well with limited data

### Feature Engineering
- **Features**: Encoded home team (`h_enc`) and away team (`a_enc`)
- **Target**: Binary outcome (Home Win vs Away Win)

---

## Key Features

### 1. Match Outcome Prediction
```python
def predict_match(home, away):
    h, a = le.transform([home, away])
    pred = model.predict([[h, a]])[0]
    return away if pred == 1 else home
```
- Takes team names as input
- Returns predicted winner

### 2. Knockout Tournament Simulation
```python
def simulate_knockout(teams):
    # Randomized bracket seeding
    # Iterative elimination rounds
    # Returns tournament champion
```
- Simulates UEFA Champions League knockout format
- Uses model predictions for each match
- Randomly shuffles initial bracket

---

## Visualizations

The model generates three key visualizations saved to `graphs/`:

### 1. **Top 5 Clubs by Victories** (`top5_clubs.png`)
- Bar chart showing clubs with most wins
- Color palette: "rocket_r"
- Annotations display exact win counts
- **Key Insight**: Barcelona (Barca) shows highest historical win rate

### 2. **Prediction Matrix** (`confusion_matrix.png`)
- Heatmap of model predictions vs actual outcomes
- Color scheme: YlOrRd (Yellow-Orange-Red)
- Shows model accuracy breakdown
- Axes: Actual vs Predicted (Home/Away)

### 3. **Match Outcome Distribution** (`outcomes.png`)
- Pie chart showing overall match statistics
- Categories: Home Wins, Away Wins, Draws
- **Key Finding**: Draws have the highest percentage
- Color coding:
  - Home Wins: Green (#2ecc71)
  - Away Wins: Red (#e74c3c)
  - Draws: Yellow (#f1c40f)

---

## Model Performance

### Metrics Tracked:
1. **Accuracy Score**: Overall prediction accuracy
2. **Classification Report**: Precision, Recall, F1-Score for both classes
3. **Confusion Matrix**: Visual representation of prediction performance

### Expected Output:
```
Accuracy: 73.5%
              precision    recall  f1-score   support

    Home Win       0.75      0.85      0.80       163
    Away Win       0.70      0.56      0.62       105

    accuracy                           0.74       268
   macro avg       0.73      0.70      0.71       268
weighted avg       0.73      0.74      0.73       268

```

---

## Tournament Simulation

### Methodology:
1. **Input**: Top 8 clubs (sampled from top performers)
2. **Bracket Generation**: Random seeding with replacement
3. **Match Predictions**: Model predicts winner for each match
4. **Elimination Rounds**: 
   - Round of 8 → Quarterfinals
   - Quarterfinals → Semifinals
   - Semifinals → Final
5. **Output**: Predicted tournament champion

### Example Output:
```
Simulated Champion: Barcelona
```

---


### Expected Outputs:
- Console: Accuracy metrics and classification report
- Files: Three PNG visualizations in `graphs/` directory
- Simulation: Predicted tournament champion

---

## Dependencies

```python
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
numpy>=1.21.0
```

### Installation:
```bash
pip install pandas matplotlib seaborn scikit-learn numpy
```

---

## Results & Insights

### Historical Patterns:

1. **Draw Probability**
   - Draws represent the **highest outcome percentage** in UCL matches
   - Reflects defensive, high-stakes nature of Champions League football
   - Model trained on non-draw matches for clearer predictions

2. **Barcelona Dominance**
   - Barcelona shows **highest win count** among top 5 clubs
   - Strong historical performance in UCL (2004-2021 period)
   - Likely tournament simulation winner based on historical success

3. **Home Advantage**
   - Analysis of home vs away win distribution
   - Visualized in outcome pie chart
   - Informs model's understanding of match dynamics

### Model Limitations:

- **No draws predicted**: Binary classification excludes draws
- **Historical bias**: Relies on 2004-2021 data
- **Feature simplicity**: Only uses team identity, not form/injuries/tactics
- **Assumes independence**: Naive Bayes assumes feature independence

### Future Improvements:

- Add temporal features (recent form, season)
- Include player statistics and team strength metrics
- Implement ensemble methods for better accuracy
- Add draw prediction capability (multi-class classification)
- Real-time data integration for current season predictions

---

## Project Structure

```
UEFA-CL-Prediction-Model/
│
├── data/
│   └── raw/
│       └── UEFA Champions League 2004-2021.csv
│
├── graphs/
│   ├── top5_clubs.png
│   ├── confusion_matrix.png
│   └── outcomes.png
│
├── uefa_cl_model.py
└── README.md
```

---

## Key Takeaways

 **Draws dominate** Champions League match outcomes  
 **Barcelona** emerges as the most successful club in the dataset  
 **Naive Bayes** provides fast, interpretable predictions  
 **Tournament simulation** demonstrates practical application  
 **Visualizations** make insights accessible and compelling  

---


**Model Type**: Supervised Machine Learning (Classification)  
**Algorithm**: Categorical Naive Bayes  
**Dataset Period**: 2004-2021 UEFA Champions League Matches  
**Output**: Match predictions + Tournament simulation

---
