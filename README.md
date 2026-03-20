# Steam Hidden Gems Analysis
## Machine Learning Classification of Undiscovered Quality Games

**Tools:** Python | pandas | scikit-learn | matplotlib | seaborn  
**Dataset:** SteamSpy API — 10,000 Steam games  
**Notebook:** [01_steam_hidden_gems.ipynb](notebooks/steam_hidden_gems_analysis.ipynb)

---

## Project Overview
This project analyses 10,000 Steam games to identify "hidden gems",
games with high ratings and strong player engagement that remain 
undiscovered by the wider gaming community.

A custom "Steam Rating" metric was engineered from raw review data, 
and a binary classifier was built to predict hidden gem status using 
commercial features independent of the label definition.

> **Dataset Source:** SteamSpy API via Kaggle (CC0 Public Domain)

---

## Business Question
*Can a game's commercial features (price, discount, concurrent users) 
predict whether it is an undiscovered quality title?*

---

## Dataset
| Property | Detail |
|---|---|
| Source | SteamSpy API via Kaggle |
| Rows | 10,000 games |
| Columns | 17 |
| License | CC0 Public Domain |

---

## Project Structure
```
steam-hidden-gems-ml/
├── data/
│   └── steam_games_dataset.csv
├── notebooks/
│   └── 01_data_cleaning_eda.ipynb
├── outputs/
│   ├── eda_plots.png
│   └── final_model_comparison.png
└── README.md
```

---
## Methodology

### 1. Data Cleaning & Feature Engineering
- Dropped `userscore` — 99.95% zero values, no analytical value
- Converted `owners` range strings to numeric midpoints
- Converted prices from cents to USD
- Converted playtime from minutes to hours
- Engineered `steam_rating` = positive / (positive + negative reviews)
- Removed 88 duplicate Steam platform listings

### 2. Hidden Gem Definition
A game was labelled a hidden gem if it met all four criteria:
| Criterion | Threshold |
|---|---|
| Steam Rating | >= 0.85 |
| Avg Playtime | >= 2 hours |
| Owners | < 200,000 |
| Total Reviews | >= 10 |

**Result:** 1,013 hidden gems identified (10.2% of dataset)

### 3. Modelling
Two classifiers were trained and compared:
- **Logistic Regression** — baseline linear model
- **Random Forest** — ensemble method with 100 trees

Both models used `class_weight='balanced'` to address the 90/10 
class imbalance.

---

## Key Challenges and Findings

### Data Leakage Detected and Fixed
Initial Random Forest achieved perfect scores (ROC-AUC: 1.000). 
This was identified as data leakage — features included the exact 
variables used to define the hidden gem label. Features were 
restricted to three commercial variables independent of the label.

### Final Model Results

| Model | ROC-AUC | Hidden Gem Recall | F1 |
|---|---|---|---|
| Logistic Regression | **0.619** | **0.576** | 0.224 |
| Random Forest | 0.604 | 0.310 | 0.223 |

### Core Finding
> *Price, discount and concurrent users are weak predictors of 
> hidden gem status. A game's commercial strategy reveals almost 
> nothing about its quality or discovery status. Hidden gems are 
> defined by community response, not pricing.*

### Unexpected Result
Logistic Regression outperformed Random Forest across key metrics. 
With limited predictive signal, a simpler linear model generalised 
better than a complex ensemble method.

---

## Sample Hidden Gems Identified

| Game | Rating | Avg Playtime | Owners | Price |
|---|---|---|---|---|
| Patrick's Parabox | 0.99 | 7.5 hrs | 150,000 | $19.99 |
| Picayune Dreams | 0.99 | 11.0 hrs | 150,000 | $4.99 |
| Batman: Arkham City | 0.99 | 24.9 hrs | 75,000 | $3.99 |
| Your Turn To Die | 0.99 | 14.7 hrs | 150,000 | $16.99 |
| I Am Your Beast | 0.98 | 4.0 hrs | 150,000 | $13.99 |

---

## Future Improvements
- Include genre and tag data as categorical features
- Apply NLP to game descriptions for quality signals
- Incorporate release date and game age as features
- Experiment with XGBoost or LightGBM
- Collect additional independent features via Steam API

---

## Author
**Brianna Owens**

[LinkedIn](https://www.linkedin.com/in/brianna-owens-42253223a) | 
[Tableau Public](https://public.tableau.com/app/profile/brianna.owens) |
