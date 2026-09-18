# EPL Player Performance Analysis

**UMBC DATA601: Introduction to Data Science** · Spring 2024  
**Dataset:** [Football Players Stats (Premier League 2021-2022)](https://www.kaggle.com/datasets/omkargowda/football-players-stats-premier-league-20212022) · Kaggle

Two-notebook series: exploratory data analysis followed by predictive modeling. The question driving both: what can player statistics tell us about individual output, and how far can a naive model take us before it hits the limits that better projection systems are built to solve?

---

## Notebooks

### 1. EDA — `DATA_601_Upadhyay_XT81177_EDAProject_21stApril-2.ipynb`

Pure exploration of EPL 2021-22 player statistics across all 20 clubs.

**Team performance analysis** — aggregated goals, assists, and disciplinary records by club; ranked by offensive output. Identifies whether clubs are built around volume scorers or distributed contributors.

**Player performance analysis** — top 10 scorers identified; goal distribution across the full player population is heavily right-skewed, confirming that elite output is concentrated in very few players.

**Univariate analysis** — histograms with KDE and boxplots for goals, assists, yellow cards, red cards, and age. Goal and assist distributions both heavily skewed; most players contribute very little offensively.

**Bivariate analysis** — scatter plot of goals vs assists reveals moderate positive correlation. Some players are pure finishers with low assist numbers; others contribute primarily through creation. Pairplot across all key numeric features.

**Multivariate analysis** — correlation heatmap across all numeric features; pairplot stratified by player position (FW, MF, DF, GK). Distinct performance clusters visible by role.

**Key finding:** forward position is the strongest predictor of goal output, but midfielders show the widest variance across both goals and assists — player projection uncertainty is highest in this group.

---

### 2. Predictive Modeling — `DATA_601_Upadhyay_XT81177_FinalProject_16thMay.ipynb`

Extends the EDA with a full preprocessing and modeling pipeline.

**Preprocessing**
- One-hot encoding applied to player position — position is one of the strongest structural predictors of output
- StandardScaler normalization across all numeric features
- Z-score outlier removal (threshold: 3 standard deviations) to prevent the elite tail from dominating the model

**Model**
- Target: goals scored (Gls)
- Features: assists (Ast), yellow cards (CrdY), red cards (CrdR), age
- Train/test split: 80/20, random_state=42
- Model: sklearn LinearRegression on normalized features

**Evaluation**
- RMSE and R-squared on held-out test set
- Actual vs predicted scatter plot with ideal prediction diagonal
- Residual plot to check for systematic error and heteroscedasticity

**Key finding:** the model captures broad output patterns but underperforms on individual high-scorers. The residual plot reveals heteroscedasticity — prediction error increases at higher predicted values, meaning the model is least reliable precisely for the players we most care about projecting. This maps the gap between a naive first model and a production projection system: what is missing is playing time, expected goals (xG), shot quality, opponent strength, and nonlinear methods that can handle interaction effects between position and system role.

---

## Stack

`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

---

## Files

| File | Description |
|---|---|
| `DATA_601_Upadhyay_XT81177_EDAProject_21stApril-2.ipynb` | Exploratory data analysis |
| `DATA_601_Upadhyay_XT81177_FinalProject_16thMay.ipynb` | Preprocessing and predictive modeling |
| `EPL.csv` | Dataset (source: Kaggle) |

---

*This work is part of a broader sports analytics portfolio. Nine published analytical articles on EPL 2023 are available at [goworldwide.co.in](https://goworldwide.co.in).*
