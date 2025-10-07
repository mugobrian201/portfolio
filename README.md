# Data Science Portfolio
👋 Introduction

Welcome to my Data Science Portfolio! This GitHub repository showcases four projects that highlight my skills in data analysis, machine learning, visualization, and predictive modeling. All datasets used in these projects were sourced from Kaggle. Each project demonstrates a complete workflow, from data cleaning and exploratory data analysis (EDA) to model building and interpretation.

The projects cover diverse domains including real estate, air quality forecasting, and disaster risk assessment. They emphasize practical applications, skill-building in Python libraries, and real-world impact. Feel free to explore the code, notebooks, and datasets in their respective folders.


## Skills Demonstrated

![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&style=flat-square)

📊 Data Handling & Cleaning
Efficiently processed and prepared datasets by addressing missing values, outliers, and time series data for robust analysis.
Pandas, NumPy, Time Series Resampling, Outlier Removal

📈 Visualization
Create insightful and interactive visualizations, including histograms, boxplots, scatter plots, and bar charts to communicate data trends effectively.
Matplotlib, Seaborn, Plotly Express, Scatter Mapbox

🤖 Machine Learning
Develop predictive models using supervised and time series algorithms, with a focus on iterative improvement and validation.
Linear Regression, Logistic Regression, Decision Trees, AR/ARMA (Scikit-learn, Statsmodels)

🔧 Feature Engineering
Enhance model performance through techniques like lag feature creation, price per unit calculations, and categorical encoding.
Lag Features, Price per m², OneHotEncoder, OrdinalEncoder

📏 Model Evaluation
Assess model performance using metrics like Mean Absolute Error (MAE), accuracy, residuals analysis, and walk-forward validation.
MAE, Accuracy Score, Residuals Analysis, Validation Curves

⚙️ Other
Apply advanced techniques such as timezone handling, hyperparameter tuning, and model interpretation to solve real-world problems.
Timezone Handling (pytz), Hyperparameter Tuning, Gini Importances, Coefficient Analysis



## Projects
### Project 1: What Drives Real Estate Prices in Mexico? A Data-Driven Analysis

This project investigates the key factors influencing house prices in Mexico, focusing on property size (surface_covered_in_m2) and location (latitude, longitude, and states). The goal was to determine whether location or size has a greater impact on prices (price_aprox_usd).

#### Key Steps:
- **Data Preparation**: Imported data using Pandas and NumPy. Inspected the dataset, dropped irrelevant columns with excessive NaN values, and removed remaining NaNs.
- **Exploratory Data Analysis (EDA)**: 
  - Visualized value counts of states.
  - Created scatter plots and Scatter Mapbox for geographic distribution.
  - Generated histograms and boxplots for surface area and prices, interpreting distributions, outliers, and skewness.
  - Plotted bar charts for mean house prices by state.
  - Engineered a new feature: price per square meter (price_aprox_usd / surface_covered_in_m2).
  - Visualized mean price per m² by state via bar charts.
- **Tools & Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly Express.
- **Findings**: Through visualizations and analysis, I concluded that property size is the dominant factor in determining sale prices, explaining more variation than geographic location alone.

#### Real-World Importance:
Understanding real estate drivers aids investors, buyers, and policymakers in making informed decisions. This analysis can inform urban planning, pricing strategies, and economic forecasting in emerging markets like Mexico, where location-based assumptions might overlook size-related value.

[View the Notebook](project1/notebook.ipynb) | [Dataset on Kaggle](https://www.kaggle.com/datasets/masoosali/mexico-real-estate)

---

### Project 2: Predicting Apartment Prices in Buenos Aires: A Data Science Approach Using Size and Location




This project builds a predictive model for apartment prices in Buenos Aires, Argentina, using Linear Regression. It explores the influence of size and location, progressing through three iterative models to demonstrate the full data science lifecycle.

#### Key Steps:
- **Data Preparation**: Imported and explored data, split into features and target.
- **Model Building (Three Parts)**:
  1. **Size Only**: Built a baseline Linear Regression model using property size.
  2. **Location Only**: Modeled prices based solely on location features.
  3. **Size + Location**: Combined both for the most accurate predictions.
- For each part: Prepared data (import, explore, split), built models (baseline, iterate, evaluate), and communicated results by extracting intercepts/coefficients to form prediction equations.
- **Feature Engineering & Pipelines**: Used SimpleImputer for missing values and Pipeline for streamlined processing.
- **Evaluation**: Calculated Mean Absolute Error (MAE); no separate test set was used, focusing on iterative improvement.
- **Tools & Libraries**: Pandas, NumPy, Matplotlib, Plotly (Graph Objects & Express), Seaborn, Scikit-learn (LinearRegression, MAE, Pipeline, SimpleImputer).

#### Real-World Importance:
This workflow provides a tool for real estate estimation, helping buyers, sellers, and agents predict values accurately. It showcases iterative modeling, essential for real-world problems like market forecasting, and can extend to investment analysis or policy-making in housing markets.

[View the Notebook](project2/notebook.ipynb) | [Dataset on Kaggle](https://www.kaggle.com/datasets/undefined) *(Replace with actual link if available)*

---

### Project 3: Beijing Air Quality Analysis Project




This time series analysis predicts PM2.5 levels in Beijing, China, using three models: Linear Regression, AutoRegressive (AR), and ARMA. The focus is on forecasting dangerous particulate matter for environmental monitoring.

#### Key Steps:
- **Data Preparation**: Localized timezones with pytz, explored data, removed outliers, resampled to daily frequency, filled NaNs with forward fill, and plotted rolling averages.
- **Model Building (Three Parts)**:
  1. **Linear Regression**: Created lag features for PM2.5; predicted using only lags (no other features).
  2. **AR Model**: Used Statsmodels for autoregression, checked residuals.
  3. **ARMA Model**: Extended to include moving averages, performed walk-forward validation.
- **Evaluation**: Split into train/test sets, plotted predictions, used MAE, and analyzed residuals. Surprisingly, Linear Regression outperformed AR and ARMA.
- **Tools & Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly Express, Scipy, Scikit-learn (LinearRegression, MAE, train_test_split), Statsmodels (ARIMA, AR, ACF/PACF, ADFuller), datetime, time, warnings.

#### Real-World Importance:
This project offers a blueprint for pollution forecasting using historical data, enabling early warning systems. In cities like Beijing, it can inform public health alerts, policy interventions, and urban planning to reduce exposure to harmful air quality, potentially saving lives.

[View the Notebook](project3/notebook.ipynb) | [Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/pm25-data-for-five-chinese-cities) *(Assumed based on description; confirm link)*

---

### Project 4: Earthquake Damage in Nepal




This project analyzes data from the 2015 Nepal earthquake to predict severe building damage using Logistic Regression and Decision Trees. It aims to identify vulnerable structures based on characteristics for better disaster preparedness.

#### Key Steps:
- **Data Preparation**: Created a wrangle function for cleaning; prepared binary target (severe damage vs. not).
- **Part 1: Logistic Regression**:
  - Explored class balance, performed train-test split.
  - Set baseline accuracy, iterated with Logistic Regression.
  - Evaluated with accuracy score; interpreted predictions using odds ratios.
- **Part 2: Decision Tree Classifier**:
  - Three-way split (train, validation, test).
  - Used OrdinalEncoder for categorical variables.
  - Tuned hyperparameters with validation curves.
  - Explained predictions via Gini importances and tree plotting.
- **Tools & Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Category Encoders (OneHotEncoder, OrdinalEncoder), Scikit-learn (LogisticRegression, DecisionTreeClassifier, accuracy_score, Pipeline, train_test_split, plot_tree), warnings.

#### Real-World Importance:
By predicting damage levels, this model aids in resource allocation, targeted rescues, and risk mitigation in earthquake-prone areas. It helps identify vulnerable building features, informing retrofitting and building codes to reduce mortality rates in future disasters, especially in remote terrains like Nepal.

[View the Notebook](project4/notebook.ipynb) | [Dataset on Kaggle](https://www.kaggle.com/datasets/undefined) *(Replace with actual link, e.g., Nepal Earthquake dataset)*

## Contact
If you'd like to discuss these projects or collaborate, reach out via [LinkedIn](https://www.linkedin.com/in/your-profile) or email at your.email@example.com.

Thank you for visiting!
