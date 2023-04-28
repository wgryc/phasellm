# Chain of Thought Plan for Data Analysis

1. Descriptive statistics and data cleaning:
   Hypothesis: Understanding the distribution of numerical variables and identifying missing or inconsistent data for preprocessing.
   Variables: All columns.
   Task: Calculate descriptive statistics (mean, median, standard deviation, etc.) for numerical columns and inspect the unique values and counts for categorical columns. Handle missing values and inconsistencies in the data.

2. Income distribution analysis:
   Hypothesis: Identifying the distribution of income across the dataset.
   Variables: income.
   Task: Analyze and calculate the proportion of individuals in each income category.

3. Correlation analysis among numeric variables:
   Hypothesis: Certain numeric variables may be associated with income.
   Variables: age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week.
   Task: Calculate the correlation matrix among numeric variables and determine which variables have a strong correlation with income or with each other.

4. Income distribution by categorical variables:
   Hypothesis: Categorical variables have an impact on the distribution of income.
   Variables: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
   Task: Group the data by each categorical variable and calculate the proportion of individuals in each income group within each category.

5. Statistical significance testing:
   Hypothesis: There are statistically significant differences in income levels based on the categorical variables.
   Variables: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
   Task: Perform a Chi-squared test or an ANOVA test for each categorical variable to determine if the differences in income distribution across their categories are statistically significant.

6. Feature encoding:
   Hypothesis: Encoding categorical variables improves the performance of machine learning models.
   Variables: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
   Task: Encode categorical variables using one-hot encoding or ordinal encoding based on their nature.

7. Feature selection:
   Hypothesis: Not all features may be useful in predicting income. Selecting important features can improve model performance.
   Variables: All columns except income.
   Task: Perform feature selection techniques such as Recursive Feature Elimination or SelectKBest (based on mutual information or F-value) to determine the best features for predicting income levels.

8. Model development:
   Hypothesis: A machine learning model can accurately predict income levels based on the selected features.
   Variables: Selected features from step 7 and income.
   Task: Split the data into training and testing sets, train machine learning models (e.g., logistic regression, random forest, xgboost) on the training set and evaluate their performance on the test set using appropriate metrics (precision, recall, F1-score, and/or AUC-ROC).

9. Model interpretation:
   Hypothesis: The machine learning model can provide insights into which features are most important in determining income levels.
   Variables: Features used in the model from step 8.
   Task: Use techniques like feature importances (from the model) or permutation importance to understand which features have the most impact on income predictions.

10. Conclusion and recommendations:
    Hypothesis: The analysis reveals insights into income inequality and suggests potential interventions.
    Variables: All columns.
    Task: Summarize the results from the previous analysis steps, identify key drivers of income inequality, and provide recommendations to address these issues.

# Code for Step #1

1. Descriptive statistics and data cleaning:
   Hypothesis: Understanding the distribution of numerical variables and identifying missing or inconsistent data for preprocessing.
   Variables: All columns.
   Task: Calculate descriptive statistics (mean, median, standard deviation, etc.) for numerical columns and inspect the unique values and counts for categorical columns. Handle missing values and inconsistencies in the data.

```python
# Calculate descriptive statistics for numerical columns
numerical_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
desc_stats = df[numerical_columns].describe()

# Inspect unique values and counts for categorical columns
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country', 'income']
unique_values_counts = df[categorical_columns].apply(lambda x: x.value_counts(dropna=False))

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Handle missing values
# Replace NaN in 'workclass', 'occupation', and 'native.country' with 'Unknown'
df['workclass'].fillna('Unknown', inplace=True)
df['occupation'].fillna('Unknown', inplace=True)
df['native.country'].fillna('Unknown', inplace=True)

# Print results
print("Descriptive statistics for numerical columns:")
print(desc_stats)
print("\nUnique values counts for categorical columns:")
print(unique_values_counts)
```


# Code for Step #2

2. Income distribution analysis:
   Hypothesis: Identifying the distribution of income across the dataset.
   Variables: income.
   Task: Analyze and calculate the proportion of individuals in each income category.

```python
# Calculate the proportion of individuals in each income category
income_counts = df['income'].value_counts()
income_proportions = income_counts / len(df)

# Print the proportions of each income group
print("Proportion of individuals in each income category:")
print(income_proportions)
```


# Code for Step #3

3. Correlation analysis among numeric variables:
   Hypothesis: Certain numeric variables may be associated with income.
   Variables: age, fnlwgt, education.num, capital.gain, capital.loss, hours.per.week.
   Task: Calculate the correlation matrix among numeric variables and determine which variables have a strong correlation with income or with each other.

```python
# Convert income to a binary variable (0 for '<=50K' and 1 for '>50K')
df['income_binary'] = df['income'].apply(lambda x: 0 if x == '<=50K' else 1)

# Calculate correlation matrix among numeric variables and the binary income variable
correlation_matrix = df[numerical_columns + ['income_binary']].corr()

# Print the correlation matrix
print("Correlation matrix among numeric variables and income:")
print(correlation_matrix)

# Remove the binary income column from the dataframe
df.drop(columns=['income_binary'], inplace=True)
```


# Code for Step #4

4. Income distribution by categorical variables:
   Hypothesis: Categorical variables have an impact on the distribution of income.
   Variables: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
   Task: Group the data by each categorical variable and calculate the proportion of individuals in each income group within each category.

```python
# Create an empty dictionary to store the income distribution for each categorical variable
income_distribution_by_categorical = {}

# Iterate through the categorical variables and compute the income distribution
for column in categorical_columns[:-1]:  # Exclude 'income' from the list
    income_group_counts = df.groupby(column)['income'].value_counts()
    total_counts = df[column].value_counts()
    income_proportion = (income_group_counts / total_counts).unstack()
    income_distribution_by_categorical[column] = income_proportion

# Print the income distribution by categorical variables
for column, distribution in income_distribution_by_categorical.items():
    print(f"Income distribution by {column}:")
    print(distribution)
    print("\n")
```


# Code for Step #5

5. Statistical significance testing:
   Hypothesis: There are statistically significant differences in income levels based on the categorical variables.
   Variables: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
   Task: Perform a Chi-squared test or an ANOVA test for each categorical variable to determine if the differences in income distribution across their categories are statistically significant.

```python
from scipy.stats import chi2_contingency

# Create an empty dictionary to store the results of the Chi-squared tests
chi_squared_results = {}

# Iterate through the categorical variables and perform the Chi-squared test
for column in categorical_columns[:-1]:  # Exclude 'income' from the list
    contingency_table = pd.crosstab(df[column], df['income'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    chi_squared_results[column] = {'Chi2': chi2, 'p-value': p_value}

# Print the results of the Chi-squared tests for the categorical variables
for column, result in chi_squared_results.items():
    print(f"Results of Chi-squared test for {column}:")
    print(f"Chi2: {result['Chi2']}, p-value: {result['p-value']}")
    print("\n")
```


# Code for Step #6

6. Feature encoding:
   Hypothesis: Encoding categorical variables improves the performance of machine learning models.
   Variables: workclass, education, marital.status, occupation, relationship, race, sex, native.country.
   Task: Encode categorical variables using one-hot encoding or ordinal encoding based on their nature.

```python
# Use one-hot encoding for nominal categorical variables
nominal_columns = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']
one_hot_encoded_df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

# Use ordinal encoding for ordinal categorical variables
education_order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Masters', 'Doctorate', 'Prof-school']
education_mapping = {edu_level: i for i, edu_level in enumerate(education_order)}
one_hot_encoded_df['education'] = one_hot_encoded_df['education'].map(education_mapping)

# Encode the 'sex' column as binary (0 for 'Female' and 1 for 'Male')
one_hot_encoded_df['sex'] = one_hot_encoded_df['sex'].apply(lambda x: 0 if x == 'Female' else 1)

# Print the first few rows of the encoded dataframe
print(one_hot_encoded_df.head())
```


# Code for Step #7

7. Feature selection:
   Hypothesis: Not all features may be useful in predicting income. Selecting important features can improve model performance.
   Variables: All columns except income.
   Task: Perform feature selection techniques such as Recursive Feature Elimination or SelectKBest (based on mutual information or F-value) to determine the best features for predicting income levels.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# Prepare the data
X = one_hot_encoded_df.drop(columns=['income'])
y = one_hot_encoded_df['income'].apply(lambda x: 0 if x == '<=50K' else 1)

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform feature selection using SelectKBest with mutual_info_classif
num_features = 10
selector = SelectKBest(mutual_info_classif, k=num_features)
selector.fit(X_scaled, y)
selected_features = selector.transform(X_scaled)

# Get the names of the selected features
selected_feature_names = X.columns[selector.get_support()]

# Print the selected features
print(f"Top {num_features} features selected:")
print(selected_feature_names)
```


# Code for Step #8

8. Model development:
   Hypothesis: A machine learning model can accurately predict income levels based on the selected features.
   Variables: Selected features from step 7 and income.
   Task: Split the data into training and testing sets, train machine learning models (e.g., logistic regression, random forest, xgboost) on the training set and evaluate their performance on the test set using appropriate metrics (precision, recall, F1-score, and/or AUC-ROC).

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Make predictions using the trained models
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

# Calculate metrics for the models
lr_metrics = precision_recall_fscore_support(y_test, lr_preds, average='weighted')
rf_metrics = precision_recall_fscore_support(y_test, rf_preds, average='weighted')
lr_accuracy = accuracy_score(y_test, lr_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)
lr_auc = roc_auc_score(y_test, lr_preds)
rf_auc = roc_auc_score(y_test, rf_preds)

# Print the metrics
print("Logistic Regression Metrics:")
print(f"Accuracy: {lr_accuracy}, Precision: {lr_metrics[0]}, Recall: {lr_metrics[1]}, F1-score: {lr_metrics[2]}, AUC-ROC: {lr_auc}")

print("\nRandom Forest Metrics:")
print(f"Accuracy: {rf_accuracy}, Precision: {rf_metrics[0]}, Recall: {rf_metrics[1]}, F1-score: {rf_metrics[2]}, AUC-ROC: {rf_auc}")
```


# Code for Step #9

9. Model interpretation:
   Hypothesis: The machine learning model can provide insights into which features are most important in determining income levels.
   Variables: Features used in the model from step 8.
   Task: Use techniques like feature importances (from the model) or permutation importance to understand which features have the most impact on income predictions.

```python
from sklearn.inspection import permutation_importance

# Get feature importances from the Random Forest model
rf_feature_importances = rf_model.feature_importances_

# Compute permutation importance for the Logistic Regression model using the test set
lr_perm_importance = permutation_importance(lr_model, X_test, y_test, n_repeats=10, random_state=42)

# Map feature importances to the feature names
rf_importance_dict = dict(zip(selected_feature_names, rf_feature_importances))
lr_importance_dict = dict(zip(selected_feature_names, lr_perm_importance.importances_mean))

# Sort the feature importances
rf_sorted_importances = sorted(rf_importance_dict.items(), key=lambda x: x[1], reverse=True)
lr_sorted_importances = sorted(lr_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the feature importances
print("Feature importances for Random Forest:")
for feature, importance in rf_sorted_importances:
    print(f"{feature}: {importance}")

print("\nPermutation importances for Logistic Regression:")
for feature, importance in lr_sorted_importances:
    print(f"{feature}: {importance}")
```


