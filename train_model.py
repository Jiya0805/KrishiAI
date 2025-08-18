import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Starting model training and hyperparameter tuning...")

# 1. Load the dataset
try:
    df = pd.read_csv('Crop_recommendation.csv')
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found. Please place the file in the project's root folder.")
    exit()

# 2. Define features (X) and target (y)
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],  # None means no limit
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 5. Initialize GridSearchCV
# It will test every combination of parameters defined in the grid.
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,            # Use 5-fold cross-validation
                           n_jobs=-1,       # Use all available CPU cores
                           verbose=2)       # Show progress

# 6. Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# 7. Get the best model
best_model = grid_search.best_estimator_

# 8. Print the best parameters and the final score
print("\n--- Hyperparameter Tuning Complete ---")
print("Best parameters found: ", grid_search.best_params_)
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

# 9. Evaluate the best model on the unseen test set
y_pred = best_model.predict(X_test)
print(f"\nAccuracy of the best model on the test set: {accuracy_score(y_test, y_pred):.2f}")
print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

# 10. Save the best model
project_root = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(project_root, 'backend', 'ml_models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'crop_predictor_model.pkl')
joblib.dump(best_model, model_path)

print(f"\nOptimized model successfully trained and saved to: {model_path}")