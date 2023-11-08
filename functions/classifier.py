
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
# Import config from the parent directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)
import config  

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    lowercase=True,
    token_pattern=r'\w{3,}'
)

def job_cat_data_process(dataframe, job_categories):

    X = vectorizer.fit_transform(dataframe['Job_Description'])
    y = dataframe[job_categories]
    return X, y


def train_multilabel_classifier(X, y, job_categories):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    multioutput_classifier = MultiOutputClassifier(rf_classifier)
    multioutput_classifier.fit(X_train, y_train)
    y_pred = multioutput_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Multilabel Classifier Model accuracy:", accuracy)
    report = classification_report(y_test, y_pred, target_names=job_categories, zero_division=0)
    print("Classification Report for Multilabel Classifier:\n", report)
    return y_test, y_pred

def train_xgboost_classifier(X, y, job_categories):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_classifier = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_xgb_classifier = XGBClassifier(**best_params, random_state=42)
    best_xgb_classifier.fit(X_train, y_train)
    y_pred = best_xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Classifier Model accuracy:", accuracy)
    report = classification_report(y_test, y_pred, target_names=job_categories, zero_division=0)
    print("Classification Report for XGBoost Classifier:\n", report)
    return y_test, y_pred

# Example usage:
X, y = job_cat_data_process(config.df, config.job_categories)

# Multilabel Classifier
y_test_mc, y_pred_mc = train_multilabel_classifier(X, y, config.job_categories)

# XGBoost Classifier
y_test_xgb, y_pred_xgb = train_xgboost_classifier(X, y, config.job_categories)

