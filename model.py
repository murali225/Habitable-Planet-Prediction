import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Load the excel file
df = pd.read_excel("Exoplanets.xlsx")

# Name of the columns and strip any case sensitivity
print(df.columns)
df.columns = df.columns.str.strip()

print(df.head())

# Separate the target and features
X = df[["Name", "Mass", "Period", "Discovery method", "Distance", "Host star temp"]]
y = df["Habitability"]

print(X.shape)
print(y.shape)

# Preprocessing steps:
# - Use TF-IDF for each text column separately
# - Standardize numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('name', TfidfVectorizer(), 'Name'),  # Convert 'Name' column to numeric
        ('discovery', TfidfVectorizer(), 'Discovery method'),  # Convert 'Discovery method' column to numeric
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing data
            ('scaler', StandardScaler())  # Scale numerical features
        ]), ['Mass', 'Period', 'Distance', 'Host star temp'])
    ]
)

# Create a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Combine the preprocessor and model in a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

# Train the model
pipeline.fit(X_train, y_train)

# Make pickle file of our model
with open("model.pkl", "wb") as file:
    pickle.dump(pipeline, file)
