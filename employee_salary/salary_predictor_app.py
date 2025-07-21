import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("Salary Data.csv")  # Make sure the file is in the same folder
    return df.dropna()

df = load_data()

# Rename columns for consistency
df = df.rename(columns={
    "Education Level": "Qualification",
    "Job Title": "Job Role",
    "Years of Experience": "Experience"
})

# Define X and y
X = df.drop(columns=["Salary"])
y = df["Salary"]

# Categorical columns
categorical_cols = ["Gender", "Qualification", "Job Role"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")


model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# Streamlit UI
# --------------------------
st.title("💼 Employee Salary Predictor")

age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", df["Gender"].unique())
qualification = st.selectbox("Qualification", df["Qualification"].unique())
job_role = st.selectbox("Job Role", df["Job Role"].unique())
experience = st.number_input("Years of Experience", 0, 50, 2)

if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Qualification": qualification,
        "Job Role": job_role,
        "Experience": experience
    }])
    predicted_salary = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: **₹{predicted_salary:,.2f}**")
