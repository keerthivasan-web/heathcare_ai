import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Disease info dictionary
disease_info = {
    'Heart Disease': {
        'Symptoms': 'Chest pain, shortness of breath, fatigue.',
        'Causes': 'High blood pressure, cholesterol, smoking.',
        'Treatment': 'Lifestyle changes, medication, surgery.'
    },
    'Diabetes': {
        'Symptoms': 'Increased thirst, frequent urination, fatigue.',
        'Causes': 'Insulin resistance, obesity, genetics.',
        'Treatment': 'Diet control, insulin therapy, exercise.'
    },
    'Hypertension': {
        'Symptoms': 'Often none, but may include headaches or dizziness.',
        'Causes': 'Obesity, stress, high salt intake.',
        'Treatment': 'Exercise, low-salt diet, medication.'
    },
    'COPD': {
        'Symptoms': 'Persistent cough, mucus production, fatigue, breathlessness.',
        'Causes': 'Smoking, environmental exposures, genetic factors.',
        'Treatment': 'Smoking cessation, inhalers, medications like ensifentrine and dupilumab.'
    },
    'Kidney Cancer': {
        'Symptoms': 'Blood in urine, back pain, fatigue.',
        'Causes': 'Smoking, obesity, high blood pressure.',
        'Treatment': 'Surgery, targeted therapy, immunotherapy.'
    },
    'Syphilis': {
        'Symptoms': 'Sores, rashes, hair loss on head and body.',
        'Causes': 'Bacterial infection through sexual contact.',
        'Treatment': 'Antibiotics, typically penicillin.'
    },
    'Myeloma': {
        'Symptoms': 'Bone pain, fatigue, frequent infections.',
        'Causes': 'Cancer of plasma cells in bone marrow.',
        'Treatment': 'Chemotherapy, stem cell transplant, clinical trials.'
    },
    'Behçet\'s Disease': {
        'Symptoms': 'Mouth sores, genital sores, eye inflammation, arthritis.',
        'Causes': 'Unknown; possibly genetic and environmental factors.',
        'Treatment': 'Immunosuppressive medications, corticosteroids.'
    },
    'None': {
        'Symptoms': 'No symptoms.',
        'Causes': 'No causes.',
        'Treatment': 'No treatment needed.'
    }
}


def create_sample_data():
    np.random.seed(42)
    n_samples = 100

    data = {
        'age': np.random.randint(1, 101, size=n_samples),
        'gender': np.random.choice(['m', 'f'], size=n_samples),
        'blood_pressure': np.random.randint(90, 201, size=n_samples),
        'cholesterol': np.random.randint(90, 201, size=n_samples),
        'glucose': np.random.randint(90, 201, size=n_samples),
        'bmi': np.round(np.random.uniform(20.0, 40.0, size=n_samples), 1),
        'smoking': np.random.choice(['yes', 'no'], size=n_samples),
        'family_history': np.random.choice(['yes', 'no'], size=n_samples),
        'physical_activity': np.random.choice(['low', 'moderate', 'high'], size=n_samples),
        'disease': np.random.choice(['Heart Disease', 'Diabetes', 'Hypertension', 'None'], size=n_samples)
    }

    df = pd.DataFrame(data)
    df.to_csv("patient_data.csv", index=False)
    print("✅ Random sample data (100 rows) created as patient_data.csv")


def train_model():
    df = pd.read_csv("patient_data.csv")

    X = df.drop('disease', axis=1)
    y = df['disease']

    le_gender = LabelEncoder()
    le_smoke = LabelEncoder()
    le_family = LabelEncoder()
    le_activity = LabelEncoder()
    le_disease = LabelEncoder()

    X['gender'] = le_gender.fit_transform(X['gender'])
    X['smoking'] = le_smoke.fit_transform(X['smoking'])
    X['family_history'] = le_family.fit_transform(X['family_history'])
    X['physical_activity'] = le_activity.fit_transform(X['physical_activity'])
    y_encoded = le_disease.fit_transform(y)

    joblib.dump(le_gender, 'le_gender.pkl')
    joblib.dump(le_smoke, 'le_smoke.pkl')
    joblib.dump(le_family, 'le_family.pkl')
    joblib.dump(le_activity, 'le_activity.pkl')
    joblib.dump(le_disease, 'le_disease.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'disease_model.pkl')
    print("✅ Model trained and saved as disease_model.pkl")


def predict_disease():
    print("\n🔮 Welcome to the AI-powered Disease Prediction System")
    print("Please enter the following patient details:")

    age = int(input("Age: "))
    gender = input("Gender (M/F): ").lower()
    bp = int(input("Blood Pressure: "))
    chol = int(input("Cholesterol: "))
    glu = int(input("Glucose Level: "))
    bmi = float(input("BMI: "))
    smoking = input("Smoking (Y/N): ").lower()
    family = input("Family History of Disease (Y/N): ").lower()
    activity = input("Physical Activity (Low/Moderate/High): ").lower()

    model = joblib.load('disease_model.pkl')
    le_gender = joblib.load('le_gender.pkl')
    le_smoke = joblib.load('le_smoke.pkl')
    le_family = joblib.load('le_family.pkl')
    le_activity = joblib.load('le_activity.pkl')
    le_disease = joblib.load('le_disease.pkl')

    try:
        gender_enc = le_gender.transform([gender])[0]
        smoking_enc = le_smoke.transform(['yes' if smoking == 'y' else 'no'])[0]
        family_enc = le_family.transform(['yes' if family == 'y' else 'no'])[0]
        activity_enc = le_activity.transform([activity])[0]

        input_dict = {
            'age': [age],
            'gender': [gender_enc],
            'blood_pressure': [bp],
            'cholesterol': [chol],
            'glucose': [glu],
            'bmi': [bmi],
            'smoking': [smoking_enc],
            'family_history': [family_enc],
            'physical_activity': [activity_enc]
        }

        input_df = pd.DataFrame(input_dict)
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[0]

        predicted_label = le_disease.inverse_transform(prediction)[0]

        print(f"\n🧬 Predicted Disease: **{predicted_label}**")

        print("\n📊 Prediction Probabilities:")
        for i, prob in enumerate(probabilities):
            label = le_disease.inverse_transform([i])[0]
            print(f"- {label}: {prob * 100:.2f}%")

        print("\n📋 Disease Information:")
        info = disease_info.get(predicted_label, {})
        for key, value in info.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"⚠️ Error in prediction: {e}")


if __name__ == "__main__":
    if not os.path.exists("patient_data.csv"):
        create_sample_data()
        train_model()
    elif not os.path.exists("disease_model.pkl"):
        train_model()
    predict_disease()
