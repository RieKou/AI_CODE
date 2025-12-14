# generate_dummy.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))

def generate_dummy(n=500, filename="tb_dummy_500.csv"):
    data = []
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    for i in range(n):
        onset = random_date(start_date, end_date)
        diagnosis = onset + timedelta(days=random.randint(5, 120))

        entry = {
            "patient_id": f"P{i+1:04d}",
            "age": random.randint(18, 80),
            "sex": random.choice(["M", "F"]),
            "occupation": random.choice(["farmer", "student", "teacher", "driver", "office"]),
            "education_level": random.choice(["none", "primary", "secondary", "university"]),
            "socioeconomic_proxy": random.choice(["BPJS", "private", "none"]),
            "symptom_onset_date": onset,
            "first_visit_date": onset + timedelta(days=random.randint(1, 20)),
            "diagnosis_date": diagnosis,
            "cough_duration_days": random.randint(5, 90),
            "hemoptysis": random.choice([0, 1]),
            "weight_loss": random.choice([0, 1]),
            "fever_night_sweats": random.choice([0, 1]),
            "smoking_status": random.choice(["never", "former", "current"]),
            "contact_with_TB_case": random.choice([0, 1]),
            "comorbidity_diabetes": random.choice([0, 1]),
            "comorbidity_HIV": random.choice([0, 1]),
            "xray_findings": random.choice(["normal", "suspicious", "typical"]),
            "geneXpert_result": random.choice(["positive", "negative", "not_done"]),
            "distance_to_healthcare_km": round(random.uniform(0.5, 50.0), 1)
        }

        data.append(entry)

    df = pd.DataFrame(data)

    df["delay_days"] = (df["diagnosis_date"] - df["symptom_onset_date"]).dt.days
    df["long_delay"] = (df["delay_days"] > 30).astype(int)

    df.to_csv(filename, index=False)
    print(f"Generated {n} rows → saved to {filename}")

if __name__ == "__main__":
    generate_dummy(500)
