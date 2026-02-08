import pandas as pd
from datasets import load_dataset

HF_DATASET = "ruslanmv/ai-medical-chatbot"

def generate_dataset_for_manual_labeling(total_count=1500):
    print(f"⬇️ Downloading data and extracting {total_count} scenarios...")
    try:
        dataset = load_dataset(HF_DATASET, split="train", streaming=True)
        results = []
        seen_texts = set()
        
        for row in dataset:
            if len(results) >= total_count: break
            
            patient_text = row.get('Patient', '').strip()
            if len(patient_text) < 25 or patient_text in seen_texts: continue 
            
            seen_texts.add(patient_text)
            results.append({
                "ID": len(results) + 1,
                "text": patient_text,
                "original_doctor": row.get('Doctor', '').strip(),
                "summary": row.get('Description', '').strip(),
                "category": "",           # e.g., Respiratory, GI
                "urgency": "",            # e.g., Mild, Alarming, Critical
                "emotional_loading": "",  # e.g., Anxiety, Fear, Neutral
                "ambiguity": "",          # e.g., Clear, Vague
                "risk_label": ""          # e.g., High Risk or Low Risk
            })

        df = pd.DataFrame(results)
        df.to_csv("validation_dataset.csv", index=False)
        print(f"✅ Created 'validation_dataset.csv' with {len(df)} rows.")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_dataset_for_manual_labeling(1500)