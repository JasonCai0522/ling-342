import json
from pathlib import Path

import pandas as pd
from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="HR26kk/modernbert-emotion-classifier",
    top_k=None,
)

df = pd.read_csv("stories.csv")

predictions = []
for text in df["responses"]:
    scores = emotion_classifier(text, top_k=None, truncation=True)
    prediction = max(scores, key=lambda result: result["score"])
    predictions.append({
        "predicted_emotion": prediction["label"],
        "is_angry": prediction["label"] == "anger",
        **{f"score_{result['label']}": result["score"] for result in scores},
    })

results = pd.concat([df, pd.DataFrame(predictions)], axis=1)
results.to_csv("stories_classified.csv", index=False)

summary = {
    "total_stories": len(predictions),
    "emotion_counts": {},
    "angry": 0,
    "not_angry": 0,
}
for prediction in predictions:
    emotion = prediction["predicted_emotion"]
    summary["emotion_counts"][emotion] = summary["emotion_counts"].get(emotion, 0) + 1
    summary["angry" if prediction["is_angry"] else "not_angry"] += 1

Path("results").mkdir(exist_ok=True)
with open("results/llm_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)
