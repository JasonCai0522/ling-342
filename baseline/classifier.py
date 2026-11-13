
from enum import StrEnum
import json
from transformers import pipeline
import torch
from typing import TypedDict
import pandas as pd

class Emotion(StrEnum):
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"


class ClassifierResult(TypedDict):
    false_positives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    example_count: int = 0
    score: float = 0
    false_positives_examples: list[dict] = []
    true_positives_examples: list[dict] = []
    true_negatives_examples: list[dict] = []
    false_negatives_examples: list[dict] = []

# Takes in a text and a target emotion, then updates classifier_result based
# on the result

# Score is calculated as just the score from the classifier if the target emotion is 
# anger, else it is 1 - anger score.

# To classify if a score is angry for TP/FP/TN/FN, we set a threshold (defaults to 75), and use that 
# 
def run_classifier( classifier, 
                    text: str,
                    true_emotion: Emotion,
                    positive: bool,
                    classifier_results: ClassifierResult,
                    target_emotion: Emotion = Emotion.ANGER, 
                    threshold: float = 0.75) -> ClassifierResult:

    classifier_result = classifier(text,
                                    top_k = None)


    print(classifier_result)
    filtered_result = list(filter(lambda x:
            x["label"] == "anger", classifier_result))

    print(filtered_result)
    score = filtered_result[0]["score"]
    if (positive):
        if (score >= threshold):
            classifier_results["true_positives"] += 1
            classifier_results["true_positives_examples"].append({
                "scores": classifier_result,
                "true_emotion": true_emotion,
                "text": text
            })
        else:
            classifier_results["false_negatives"] += 1
            classifier_results["false_negatives_examples"].append({
                "scores": classifier_result,
                "true_emotion": true_emotion,
                "text": text
            })
    else: # Negative
        if (score < threshold):
            classifier_results["true_negatives"] += 1
            classifier_results["true_negatives_examples"].append({
                "scores": classifier_result,
                "true_emotion": true_emotion,
                "text": text
            })
        else:
            classifier_results["false_positives"] += 1
            classifier_results["false_positives_examples"].append({
                "scores": classifier_result,
                "true_emotion": true_emotion,
                "text": text
            })

    classifier_results["score"] += (score) if positive else (1 - score)
    classifier_results["example_count"] += 1

    return classifier_results
    

def load_training_data_from_csv(name: str):
    return pd.read_csv(f"./dataset/dataset/df1.csv")


emotion_classifier = pipeline("text-classification", 
                      model="HR26kk/modernbert-emotion-classifier",
                      top_k = None)


df = load_training_data_from_csv("training")
print(df)

classifier_results = ClassifierResult(false_positives = 0,
    true_positives = 0,
    true_negatives = 0,
    false_negatives = 0,
    example_count = 0,
    score = 0,
    false_positives_examples = [],
    true_positives_examples = [],
    true_negatives_examples = [],
    false_negatives_examples = [])

for row in df.itertuples():

    print(row.emotion)
    classifier_results = run_classifier(emotion_classifier, 
                                        row.text, row.emotion, 
                                        (row.emotion == "angry"), 
                                        classifier_results)
                                        


print(classifier_results)
with open("./results/classifier_results.json", "w", encoding="utf-8") as f:
  json.dump(classifier_results, f, indent=4)


