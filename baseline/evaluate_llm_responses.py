from transformers import pipeline
import torch



emotion_classifier = pipeline("text-classification", model="HR26kk/modernbert-emotion-classifier", top_k = None)

# text = story
text = "I am so happy"
classifier_result = emotion_classifier(text, top_k = None)
print(classifier_result)

# [{'label': 'joy', 'score': 0.9995891451835632}, {'label': 'sadness', 'score': 0.0001925773685798049}, {'label': 'love', 'score': 8.270343823824078e-05}, 
# {'label': 'surprise', 'score': 6.527118239318952e-05}, {'label': 'fear', 'score': 4.070326031069271e-05}, {'label': 'anger', 'score': 2.9601442292914726e-05}]