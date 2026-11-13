import torch

from transformers import pipeline
import pandas as pd




def run_classifier(classifier, text, target_emotion, true_emotion):
    classifier_result = classifier("text")
    print(classifier_result)
    filtered_result = filter(lambda x:
            x.label == target_emotion, 
        classifier_result)
    
    

    return (filtered_result[0].score)
    

    
    
    


def load_training_data_from_csv():
    return 


emotion_classifier = pipeline("sentiment-analysis", 
                      model="michellejieli/emotion_text_classifier",
                      top_k = None)


run_classifier(emotion_classifier, "I am happy", "angry")






