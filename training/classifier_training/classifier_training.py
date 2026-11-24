from huggingface_hub import notebook_login
from datasets import load_dataset, Features, Value
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          DataCollatorWithPadding, 
                          TrainingArguments, 
                          Trainer)
import numpy as np

notebook_login()

training_df = load_dataset("csv", data_files= {"train": "./dataset/dataset/trainingLong.csv",
                                               "test": "./dataset/dataset/testingLong.csv"})

id2label = {
    0: "not_angry",
    1: "angry"
}

label2id = {
    "not_angry": 0,
    "angry": 1
}

training_df = training_df.map(
    lambda example: {"labels": int(example["labels"] == "angry")},
    features=Features({
        **training_df["train"].features,
        "labels": Value("int64"),
    }),
)

# Training example
# training_df[0]
# {
#     "label": "not_angry",
#     "text": "Mara stood behind the orange demolition fence with her sketchbook pressed against her ribs. The house looked smaller than she remembered, its porch sagging beneath a tangle of ivy, but every window still held a piece of her: the kitchen where she learned to crack eggs, the upstairs room where she painted stars across the ceiling, the hallway where her father measured her height in pencil marks. She opened to a clean page and began drawing. Around her, machines groaned and workers shouted instructions. Mara’s hand moved faster. By the time the first wall came down, she had filled the paper with doors, windows, and tiny figures carrying one another safely outside. “This isn’t the end of it,” she said, tucking the sketch into her coat. That evening, she called the neighborhood council and offered to design a community mural from the house’s surviving details."
# happy,"At thirty-seven, Thomas sent his saliva sample to a genealogy company on a whim. Three weeks later, the results revealed that the family tree he had drawn as a child belonged to someone else.
# }

tokenizer = AutoTokenizer.from_pretrained("HR26kk/modernbert-emotion-classifier")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_training_df = training_df.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Skipped Evaluate Section on Hugging Face

# TODO: See if we need this later
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {}


# Check if we need id2label stuff
model = AutoModelForSequenceClassification.from_pretrained(
    "HR26kk/modernbert-emotion-classifier",
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    output_dir="./trained_model/trained_classification_model_long", 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub= False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_df["train"],
    eval_dataset=tokenized_training_df["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics, # so like i deleted the function cuz it was part of evaluate
)

trainer.train()
trainer.save_model()