import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


class TransformerEmotionClassifier:
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_labels=7,
        device=None,
        model_path=None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        if model_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label={i: label for i, label in enumerate(LABELS)},
                label2id={label: i for i, label in enumerate(LABELS)},
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts, batch_size=16, max_length=128):
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            ).cpu().numpy()

            predictions.extend(
                LABELS[int(index)]
                for index in batch_predictions
            )

        return predictions

    def predict_proba(self, texts, batch_size=16, max_length=128):
        probabilities = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

            batch_probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            ).cpu().numpy()

            probabilities.append(batch_probabilities)

        return np.concatenate(probabilities, axis=0)

    def save(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, output_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            output_dir
        )

        self.model.to(self.device)
        self.model.eval()
