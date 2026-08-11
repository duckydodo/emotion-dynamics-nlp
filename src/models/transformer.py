import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_ID.items()
}


class TransformerEmotionClassifier:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        device=None,
    ):

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model.to(self.device)

    def predict(self, texts, batch_size=16, max_length=128):

        self.model.eval()

        texts = list(texts)

        predictions = []

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[
                start:start + batch_size
            ]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.no_grad():

                outputs = self.model(**encoded)

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

        return [
            ID_TO_LABEL[prediction]
            for prediction in predictions
        ]

    def predict_proba(
        self,
        texts,
        batch_size=16,
        max_length=128,
    ):

        self.model.eval()

        texts = list(texts)

        probabilities = []

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[
                start:start + batch_size
            ]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.no_grad():

                outputs = self.model(**encoded)

            batch_probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            probabilities.extend(
                batch_probabilities.cpu().numpy()
            )

        return probabilities

    def save(self, output_dir):

        output_dir = str(output_dir)

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load(self, output_dir):

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                output_dir
            )
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                output_dir
            )
        )

        self.model.to(self.device)
