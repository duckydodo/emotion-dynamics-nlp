import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class TransformerEmotionClassifier:

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_labels=7,
        device=None,
    ):

        self.model_name = model_name
        self.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.model.to(self.device)

    def predict(self, texts):

        self.model.eval()

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = self.model(**encoded)

        predictions = torch.argmax(
            outputs.logits,
            dim=1,
        )

        return predictions.cpu().numpy()

    def predict_proba(self, texts):

        self.model.eval()

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = self.model(**encoded)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        return probabilities.cpu().numpy()
