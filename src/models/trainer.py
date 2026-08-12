import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from src.evaluation.metrics import evaluate
from src.models.transformer import (
    LABELS,
    LABEL_TO_ID,
    ID_TO_LABEL,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmotionDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128,
    ):
        self.texts = list(texts)

        if labels is None:
            self.labels = None
        else:
            self.labels = [
                LABEL_TO_ID[label]
                for label in labels
            ]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(
                self.labels[index],
                dtype=torch.long,
            )

        return item


class TransformerTrainer:

    def __init__(
        self,
        classifier,
        learning_rate=2e-5,
        batch_size=16,
        epochs=3,
        max_length=128,
        seed=42,
    ):
        self.classifier = classifier
        self.model = classifier.model
        self.tokenizer = classifier.tokenizer
        self.device = classifier.device

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_length = max_length
        self.seed = seed

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.history = []

    def _create_loader(
        self,
        texts,
        labels=None,
        shuffle=False,
    ):
        dataset = EmotionDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _run_epoch(self, loader):

        self.model.train()

        total_loss = 0.0

        for batch in loader:

            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }

            self.optimizer.zero_grad()

            outputs = self.model(**batch)

            loss = outputs.loss

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _predict_loader(self, loader):

        self.model.eval()

        all_predictions = []
        all_probabilities = []
        all_labels = []

        with torch.no_grad():

            for batch in loader:

                labels = batch.pop(
                    "labels",
                    None,
                )

                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }

                outputs = self.model(**batch)

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1,
                )

                predictions = torch.argmax(
                    probabilities,
                    dim=1,
                )

                all_predictions.extend(
                    predictions.cpu().tolist()
                )

                all_probabilities.extend(
                    probabilities.cpu().numpy()
                )

                if labels is not None:
                    all_labels.extend(
                        labels.numpy().tolist()
                    )

        return (
            all_labels,
            all_predictions,
            np.array(all_probabilities),
        )

    def validate(
        self,
        texts,
        labels,
    ):

        loader = self._create_loader(
            texts=texts,
            labels=labels,
            shuffle=False,
        )

        y_true, y_pred, probabilities = (
            self._predict_loader(loader)
        )

        y_true = [
            ID_TO_LABEL[label]
            for label in y_true
        ]

        y_pred = [
            ID_TO_LABEL[prediction]
            for prediction in y_pred
        ]

        results = evaluate(
            y_true,
            y_pred,
            labels=LABELS,
        )

        return results, probabilities

    def _save_training_checkpoint(
        self,
        checkpoint_dir,
        epoch,
        best_macro_f1,
    ):

        checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_macro_f1": best_macro_f1,
                "history": self.history,
            },
            checkpoint_dir / "latest_training_state.pt",
        )

    def load_training_checkpoint(
        self,
        checkpoint_dir,
    ):

        checkpoint_dir = Path(checkpoint_dir)

        checkpoint = torch.load(
            checkpoint_dir / "latest_training_state.pt",
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.model.to(self.device)

        self.history = checkpoint["history"]

        return {
            "epoch": checkpoint["epoch"],
            "best_macro_f1": checkpoint["best_macro_f1"],
        }

    def _save_best_model(
        self,
        checkpoint_dir,
        best_macro_f1,
    ):

        best_dir = (
            Path(checkpoint_dir)
            / "best"
        )

        self.classifier.save(
            best_dir
        )

        with open(
            best_dir / "training_config.json",
            "w",
        ) as file:

            json.dump(
                {
                    "model_name": self.classifier.model_name,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "max_length": self.max_length,
                    "seed": self.seed,
                    "best_val_macro_f1": best_macro_f1,
                    "labels": LABELS,
                },
                file,
                indent=2,
            )

        return best_dir

    def fit(
        self,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        checkpoint_dir=None,
        resume=False,
    ):

        set_seed(self.seed)

        train_loader = self._create_loader(
            texts=train_texts,
            labels=train_labels,
            shuffle=True,
        )

        start_epoch = 1
        best_macro_f1 = -1.0

        if resume and checkpoint_dir is not None:

            checkpoint_path = (
                Path(checkpoint_dir)
                / "latest_training_state.pt"
            )

            if checkpoint_path.exists():

                state = self.load_training_checkpoint(
                    checkpoint_dir
                )

                start_epoch = state["epoch"] + 1
                best_macro_f1 = state["best_macro_f1"]

                print(
                    f"Resuming from epoch "
                    f"{state['epoch']}"
                )

        for epoch in range(
            start_epoch,
            self.epochs + 1,
        ):

            train_loss = self._run_epoch(
                train_loader
            )

            results, _ = self.validate(
                val_texts,
                val_labels,
            )

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": results.accuracy,
                "val_macro_f1": results.macro_f1,
                "val_weighted_f1": results.weighted_f1,
            }

            self.history.append(
                epoch_result
            )

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Accuracy: {results.accuracy:.4f} | "
                f"Val Macro F1: {results.macro_f1:.4f}"
            )

            if results.macro_f1 > best_macro_f1:

                best_macro_f1 = results.macro_f1

                if checkpoint_dir is not None:

                    best_dir = self._save_best_model(
                        checkpoint_dir,
                        best_macro_f1,
                    )

                    print(
                        f"Saved best model to "
                        f"{best_dir}"
                    )

            if checkpoint_dir is not None:

                self._save_training_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    best_macro_f1=best_macro_f1,
                )

        return self.history

    def predict(self, texts):

        loader = self._create_loader(
            texts=texts,
            labels=None,
            shuffle=False,
        )

        _, predictions, probabilities = (
            self._predict_loader(loader)
        )

        predicted_labels = [
            ID_TO_LABEL[prediction]
            for prediction in predictions
        ]

        return predicted_labels, probabilities
