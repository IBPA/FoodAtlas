# -*- coding: utf-8 -*-
"""FoodAtlas NLI model loading module.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * TODOs

"""
import warnings

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging,
)
from tqdm import tqdm

from .utils import get_all_metrics

logging.set_verbosity_error()


class FoodAtlasEntailmentModel:

    def __init__(
            self,
            path_or_name,
            path_model_state=None,
            device='cuda',
            # verbose=True
            ):
        self.path_or_name = path_or_name
        self.path_model_state = path_model_state
        self.device = device

        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        """Load model from path or name.

        Returns:
            model (
                transformers.modeling_auto.AutoModelForSequenceClassification
                ): The model.
            tokenizer (transformers.tokenization_auto.AutoTokenizer): The
                tokenizer.

        """
        if self.path_or_name == 'biobert':
            model = AutoModelForSequenceClassification.from_pretrained(
                'dmis-lab/biobert-v1.1', num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            tokenizer._pad_token_type_id = 1
        else:
            raise NotImplementedError(
                f"Model {self.path_or_name} not implemented."
            )

        if self.path_model_state is not None:
            model.load_state_dict(torch.load(self.path_model_state))

        return model, tokenizer

    def save_model(self, path_model_state):
        """Save the model state.

        Args:
            path_model_state (str): Path to the model state.

        """
        torch.save(self.model.state_dict(), path_model_state)

    def train(
            self,
            data_loader_train: torch.utils.data.DataLoader,
            epochs: int = 1,
            lr: float = 1e-3) -> dict:
        """Train the model.

        Args:
            data_loader_train (torch.utils.data.DataLoader): The data loader.
            epochs (int): The number of epochs.
            lr (float): The learning rate.

        Returns:
            train_stats (dict): The training statistics.

        """
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        train_stats = {}
        for epoch in range(epochs):
            self.model.train()
            y_true = []
            y_pred = []
            y_score = []
            n_samples = 0
            loss_total = 0.0
            pbar = tqdm((data_loader_train), position=0, leave=True)
            for (input_ids, attention_mask, token_type_ids), y in pbar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                loss, output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=y
                ).values()
                loss.backward()
                optimizer.step()

                n_samples += len(y)
                loss_total += loss.detach().cpu().numpy()
                y_true += y.detach().cpu().numpy().tolist()
                y_pred += output.argmax(dim=1).detach().cpu().numpy().tolist()
                y_score += output.softmax(dim=1)[
                    :, 1].detach().cpu().numpy().tolist()

                # torch.cuda.empty_cache()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                metrics = get_all_metrics(y_true, y_pred)

            train_stats[f'train_iter_{epoch}'] = {
                'metrics': metrics,
                'loss': loss_total / n_samples,
            }

        return train_stats

    def evaluate(
            self,
            data_loader: torch.utils.data.DataLoader
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Evaluate the model.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader.

        Returns:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
            y_score (np.ndarray): The predicted scores.
            loss (float): The average loss.

        """
        self.model.eval()
        self.model.to(self.device)

        y_true = []
        y_pred = []
        y_score = []
        n_samples = 0
        loss_total = 0.0
        for (input_ids, attention_mask, token_type_ids), y in data_loader:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                loss, output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=y
                ).values()

            n_samples += len(y)
            loss_total += loss.detach().cpu().numpy()
            y_true += y.detach().cpu().numpy().tolist()
            y_pred += output.argmax(dim=1).detach().cpu().numpy().tolist()
            y_score += output.softmax(dim=1)[
                :, 1].detach().cpu().numpy().tolist()
        loss = loss_total / n_samples

        return y_true, y_pred, y_score, loss

    def predict(
            self,
            data_loader: torch.utils.data.DataLoader
            ) -> list[float]:
        """Predict the inputs.

        Args:
            data_loader (torch.utils.data.DataLoader): The data loader without
                labels.

        Returns:
            y_score (list[float]): The predicted scores.

        """
        self.model.eval()
        self.model.to(self.device)

        y_score = []
        pbar = tqdm((data_loader), position=0, leave=True)
        for (input_ids, attention_mask, token_type_ids), _ in pbar:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits

            y_score += output.softmax(dim=1)[
                :, 1].detach().cpu().numpy().tolist()

        return y_score
