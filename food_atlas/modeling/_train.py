import torch
import wandb


def train_nli_model(
        model,
        data_loader_train,
        data_loader_val=None,
        epochs=100,
        lr=1e-3,
        device='cuda'):
    """
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        n_samples = 0
        loss_total = 0.0
        acc_total = 0

        for (input_ids, attention_mask, token_type_ids), y \
                in data_loader_train:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss, outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=y
            ).values()
            loss.backward()
            optimizer.step()

            n_samples += len(y)
            y_pred = outputs.argmax(dim=1).detach().cpu().numpy()
            acc_total += (y_pred == y.detach().cpu().numpy()).sum()
            loss_total += loss.detach().cpu().numpy()

        print(
            f"Iter {epoch + 1}: train loss - "
            f"{round(loss_total / n_samples, 3)}, "
            f"train acc - {round(acc_total / n_samples, 3)}"
        )


if __name__ == '__main__':
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import pandas as pd

    from . import (
        get_food_atlas_data_loader
    )

    torch.manual_seed(0)

    data = pd.read_csv("tests/data/small.csv", sep="\t")
    data = data[['premise', 'hypothesis_string', 'label']]
    data = data.rename({'hypothesis_string': 'hypothesis'}, axis=1)

    model = AutoModelForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-v1.1', num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    tokenizer._pad_token_type_id = 1

    data_loader = get_food_atlas_data_loader(
        data=data,
        tokenizer=tokenizer,
        max_seq_len=100,
        batch_size=4,
    )

    train_nli_model(
        model=model,
        data_loader_train=data_loader,
    )
