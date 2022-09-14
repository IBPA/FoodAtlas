import torch
# import wandb


def evaluate_nli_model(
        model,
        data_loader,
        device='cuda'):
    """
    """
    model.eval()
    model.to(device)
    n_samples = 0
    loss_total = 0.0
    acc_total = 0

    for (input_ids, attention_mask, token_type_ids), y in data_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        y = y.to(device)

        with torch.no_grad():
            loss, outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=y
            ).values()

        n_samples += len(y)
        y_pred = outputs.argmax(dim=1).detach().cpu().numpy()
        acc_total += (y_pred == y.detach().cpu().numpy()).sum()
        loss_total += loss.detach().cpu().numpy()

    print(
        f"train loss - {round(loss_total / n_samples, 3)}, "
        f"train acc - {round(acc_total / n_samples, 3)}"
    )


def train_nli_model(
        model,
        data_loader_train,
        data_loader_val=None,
        epochs=1,
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

        if data_loader_val is not None:
            evaluate_nli_model(model, data_loader_val, device)
