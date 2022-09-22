import torch


def evaluate(
        model,
        data_loader,
        device='cuda'):
    """
    """
    model.eval()
    model.to(device)

    y_true = []
    y_pred = []
    y_score = []
    n_samples = 0
    loss_total = 0.0
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
        loss_total += loss.detach().cpu().numpy()
        y_true += y.detach().cpu().numpy().tolist()
        y_pred += outputs.argmax(dim=1).detach().cpu().numpy().tolist()
        y_score += outputs.softmax(dim=1)[:, 1].detach().cpu().numpy().tolist()

    return y_true, y_pred, y_score, loss_total / n_samples
