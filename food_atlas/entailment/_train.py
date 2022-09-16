import warnings

import torch
from tqdm import tqdm
# import wandb

from . import get_all_metrics, evaluate

# def evaluate_nli_model(
#         model,
#         data_loader,
#         device='cuda'):
#     """
#     """
#     model.eval()
#     model.to(device)

#     n_samples = 0
#     loss_total = 0.0
#     tn_total = 0
#     fp_total = 0
#     fn_total = 0
#     tp_total = 0
#     for (input_ids, attention_mask, token_type_ids), y in data_loader:
#         input_ids = input_ids.to(device)
#         attention_mask = attention_mask.to(device)
#         token_type_ids = token_type_ids.to(device)
#         y = y.to(device)

#         with torch.no_grad():
#             loss, outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 labels=y
#             ).values()

#         n_samples += len(y)
#         loss_total += loss.detach().cpu().numpy()
#         y_true = y.detach().cpu().numpy()
#         y_pred = outputs.argmax(dim=1).detach().cpu().numpy()
#         y_score = outputs.softmax(dim=1)[:, 1].detach().cpu().numpy().tolist()

#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             metrics = get_all_metrics(y_true, y_pred)
#         tn_total += metrics['tn']
#         fp_total += metrics['fp']
#         fn_total += metrics['fn']
#         tp_total += metrics['tp']

#     result = {
#         'tn': tn_total,
#         'fp': fp_total,
#         'fn': fn_total,
#         'tp': tp_total,
#         'loss': loss_total / n_samples
#     }

#     return result


def train(
        model,
        data_loader_train,
        data_loader_val=None,
        epochs=1,
        lr=1e-3,
        # verbose_step_size=20,
        device='cuda'):
    """
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    result = {}
    for epoch in range(epochs):
        model.train()
        n_samples = 0
        loss_total = 0.0
        # tn_total = 0
        # fp_total = 0
        # fn_total = 0
        # tp_total = 0

        pbar = tqdm((data_loader_train), position=0, leave=True)
        for i_step, ((input_ids, attention_mask, token_type_ids), y) \
                in enumerate(pbar):
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
            loss_total += loss.detach().cpu().numpy()
            y_true = y.detach().cpu().numpy()
            y_pred = outputs.argmax(dim=1).detach().cpu().numpy()
            y_score = outputs.softmax(
                dim=1)[:, 1].detach().cpu().numpy().tolist()

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                metrics = get_all_metrics(y_true, y_pred, y_score)

            # # Evaluate for each verbose step size.
            # tn_total += metrics['tn']
            # fp_total += metrics['fp']
            # fn_total += metrics['fn']
            # tp_total += metrics['tp']
            # if i_step % verbose_step_size == 0 and i_step > 0:
            #     wandb.log(
            #         {
            #             'train_prec': tp_total / (tp_total + fp_total),
            #             'train_recall': tp_total / (tp_total + fn_total),
            #             'train_f1': 2 * tp_total /
            #             (2 * tp_total + fp_total + fn_total),
            #             'train_accuracy': (tp_total + tn_total) /
            #             (tp_total + tn_total + fp_total + fn_total),
            #             'training_loss': round(loss_total / n_samples, 3),
            #         },
            #     )

        # result_epoch_train = {
        #     'tn': tn_total,
        #     'fp': fp_total,
        #     'fn': fn_total,
        #     'tp': tp_total,
        #     'loss': loss_total / n_samples
        # }
        result[f'train_iter_{epoch}'] = {
            'metrics': metrics,
            'loss': loss_total / n_samples,
        }

    if data_loader_val is not None:
        y_true, y_pred, y_score, loss = evaluate(
            model, data_loader_val, device)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            metrics = get_all_metrics(y_true, y_pred, y_score)

        result['val'] = {
            'metrics': metrics,
            'loss': loss,
        }

    return result
