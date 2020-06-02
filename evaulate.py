import torch
import numpy as np
import IPython

def evaluate_rnn(model, criterion, optimizer, data_loader, device):
    model.eval()
    data_iter = iter(data_loader)
    num_batches = len(data_iter)
    batch_index = 0
    preds = []; ground_truth = []; loss_avg = []
    with torch.no_grad():
        while batch_index < num_batches:
            data, labels = data_iter.next()
            data = data.to(device, non_blocking=True)
            labels = labels.cuda(device, non_blocking=True)
            
            data = torch.transpose(data, 0, 1)
#             labels = labels.squeeze() # uncomment for bert, comment for glove
            
            logits = model(data)
            preds_batch = torch.sigmoid(logits)
            preds_batch = torch.where(
                preds_batch >= 0.5, 
                torch.tensor(1.0).to(device),
                torch.tensor(0.0).to(device)
            )
            
            preds.extend(preds_batch.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
            
            loss = criterion(logits, labels)
            loss_avg.append(loss.item())
            batch_index += 1

    return np.array(preds), np.array(ground_truth), loss_avg


def eval_perf_binary(Y, Y_):
    tp = sum(np.logical_and(Y==Y_, Y_==True))
    fn = sum(np.logical_and(Y!=Y_, Y_==True))
    tn = sum(np.logical_and(Y==Y_, Y_==False))
    fp = sum(np.logical_and(Y!=Y_, Y_==False))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp+fn + tn+fp)
    f1 = (2 * precision * recall) / (precision + recall)
    return accuracy, recall, precision, f1