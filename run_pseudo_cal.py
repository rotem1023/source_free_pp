from loaders.load_data import *
from loaders.load_model import *
from calibration.find_temp import FindTemp
from calibration.calibration_loss import CalibrationLoss

from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def _cal_predictions(model, loader):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)  # Create an iterator
        for _ in range(len(loader)):
            data = next(iter_test)  # Use Python's built-in next()
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_inputs = inputs.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_inputs = torch.cat((all_inputs, inputs.cpu()), 0)
    _, predict = torch.max(all_output, 1)
    all_output_numpy = all_output.numpy()
    predict_numpy = predict.numpy()

    # X, logits, prediotions, labels
    return all_inputs, all_output_numpy, predict_numpy, all_label


def pseudo_target_synthesis (x , lam , pred_a ):
    # Random batch index .
    rand_idx = torch . randperm ( x . shape [0])
    inputs_a = x
    inputs_b = x [ rand_idx ]
    # Obtain model predictions and pseudo labels (pl ).
    pred_a = pred_a
    pl_a = pred_a
    pl_b = pl_a [ rand_idx ]
    # Select the samples with distinct labels for the mixup .
    diff_idx = ( pl_a != pl_b ). nonzero ()
    # Mixup with images and labels .
    pseudo_inputs = lam * inputs_a + (1 - lam ) * inputs_b
    if lam > 0.5:
        pseudo_labels = pl_a
    else :
        pseudo_labels = pl_b
    return pseudo_inputs [ diff_idx ] , pseudo_labels [ diff_idx ]


def calc_optimal_temp( pseudo_preds , pseudo_labels):
    # Apply temperature scaling to estimate the
    # pseudo - target temperature as the real temperature .
    calib_method = FindTemp ()
    pseudo_temp = calib_method ( pseudo_preds , pseudo_labels )
    return pseudo_temp

def run_pseudo_cal(model, loader, lam = 0.65):
    x, logits, predictions, labels  = _cal_predictions(model = model, loader = loader)

    pseudo_x , pseudo_y = pseudo_target_synthesis(x, lam, predictions)
    batch_size = 64  # Set your desired batch size
    dataset = TensorDataset(pseudo_x, torch.tensor(pseudo_y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate through the DataLoader to process the data in batches
    all_pseudo_pred = []
    for batch_idx, (inputs_batch, labels_batch) in enumerate(data_loader):
        # Move inputs_batch and labels_batch to GPU if needed
        inputs_batch, labels_batch = inputs_batch.cuda(), labels_batch.cuda()

        # Compute predictions for the current batch
        pseudo_pred_batch = model(inputs_batch)

        # Store the predictions (you can append them or process them here)
        all_pseudo_pred.append(pseudo_pred_batch.detach().cpu().numpy())  # Move to CPU for saving

    # Concatenate all predictions
    all_pseudo_pred = np.concatenate(all_pseudo_pred, axis=0)
    
    T = calc_optimal_temp(pseudo_preds=all_pseudo_pred, pseudo_labels = pseudo_y)
    return T, logits, labels

def run_pseudo_cal_with_loss(model, loader, n_bins, adaEce, lam = 0.65):
    T , logits, labels = run_pseudo_cal(model = model, loader = loader, lam = lam)



if __name__ == "__main__":
    train_loader, validation_loader = load_data(dataset="VISDA", domain="SR", year='2019')
    model = load_model(dataset="VISDA", model_name="shot", src_domain='S', tgt_domain='R', year='2019')