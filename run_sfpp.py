from loaders.load_data import *
from loaders.load_model import *
from loaders.load_num_classes import *
import numpy as np
from backpack import backpack
from backpack.extensions import BatchGrad
import math



def batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.

    Taken from the official torch implementation:
    https://github.com/pytorch/pytorch/blob/main/torch/distributions/multivariate_normal.py
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = (
        torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)

def log_likelihood_multivar(x, mean, cov, num_classes, L2_regularizer):
    # mu = µ.expand(x.shape + (-1,))

    mahal = batch_mahalanobis(cov, x - mean)
    if L2_regularizer:
         mahal = mahal / (torch.norm(cov, 2))
    # Variance: my_var = cov.pow(2).sum(-1)
    # This value is a constant
    k_ln_2pi = num_classes * math.log(2 * math.pi)


    # ln|Sigma|
    log_det = 2 * cov.diagonal(dim1=-2, dim2=-1).log().sum()
    final_log_prob = log_det - 0.5 * mahal + k_ln_2pi  # )

    final_log_prob = final_log_prob
    if L2_regularizer:
        final_log_prob = final_log_prob / torch.norm(cov, 2)
    return final_log_prob


def log_prob_logsumexp_trick(gmm_stats, logits, num_classes, L2_regularizer=False):
    log_probs, mahals = [], []
    cov, mean_per_class, log_probs_means = gmm_stats["cov"], gmm_stats["mean_per_class"], gmm_stats["log_probs_means"]
    cov = torch.linalg.cholesky(cov)
    for cls in range(num_classes):
        if cls in mean_per_class.keys():
            log_probs.append(log_likelihood_multivar(logits, mean_per_class[cls], cov, num_classes,
                                                     L2_regularizer=L2_regularizer))

    log_probs = torch.stack(log_probs, dim=0)

    log_probs = log_probs - torch.logsumexp(log_probs_means, dim=1).unsqueeze(1)
    log_probs = log_probs - torch.logsumexp(log_probs, dim=0).unsqueeze(0)

    log_exp = torch.exp(log_probs)
    total_probs = torch.sum(log_exp, dim=0)
    final_probabilities = (log_exp / total_probs).T

    try:
        assert torch.allclose(torch.sum(final_probabilities, dim=1),
                              torch.ones(final_probabilities.shape[0], device=final_probabilities.device),
                          rtol=1e-4)
    except AssertionError:
        print("PROBABILITIES DO NOT SUM TO 1")
        print("Logits", logits)
        print("Final probabilities", final_probabilities)
        raise AssertionError
    return final_probabilities

def get_gmm_probab(classifier, features, num_classes, gmm_stats):
    """
    Forward pass through the classifier and get the probability using the GMM statistics
    Args:
        classifier: The last classifier layer of the network
        features: The features from the feature extractor
        num_classes:
        gmm_stats: dictionary containing the mean, covariance, mins, maxs, count_per_class of the target features
        per_class_logs:

    Returns: The probability of the features belonging to each class, the logits, and the logits without temperature

    """
    classifier.zero_grad()
    logits_without_temp = classifier(features)
    logits = copy.deepcopy(logits_without_temp.detach())
    logits_without_temp = (logits_without_temp - gmm_stats['mins']) / (gmm_stats['maxs'] - gmm_stats['mins'])
    prob = log_prob_logsumexp_trick(gmm_stats, logits_without_temp, num_classes,
                                        L2_regularizer=False)
    return prob, logits

def get_per_class_mean_and_cov(dataloader, classifier, num_classes, device="cuda:0"):

    """
    Pass all samples through the network and store all the logits
    Calculate the mean and covariance matrix for each class
    Based on these means and covariances - calculate the multivariate normal distributions for each class

    Args:
        dataloader: Target loader that iterates through the target features
        classifier: Last fully connected layer of the network
        num_classes:
        device:

    Returns: Dictionary with mean_per_class, covariance
    """


    logits_per_class = {cls: [] for cls in range(num_classes)}

    all_logits = []
    for batch_number, data in enumerate(dataloader):
        features, lbl = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            logits_without_temp = classifier(features)
            all_logits.append(logits_without_temp.detach().cpu().numpy())
            predicted_classes = torch.argmax(logits_without_temp, dim=1)

            # Store the logits for each class according to the predicted class
            for i, pred_class in enumerate(predicted_classes):
                logits_per_class[pred_class.item()].append(logits_without_temp[i])  # .detach().cpu().numpy())

    gmm_stats = compute_gmm_stats(logits_per_class, num_classes=num_classes)

    return gmm_stats

def get_grad_norm_pytorch(classifier, predicted_prob, target_prob):
    loss = custom_ce_loss(predicted_prob, target_prob)
    loss_summed = torch.sum(loss)
    with backpack(BatchGrad()):
        loss_summed.backward(retain_graph=True)
    # modified from: classifier.weight.grad_batch
    batch_grad = classifier.fc.weight.grad_batch
    grad_reshaped = batch_grad.reshape((batch_grad.shape[0], -1))
    grad_norm = torch.norm(grad_reshaped, dim=1)
    return grad_norm  #  newton_grad_norm


def custom_ce_loss(predicted_prob, target_prob):
    loss = -(target_prob*torch.log(predicted_prob+1e-8))  # /N
    return loss


def cov_pytorch(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def get_per_class_log_probs(mean_per_class, cov, num_classes, L2_regularizer=True):
    """
    Compute the log probabilities of the means of each class

    Args:
        mean_per_class:
        cov:
        num_classes:
        L2_regularizer:

    Returns:

    """
    cov = torch.linalg.cholesky(cov)
    log_probs_means = {}
    for cls in range(num_classes):
        if mean_per_class[cls] is not None:
            log_probs_means[cls] = []
            for cls1 in range(num_classes):
                if cls1 in mean_per_class.keys():
                    if cls != cls1:
                        if mean_per_class[cls1] is not None:
                            log_probs_means[cls].append(log_likelihood_multivar(mean_per_class[cls1],
                                                                    mean_per_class[cls],
                                                                    cov, num_classes,
                                                                    L2_regularizer=L2_regularizer))
            log_probs_means[cls] = torch.stack(log_probs_means[cls], dim=0)
    log_probs_means = torch.stack(list(log_probs_means.values()), dim=0)
    return log_probs_means

def compute_gmm_stats(logits_per_class, num_classes, device="cuda:0"):

    """
    Compute the mean and covariance matrix for each class

    Args:
        logits_per_class: dictionary with keys as class labels and values as lists of logits
        num_classes:
        device:

    Returns: Dictionary with mean_per_class, covariance

    """

    gmm_stats = {}
    mean_per_class, cov_per_class, count_per_class = {}, {}, {}

    # Combine all the logits from the dictionary into one array

    all_logits = [logit for i in list(logits_per_class.keys()) for logit in logits_per_class[i]]
    all_logits = torch.stack(all_logits)

    # Normalize all logits and calculate the covariance matrix
    all_maxs, all_mins = torch.max(all_logits, dim=0)[0], torch.min(all_logits, dim=0)[0]
    all_logits = (all_logits - all_mins) / (all_maxs - all_mins)

    # mean of all logits
    cov_matr = cov_pytorch(all_logits, rowvar=False, bias=False)

    # Add small value to the diagonal to avoid division by zero
    cov_matr = cov_matr + torch.eye(cov_matr.shape[0], device=device) * 1e-5


    for cls in range(num_classes):
        if cls in logits_per_class.keys():
            count_per_class[cls] = len(logits_per_class[cls])
            if len(logits_per_class[cls]) > 5:
                logits_per_class[cls] = torch.stack(logits_per_class[cls])
                logits_per_class[cls] = (logits_per_class[cls] - all_mins) / (all_maxs - all_mins)
                mean_per_class[cls] = torch.mean(logits_per_class[cls], dim=0)
            else:
                # make a covariance matrix of zeros
                mean_per_class[cls] = torch.zeros(all_logits.shape[1], device=device)
                # mean_per_class[cls] = torch.mean(all_logits, dim=0)
        else:
            # make a covariance matrix of zeros
            mean_per_class[cls] = torch.zeros(all_logits.shape[1], device=device)


    log_probs_means = get_per_class_log_probs(mean_per_class, cov_matr, num_classes, L2_regularizer=False)

    gmm_stats["mean_per_class"] = mean_per_class
    gmm_stats["cov"] = cov_matr
    gmm_stats["maxs"] = all_maxs
    gmm_stats["mins"] = all_mins

    gmm_stats["log_probs_means"] = log_probs_means

    return gmm_stats


def load_features(dataloader, feature_extractor, device):
    """
    Extract features from the dataloader using the feature_extractor and return them as a TensorDataset.

    :param dataloader: iterable dataset
    :param feature_extractor: the network without the last layer
    :param device: Device to use ('cuda' or 'cpu')
    :return: TensorDataset with the features and labels
    """
    feature_extractor.eval()
    feature_extractor.to(device)

    all_features, all_labels = [], []

    # Extract features for all batches
    for data in dataloader:
        im, lbl = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            features = feature_extractor(im)
            all_features.append(features.cpu())  # Keep on CPU to avoid memory issues
            all_labels.append(lbl.cpu())

    # Concatenate all features and labels into single tensors
    all_features = torch.cat(all_features, dim=0)  # Combine along the batch dimension
    all_labels = torch.cat(all_labels, dim=0)

    # Create a TensorDataset
    dset = torch.utils.data.TensorDataset(all_features, all_labels)

    return dset

def predict_accuracy(classifier, feature_loader, num_classes=10, device="cuda:0"):
    """
    Predict the accuracy of the classifier using the method described in the paper
    Consists of 3 steps:
    1. Calculate the GMM statistics by calculating the mean and covariance of the logits for each class
    2. Calculate the probability of the features belonging to each class using the GMM statistics
    3. Make a prediction for each sample based on the gradient norm to the target class and the random class.
        If the gradient norm to the predicted class is higher than the gradient norm to the random class, the sample is predicted incorrectly
        If the gradient norm to the random class is higher than the gradient norm to the target class, the sample is predicted correctly
    Args:
        classifier: The last layer (Fully connected layer) of the model
        feature_loader: DataLoader iterating through the features
        num_classes: Number of classes in the dataset
        device: Device to run the computations on

    Returns: Ground truth accuracy, Predicted Accuracy, Mean Absolute Error between the two

    """

    classifier.eval()
    classifier.to(device)

    # Step 1: Calculate the GMM statistics
    gmm_stats = get_per_class_mean_and_cov(feature_loader, classifier, num_classes=num_classes)

    res = {"labels": [], "pred": [],  "weight2rand": [], "weight2target": []}

    for data in feature_loader:
        features, lbl = data[0].to(device), data[1].to(device)

        # Step 2: Adapt the probabilities based on the GMM statistics
        prob, logits = get_gmm_probab(classifier, features, num_classes, gmm_stats)
        pred_class = torch.argmax(logits, dim=1).detach().cpu()

        # Store the gradients
        uniform_dist = torch.ones_like(prob) / num_classes
        grad_weight_to_rand = get_grad_norm_pytorch(classifier, prob, uniform_dist)

        prob, logits = get_gmm_probab(classifier, features, num_classes, gmm_stats)
        pseudo_labels = torch.argmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(pseudo_labels, num_classes=num_classes)
        # Log one_hot
        grad_weight_to_target = get_grad_norm_pytorch(classifier, prob, one_hot)

        # ========   Logs   =========
        # indx - index of a sample in a batch. Log each sample separately
        for indx, label in enumerate(lbl):
            res["labels"].append(label.item())
            res["pred"].append(pred_class[indx].item())
            res["weight2rand"].append(grad_weight_to_rand[indx].item())
            res["weight2target"].append(grad_weight_to_target[indx].item())

    # Ground Truth Accuracy
    gt_acc = sum(np.array(res["labels"]) == np.array(res["pred"])) / len(res["labels"])

    # Step 3: Predict the performance:
    #   The sample is predicted to be correctly classified
    #   if the gradient norm to the random class is higher than the gradient norm to the predicted class
    predicted_accuracy = sum(np.array(res["weight2rand"]) >= np.array(res["weight2target"])) / len(res["labels"])

    # MAE: Mean Absolute Error - the difference between the ground truth accuracy and the predicted accuracy
    mae = abs(gt_acc - predicted_accuracy)

    return gt_acc, predicted_accuracy, mae

def run_sffp(model, loader, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_set = load_features(dataloader=loader, feature_extractor=model.feature_extractor, device=device)

    # Create a DataLoader to iterate through the features
    feature_loader = DataLoader(feature_set, batch_size=32, shuffle=False, num_workers=0)
    
    # Predict the accuracy using our method
    gt_acc, predicted_acc, mae = predict_accuracy(model.classifier, feature_loader,
                                                      num_classes=num_classes, device=device)
    return gt_acc, predicted_acc, mae
    


if __name__ == "__main__":
    dataset = 'VISDA'
    year = '2019'
    train_loader, validation_loader = load_data(dataset=dataset, domain="SR", year=year)
    model = SfppModel(dataset=dataset, model_name="shot", src_domain='S', tgt_domain='R', year=year)
    
    gt_acc, predicted_acc, mae = run_sffp(model=model, loader=validation_loader, num_classes=get_num_classes(dataset=dataset))
    print(f"Dataset: {dataset} , year: {year}\n"
          f"GT Accuracy: {round(gt_acc* 100, 2)} "
          f"Predicted Accuracy: {round(predicted_acc*100, 2)} "
          f"MAE: {round(mae*100, 2)} \n")