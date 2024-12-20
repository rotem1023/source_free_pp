import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class CalibrationLoss(nn.Module):
    '''
    Calculate the Expected Calibration Error (ECE or AdaECE) of a model.
    '''
    def __init__(self, n_bins=15, LOGIT=True, adaECE=False):
        super(CalibrationLoss, self).__init__()
        self.nbins = n_bins
        self.LOGIT = LOGIT
        self.adaECE = adaECE

    def forward(self, logits, labels):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        bin_lowers, bin_uppers = self._claculate_bin_boundaries(confidences)
        ece = torch.zeros(1, device=logits.device)

        # Calculated ECE\adaECE in each bin
        for i in range(len(bin_lowers)):
            bin_lower = bin_lowers[i]
            bin_upper = bin_uppers[i]

            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            # probability for example to be in the bin
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0 and (self.adaECE or in_bin.sum() > 20):
                # accuracy in the bin
                accuracy_in_bin = self._calculate_accuracy_in_bin(in_bin, correctness)
                # average confidence in the bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # ECE\adaECE in the bin
                cur_ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                ece += cur_ece
        return ece



    def _claculate_bin_boundaries(self, confidences):
        '''
        Calculate the bin boundaries for the ECE\adaECE calculation
        :param confidences: array of confidences
        :return: two arrays of bin boundaries (lower and upper)
        '''
        if self.adaECE:
            n, bin_boundaries = np.histogram(confidences.cpu().detach(),
                                             self._histedges_equalN(confidences.cpu().detach()))
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        else:
            bin_boundaries = torch.linspace(0, 1, self.nbins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
        return bin_lowers, bin_uppers

    def _histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                         np.arange(npt),
                         np.sort(x))

    def _calculate_accuracy_in_bin(self, in_bin, correctness):
        accuracy_in_bin = correctness[in_bin].float().mean()
        return accuracy_in_bin