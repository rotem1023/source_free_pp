import torch
from torch import nn
from calibration.calibration_loss import CalibrationLoss
from scipy import optimize


ranges = (slice(1, 10, 0.05),)


class FindTemp(nn.Module):
    '''
    Find the best temperature T
    '''

    def __init__(self, n_bins=15, LOGIT=True, adaECE=False):
        super(FindTemp, self).__init__()
        self.adaECE = adaECE
        self.n_bins = n_bins
        self.LOGIT = LOGIT

    def find_best_T(self, logits, labels):
        '''
        Find the best temperature T for the given logits and labels
        :param logits: logits of the model
        :param labels: pseudo labels generated in the first stage of the algorithm
        :return: the best temperature T
        '''
        ece_loss = CalibrationLoss(adaECE=self.adaECE, n_bins=self.n_bins, LOGIT=self.LOGIT)

        def eval(x):
            "x ==> temperature T"
            x = torch.from_numpy(x)
            if (x < 0):
                return 1
            scaled_logits = logits.float() / x
            # calculate the ECE\adaECE loss with the given temperature
            return ece_loss.forward(scaled_logits, labels)

        return self._calc_optimal_T(eval)

    def _calc_optimal_T(self, eval_func):
        '''
        calculate the calibration loss for different temperatures and return the best one
        :param eval_func: function that calculate the calibration loss
        :return:
        '''
        global ranges
        optimal_parameter = optimize.brute(eval_func, ranges, finish=optimize.fmin)
        return optimal_parameter[0]