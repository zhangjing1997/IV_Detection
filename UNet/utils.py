import sys
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function

## --- logging utils ---
class Logger(object):
    """
    make pring statement simultaneously print onto console and into a logfile.
    """
    def __init__(self, logfile_name):
        self.terminal = sys.stdout
        self.log = open(logfile_name, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility. this handles the flush command by doing nothing. you might want to specify some extra behavior here.
        pass


## --- visualization utils ---
def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


## --- loss computing ---
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

## 类似于IOU的一种metric
def dice_coeff(pred, target):
    """Dice coeff for batch samples"""
    if pred.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, (pred_mask, target_mask) in enumerate(zip(pred, target)):
        s = s + DiceCoeff().forward(pred_mask, target_mask)

    return s / (i + 1)