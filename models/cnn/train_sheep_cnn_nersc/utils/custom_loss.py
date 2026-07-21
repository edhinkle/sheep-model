"""
Custom Loss functions for SHEEP CNN
"""

import torch
import torch.distributed as dist
from torch import Tensor

from torch.nn import _reduction as _Reduction, functional as F

from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.module import Module
from torch.nn.modules.loss import _Loss

# Copied MSELoss from https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/loss.py#L563
# Then, modified to add a `weight` argument to the forward function, which is used to weight the loss 
# for each sample in the batch.
class WeightedMSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each.

    The mean operation still operates over all the elements, and divides by :math:`N`.

    The division by :math:`N` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    Examples:

        >>> loss = nn.MSELoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        # "weight" should be an optional argument for mse_loss(), but is throwing errors
        #weights = torch.ones_like(target)/(target + torch.full_like(target, 1e-8))**2
        #return F.mse_loss(input, target, reduction=self.reduction, weight=weights)
        weights = torch.ones_like(target) / (target + torch.full_like(target, 1e-8))**2
        mse = F.mse_loss(input, target, reduction='none')  # Get unreduced loss
        weighted_mse = weights * mse
    
        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:  # 'none'
            return weighted_mse

# Copied MSELoss from https://github.com/pytorch/pytorch/blob/v2.12.0/torch/nn/modules/loss.py#L563
# Then, modified to add a `weight` argument to the forward function, which is used to weight the loss 
# for each sample in the batch.
class WeightedL1Loss(_Loss):
    r"""Creates a criterion that measures the L1 loss between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`N` elements each.

    The mean operation still operates over all the elements, and divides by :math:`N`.

    The division by :math:`N` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    Examples:

        >>> loss = nn.L1Loss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    __constants__ = ["reduction"]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Runs the forward pass.
        """
        # "weight" should be an optional argument for mse_loss(), but is throwing errors
        #weights = torch.ones_like(target)/(target + torch.full_like(target, 1e-8))
        #return F.mse_loss(input, target, reduction=self.reduction, weight=weights)
        weights = torch.ones_like(target) / (target + torch.full_like(target, 1e-8))
        l1 = F.l1_loss(input, target, reduction='none')  # Get unreduced loss
        weighted_l1 = weights * l1

        if self.reduction == 'mean':
            return weighted_l1.mean()
        elif self.reduction == 'sum':
            return weighted_l1.sum()
        else:  # 'none'
            return weighted_l1

