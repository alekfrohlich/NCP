"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019
Ref Code: https://github.com/huangleiBuaa/IterNorm-pytorch/blob/master/extension/normailzation/iterative_normalization.py
"""

import torch.nn
from torch.nn import Parameter

__all__ = ['iterative_normalization', 'IterNorm']

class iterative_normalization_py(torch.autograd.Function):
    """
    Implements Iterative Normalization as described in the paper
    "Iterative Normalization: Beyond Standardization towards Efficient Whitening".

    This performs approximate ZCA whitening using Newton's Iterations.

    Forward Pass:
    1. Compute the mean of the input data and center it.
    2. Compute the covariance matrix of the centered data.
    3. Normalize the covariance matrix by its trace to ensure convergence.
    4. Apply Newton's iterations to approximate the inverse square root of the covariance matrix.
    5. Use the whitening matrix to transform the centered data.

    Backward Pass:
    Compute the gradients w.r.t the input and intermediate variables used during the forward pass.
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # Reshape input tensor for group-wise computation
        ctx.g = X.size(1) // nc  # Number of groups (G)
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)  # Shape: (G, C, M)
        _, d, m = x.size()  # d = channels per group, m = spatial dimensions

        saved = []
        if training:
            # Compute mini-batch mean
            mean = x.mean(-1, keepdim=True)  # Shape: (G, C, 1)
            xc = x - mean  # Centered input, xc = X - \mu
            saved.append(xc)

            # Covariance matrix: Cx = (1/m) * xc * xc^T + \epsilon * I
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)  # Initialize P_0 = I
            Cx = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))

            # Normalize covariance matrix: Cx_N = Cx / tr(Cx)
            rTr = (Cx * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Cx_N = Cx * rTr
            saved.append(Cx_N)

            # Newton's iterations to approximate Cx^{-1/2}
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Cx_N)

            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # Whitening matrix wm = Cx^{-1/2}

            # Update running statistics
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat

        # Whitened output: Xn = wm * xc
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # Centered input
        rTr = saved[1]  # Trace normalization scalar
        sn = saved[2].transpose(-2, -1)  # Normalized Cx (transposed for backward)
        P = saved[3:]  # Iteration results (P_k)
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)  # Reshape grad for group-wise computation
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(1.5, -0.5, g_tmp, P2)
            g_P.baddbmm_(1, -0.5, P2, g_tmp)
            g_P.baddbmm_(1, -0.5, P[k - 1].matmul(g_tmp), P[k - 1])
        g_sn += g_P
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None

class IterNorm(torch.nn.Module):
    """
    Iterative Normalization Module.

    Attributes:
        num_features (int): Number of input features (C).
        num_groups (int): Number of groups for group-wise whitening.
        T (int): Number of iterations for Newton's method.
        eps (float): Epsilon for numerical stability.
        momentum (float): Momentum for running statistics.
        affine (bool): Whether to include learnable scale and shift parameters.

    Args:
        num_features (int): Total number of input channels.
        num_groups (int): Number of groups for group-wise whitening.
        T (int): Number of Newton iterations for approximate whitening.
        eps (float): Small constant to ensure numerical stability.
        momentum (float): Momentum for running statistics.
        affine (bool): Enable/disable learnable parameters (scale and bias).
    """
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim

        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "Invalid num_groups or num_channels."

        self.num_groups = num_groups
        self.num_channels = num_channels

        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        self.register_buffer('running_wm', torch.eye(num_channels).expand(num_groups, num_channels, num_channels))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        """
        Forward pass of Iterative Normalization.

        Args:
            X (torch.Tensor): Input tensor of shape (N, C, ...).

        Returns:
            torch.Tensor: Normalized output of the same shape as input.
        """
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T,
                                                 self.eps, self.momentum, self.training)
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)



if __name__ == '__main__':
    ItN = IterNorm(64, num_groups=8, T=10, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    #x = torch.randn(32, 64, 14, 14)
    x = torch.randn(128, 64)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().view(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))