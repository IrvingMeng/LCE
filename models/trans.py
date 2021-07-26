import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter

__all__ = ['RBTBlock']

class RBTBlock(nn.Module):
    """
    The RBT block introduced in <Unified Representation Learning for Cross Model Compatibility>
    """
    def __init__(self, in_planes, out_planes, num_paths=4):
        super(RBTBlock, self).__init__()
        if not isinstance(num_paths, int) or not 0 <= num_paths <= 4:
            raise ValueError('num_paths: {}'.format(num_paths))
        self.num_paths = num_paths
        if self.num_paths == 0:
            print('No need to construct trans.path since num_paths is 0.')
        for i in range(self.num_paths):
            print('Construct trans.path{}: {} -> {}'.format(i + 1, in_planes, out_planes))
            setattr(self, 'path{}'.format(i + 1), self._make_onepath(in_planes, out_planes))

    def _make_onepath(self, in_planes, out_planes):
        return nn.Sequential(nn.Linear(in_planes, 16, bias=False),
                             nn.BatchNorm1d(16, eps=2e-05, momentum=0.9),
                             nn.PReLU(16),
                             nn.Linear(16, 16, bias=False),
                             nn.BatchNorm1d(16, eps=2e-05, momentum=0.9),
                             nn.PReLU(16),
                             nn.Linear(16, out_planes, bias=False),
                             nn.BatchNorm1d(out_planes, eps=2e-05, momentum=0.9),
                             nn.PReLU(out_planes)
                             )

    def forward(self, feat):
        output = feat
        for i in range(self.num_paths):
            output = output + getattr(self, 'path{}'.format(i + 1))(feat)
        return output