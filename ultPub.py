import json
import torch
from torch.utils import data
from torch.autograd import Variable
from collections import namedtuple


def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

class Hps(object):
    def __init__(self):
        self.hps = namedtuple('hps', [
            'lr',
            'alpha_dis',
            'alpha_enc',
            'beta_dis', 
            'beta_gen', 
            'beta_clf',
            'lambda_',
            'ns', 
            'enc_dp', 
            'dis_dp', 
            'max_grad_norm',
            'seg_len',
            'emb_size',
            'n_speakers',
            'n_latent_steps',
            'n_patch_steps', 
            'batch_size',
            'lat_sched_iters',
            'enc_pretrain_iters',
            'dis_pretrain_iters',
            'patch_iters', 
            'iters',
            ]
        )
        default = \
            [1e-4, 1, 1e-4, 0, 0, 0, 10, 0.01, 0.5, 0.1, 5, 128, 128, 8, 5, 0, 32, 50000, 5000, 5000, 30000, 60000]
        self._hps = self.hps._make(default)

    def get_tuple(self):
        return self._hps

    def load(self, path):
        with open(path, 'r') as f_json:
            hps_dict = json.load(f_json)
        self._hps = self.hps(**hps_dict)

    def dump(self, path):
        with open(path, 'w') as f_json:
            json.dump(self._hps._asdict(), f_json, indent=4, separators=(',', ': '))

def to_var(x, requires_grad=True):
    x = Variable(x, requires_grad=requires_grad)
    return x.cuda() if torch.cuda.is_available() else x

class DataLoader(object):
    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.n_elements = len(self.dataset[0])
        self.batch_size = batch_size
        self.index = 0

    def all(self, size=1000):
        samples = [self.dataset[self.index + i] for i in range(size)]
        batch = [[s for s in sample] for sample in zip(*samples)]
        batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]

        if self.index + 2 * self.batch_size >= len(self.dataset):
            self.index = 0
        else:
            self.index += self.batch_size
        return tuple(batch_tensor)

    def __iter__(self):
        return self

    def __next__(self):
        samples = [self.dataset[self.index + i] for i in range(self.batch_size)]
        batch = [[s for s in sample] for sample in zip(*samples)]
        batch_tensor = [torch.from_numpy(np.array(data)) for data in batch]

        if self.index + 2 * self.batch_size >= len(self.dataset):
            self.index = 0
        else:
            self.index += self.batch_size
        return tuple(batch_tensor)