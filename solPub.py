import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle
from model import Encoder
from model import Decoder
from model import SpeakerClassifier
from model import PatchDiscriminator
from ultPub import to_var
from ultPub import Hps
from ultPub import DataLoader
from ultPub import cc
import os

class Solver(object):
    def __init__(self, hps, data_loader):
        self.hps = hps
        self.data_loader = data_loader
        self.model_kept = []
        self.max_keep = 100
        self.build_model()

    def build_model(self):
        hps = self.hps
        ns = self.hps.ns
        emb_size = self.hps.emb_size
        self.Encoder = cc(Encoder(ns=ns, dp=hps.enc_dp))
        self.Decoder = cc(Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size))
        self.Generator = cc(Decoder(ns=ns, c_a=hps.n_speakers, emb_size=emb_size))
        self.SpeakerClassifier = cc(SpeakerClassifier(ns=ns, n_class=hps.n_speakers, dp=hps.dis_dp))
        self.PatchDiscriminator = cc(nn.DataParallel(PatchDiscriminator(ns=ns, n_class=hps.n_speakers)))
        betas = (0.5, 0.9)
        params = list(self.Encoder.parameters()) + list(self.Decoder.parameters())
        self.ae_opt = optim.Adam(params, lr=self.hps.lr, betas=betas)
        self.clf_opt = optim.Adam(self.SpeakerClassifier.parameters(), lr=self.hps.lr, betas=betas)
        self.gen_opt = optim.Adam(self.Generator.parameters(), lr=self.hps.lr, betas=betas)
        self.patch_opt = optim.Adam(self.PatchDiscriminator.parameters(), lr=self.hps.lr, betas=betas)

    def load_model(self, model_path, enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            self.Decoder.load_state_dict(all_model['decoder'])
            self.Generator.load_state_dict(all_model['generator'])
            if not enc_only:
                self.SpeakerClassifier.load_state_dict(all_model['classifier'])
                self.PatchDiscriminator.load_state_dict(all_model['patch_discriminator'])

    def set_eval(self):
        self.Encoder.eval()
        self.Decoder.eval()
        self.Generator.eval()
        self.SpeakerClassifier.eval()
        self.PatchDiscriminator.eval()

    def test_step(self, x, c, gen=False):
        self.set_eval()
        x = to_var(x).permute(0, 2, 1)
        enc = self.Encoder(x)
        x_tilde = self.Decoder(enc, c)
        if gen:
            x_tilde += self.Generator(enc, c)
        return x_tilde.data.cpu().numpy()
