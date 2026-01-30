"""
Copyright (c) 2020-2021 Alexander Lambert

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import torch
from pathlib import Path
from Bayesian_Hilbert_Maps.bhmlib.BHM.pytorch.bhm_pytorch import BHM_PYTORCH


class BayesianHilbertMap:
    def __init__(self, file_path=None, limits=((-10, 20,), (-25, 5)), dim=2, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        # Load trained Bayesian Hilbert Map
        params = torch.load(file_path)
        self.dim = dim
        self.bhm = BHM_PYTORCH(torch_kernel_func=True, device=device, cell_max_min=limits, file=file_path) #, **params)
        # self.bhm.load(file_path)
        if limits is not None:
            self.limits = torch.tensor(limits, device=device)
        else: 
            self.limits = limits

    def log_prob(self, x):
        log_p = self.bhm.log_prob_vacancy(x)

        if self.limits is not None:
            scale = 1.
            print_str = f"log_p: {log_p[0]:5.6f} "
            log_p -= torch.exp(-scale*(x[:, 0] - self.limits[0, 0]))
            log_p -= torch.exp( scale*(x[:, 0] - self.limits[0, 1]))
            log_p -= torch.exp(-scale*(x[:, 1] - self.limits[1, 0]))
            log_p -= torch.exp( scale*(x[:, 1] - self.limits[1, 1]))
            if self.dim == 3:
                log_p -= torch.exp(-scale*(x[:, 2] - self.limits[2, 0]))
                log_p -= torch.exp( scale*(x[:, 2] - self.limits[2, 1]))
            print(f"{print_str} new: {log_p[0]:5.6f} x: {x[0,0] - self.limits[0,0]:5.6f} {x[0,0] - self.limits[0,1]:5.6f} y: {x[0,1] - self.limits[1,0]:5.6f} {x[0,1] - self.limits[1,1]:5.6f} z: {x[0,2] - self.limits[2,0]:5.6f} {x[0,2] - self.limits[2,1]:5.6f}")        
        else:
            print(f"log_p: {log_p[0]:5.6f}")
        #     log_p_mod = log_p - log_diff
        # else:
        #     log_p_mod = log_p
        return log_p

    def grad_log_p(self, x):
        return self.bhm.grad_log_p_vacancy(x)


if __name__ == '__main__':

    from Bayesian_Hilbert_Maps import bhmlib
    bhm_path = Path(bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter010.pt'

    bhm = BayesianHilbertMap(model_file)

