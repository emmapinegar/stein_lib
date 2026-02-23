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
    def __init__(self, file_path=None, limits=((-10, 20,), (-25, 5)), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dim=2, limit_scale=1):
        self.dim = dim
        self.device = device
        self.limit_scale = limit_scale
        # Load trained Bayesian Hilbert Map
        params = torch.load(file_path)
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                params[k] = v.to(device)
        if limits is not None:
            self.limits = torch.tensor(limits).to(device)
        else:
            self.limits = None
        self.bhm = BHM_PYTORCH(torch_kernel_func=True, cell_max_min=self.limits, limit_scale=self.limit_scale, **params)


    def log_prob(self, x):
        if x.dim() == 1:
            x = x.view(1, -1)
        log_p = self.bhm.log_prob_vacancy(x)
        print_str = f"log_p: {log_p[0]:5.6f} "
        if self.limits is not None:

            log_p -= torch.exp(-self.limit_scale*(x[:, 0] - self.limits[0, 0]))
            log_p -= torch.exp( self.limit_scale*(x[:, 0] - self.limits[0, 1]))
            log_p -= torch.exp(-self.limit_scale*(x[:, 1] - self.limits[1, 0]))
            log_p -= torch.exp( self.limit_scale*(x[:, 1] - self.limits[1, 1]))
            z_str = ""
            if self.dim > 2:
                log_p -= torch.exp(-self.limit_scale*(x[:, 2] - self.limits[2, 0]))
                log_p -= torch.exp( self.limit_scale*(x[:, 2] - self.limits[2, 1]))  
                z_str = f" z: {x[0,2] - self.limits[2,0]:5.6f} {x[0,2] - self.limits[2,1]:5.6f}"
            print_str = f"{print_str} new: {log_p[0]:5.6f} x: {x[0,0] - self.limits[0,0]:5.6f} {x[0,0] - self.limits[0,1]:5.6f} y: {x[0,1] - self.limits[1,0]:5.6f} {x[0,1] - self.limits[1,1]:5.6f} {z_str}"    
        print(print_str)                              
        return log_p

    def grad_log_p(self, x, sub_limits=True):
        return self.bhm.grad_log_p_vacancy(x, sub_limits=sub_limits)

if __name__ == '__main__':

    import Bayesian_Hilbert_Maps.bhmlib
    bhm_path = Path(Bayesian_Hilbert_Maps.bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_intel_res0.25_iter010.pt'

    bhm = BayesianHilbertMap(model_file)

