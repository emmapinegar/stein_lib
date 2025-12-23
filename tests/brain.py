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

import torch
import numpy as np
from torch.distributions import Normal, Uniform
from stein_lib.models.gaussian_mixture import mixture_of_gaussians
from stein_lib.svgd.svgd import SVGD
from pathlib import Path
from stein_lib.models.bhm import BayesianHilbertMap
from stein_lib.utils import create_movie_2D, plot_graph_2D, plot_graph_2D_slices, create_movie_2D_slices
from stein_lib.prm_utils import get_graph

torch.set_default_dtype(torch.float64)

def test_brain_3D():
    ###### Params ######
    # num_particles = 100
    num_particles = 5000
    # iters = 3000
    # iters = 200
    iters = 100
    # iters = 1

    # Sample intial particles
    torch.manual_seed(1)

    ## Large Gaussian in center of remind map.
    prior_dist = Normal(loc=torch.tensor([30.,90.,15.]), scale=torch.tensor([10.,10.,5.]))

    ## Small gaussian in corner of remind map.
    # prior_dist = Normal(loc=torch.tensor([12.,-3.]), scale=torch.tensor([1.,1.]))

    ## Two small gaussians in opposing corners of remind map.
    # sigma = 5.
    # radii_list = [[sigma, sigma],] * 2
    # prior_dist = mixture_of_gaussians(num_comp=2, mu_list=[[12.,-3.], [-5, -18] ], sigma_list=radii_list,)

    # Uniform distribution
    # prior_dist = Uniform(low=torch.tensor([0., 20., -30.]), high=torch.tensor([80., 140., 40.]))


    particles_0 = prior_dist.sample((num_particles,))

    # Load model
    from Bayesian_Hilbert_Maps import bhmlib
    bhm_path = Path(bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_remind_res1_iter114.pt'
    # model_file = '/tmp/bhm_intel_res0.25_iter100.pt'
    ax_limits = [[-45, 112],[5, 167],[-52,59]]
    model = BayesianHilbertMap(model_file, ax_limits, dim=3)

    #================== SVGD ===========================
    particles = particles_0.clone().cpu().numpy()
    particles = torch.from_numpy(particles)

    # kernel_base_type = 'RBF'
    # # optimizer_type = 'SGD'
    # optimizer_type = 'Adam'
    # step_size = 1.
    # svgd = SVGD(kernel_base_type=kernel_base_type, kernel_structure=None, median_heuristic=False, repulsive_scaling=1.,
    #     geom_metric_type=None, verbose=True, bandwidth=5.,)

    kernel_base_type = 'RBF_Anisotropic'
    # optimizer_type = 'SGD'
    optimizer_type = 'Adam'
    step_size = 1
    # step_size = 0.
    svgd = SVGD(kernel_base_type=kernel_base_type, kernel_structure=None, median_heuristic=False, repulsive_scaling=5.,
        geom_metric_type='fisher', verbose=True, bandwidth=5.,)


    # kernel_base_type = 'RBF_Anisotropic'
    # optimizer_type = 'LBFGS' # 'FullBatchLBFGS'
    # step_size = 0.1
    # svgd = SVGD(kernel_base_type=kernel_base_type, kernel_structure=None, median_heuristic=False, repulsive_scaling=1.,
    #     geom_metric_type='fisher', verbose=True, bandwidth=5.,)

    ## Optimize
    (particles, p_hist, pw_dists, pw_dists_scaled) = svgd.apply(particles, model, iters, step_size, use_analytic_grads=False, optimizer_type=optimizer_type,)

    print("\nMean Est.: ", particles.mean(0))
    print("Std Est.: ", particles.std(0))

    #=============================================

    plot_graph_2D_slices(particles.detach(), model.log_prob, ax_limits=ax_limits, to_numpy=True,
        save_path='./figures/graph_svgd_{}_bhm_remind3d_np_{}_eps_{}.png'.format(kernel_base_type, num_particles, step_size,),)    

    # Make movie
    create_movie_2D_slices(p_hist, model.log_prob, to_numpy=True, save_path='./figures/svgd_{}_bhm_remind3d_np_{}_eps_{}.mp4'.format(kernel_base_type, num_particles, step_size,),
        ax_limits=ax_limits, opt='SVGD', kernel_base_type=kernel_base_type, num_particles=num_particles, eps=step_size,)


def test_brain_2D():
    ###### Params ######
    # num_particles = 100
    num_particles = 500
    iters = 500
    # iters = 200
    # iters = 100
    # iters = 1

    # Sample intial particles
    torch.manual_seed(1)

    ## Large Gaussian in center of intel map.
    prior_dist = Normal(loc=torch.tensor([30.,90.]), scale=torch.tensor([10., 10.]))
    particles_0 = prior_dist.sample((num_particles,))
    print(f"particles: {particles_0.size()}")

    ## Small gaussian in corner of intel map.
    # prior_dist = Normal(loc=torch.tensor([12.,-3.]), scale=torch.tensor([1.,1.]))

    ## Two small gaussians in opposing corners of intel map.
    # sigma = 5.
    # radii_list = [[sigma, sigma],] * 2
    # prior_dist = mixture_of_gaussians(num_comp=2, mu_list=[[12.,-3.], [-5, -18] ], sigma_list=radii_list,)

    ## Uniform distribution
    prior_dist = Uniform(low=torch.tensor([-25., 25.]), high=torch.tensor([92., 147.]))
    particles_1 = prior_dist.sample((num_particles//2,))

    particles_0 = torch.vstack((particles_0, particles_1))
    print(f"particles: {particles_0.size()}")

    

    # Load model
    from Bayesian_Hilbert_Maps import bhmlib
    bhm_path = Path(bhmlib.__path__[0]).resolve()
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_remind_res1_iter000.pt'
    # model_file = '/tmp/bhm_intel_res0.25_iter100.pt'
    ax_limits = [[-45, 112],[5, 167]]
    model = BayesianHilbertMap(model_file, ax_limits)

    #================== SVGD ===========================
    particles = particles_0.clone().cpu().numpy()
    particles = torch.from_numpy(particles)

    # kernel_base_type = 'RBF'
    # # optimizer_type = 'SGD'
    # optimizer_type = 'Adam'
    # step_size = 1.
    # svgd = SVGD(kernel_base_type=kernel_base_type, kernel_structure=None, median_heuristic=False, repulsive_scaling=1.,
    #     geom_metric_type=None, verbose=True, bandwidth=5.,)

    kernel_base_type = 'RBF_Anisotropic'
    # optimizer_type = 'SGD'
    optimizer_type = 'Adam'
    step_size = 0.25
    # step_size = 0.
    svgd = SVGD(kernel_base_type=kernel_base_type, kernel_structure=None, median_heuristic=False, repulsive_scaling=3.,
                geom_metric_type='fisher', verbose=True, bandwidth=5.,)


    # kernel_base_type = 'RBF_Anisotropic'
    # optimizer_type = 'LBFGS' # 'FullBatchLBFGS'
    # step_size = 0.1
    # svgd = SVGD(kernel_base_type=kernel_base_type, kernel_structure=None, median_heuristic=False, repulsive_scaling=1.,
    #     geom_metric_type='fisher', verbose=True, bandwidth=5.,)

    ## Optimize
    (particles, p_hist, pw_dists, pw_dists_scaled) = svgd.apply(particles, model, iters, step_size, 
                                                                use_analytic_grads=False, optimizer_type=optimizer_type,)

    print("\nMean Est.: ", particles.mean(0))
    print("Std Est.: ", particles.std(0))

    #=============================================

    # Plot Graph
    plot_graph_2D(particles.detach(), np.array([]), model.log_prob, edge_vals=None, edge_coll_thresh=None, edge_coll_pts=None, 
                  ax_limits=ax_limits, to_numpy=True, save_path='./figures/graph_svgd_{}_bhm_remind_np_{}_eps_{}.png'.format(kernel_base_type, num_particles, step_size,),)

    # Make movie
    create_movie_2D(p_hist, model.log_prob, to_numpy=True, save_path='./figures/svgd_{}_bhm_remind_np_{}_eps_{}.mp4'.format(kernel_base_type, num_particles, step_size,),
                    ax_limits=ax_limits, opt='SVGD', kernel_base_type=kernel_base_type, num_particles=num_particles, eps=step_size,)
