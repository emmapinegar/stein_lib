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
from stein_lib.svgd.base_kernels import RBF, RBF_Anisotropic
from stein_lib.utils import create_movie_2D, plot_graph_2D, plot_graph_2D_slices, plot_graph_2D_gradient_slices, create_movie_2D_slices, create_trace_3D
from stein_lib.prm_utils import get_graph

dtype = torch.float32
torch.set_default_dtype(dtype)
device_ = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device_)

def test_brain_3D():
    ###### Params ######

    num_particles = 10000
    iters = 20
    analytic_grads = True

    median_heuristic = True
    if median_heuristic:
        bandwidth = -1.
    else:
        bandwidth = 250.

    repulsive_scaling = 1
    if analytic_grads:
        repulsive_scaling = -repulsive_scaling
    step_size = 1.

    lim_func = False
    limit_scale = 1
    transform = np.loadtxt("./remind_001_obstacles.txt", max_rows=4)


    # Sample intial particles
    torch.manual_seed(5432876)

    grad_ax_limits = [[35., 180.],[55., 230.],[30., 140.]]
    plot_buffer = 5.
    sample_buffer = 0.5

    plot_ax_limits = [[0., 0.], [0., 0.], [0., 0.]]
    for limit_ind in range(len(plot_ax_limits)):
        plot_ax_limits[limit_ind][0] = grad_ax_limits[limit_ind][0] - plot_buffer
        plot_ax_limits[limit_ind][1] = grad_ax_limits[limit_ind][1] + plot_buffer 

    low_sample_limits = [grad_ax_limits[0][0] + sample_buffer, grad_ax_limits[1][0] + sample_buffer, grad_ax_limits[2][0] + sample_buffer]
    high_sample_limits = [grad_ax_limits[0][1] - sample_buffer, grad_ax_limits[1][1] - sample_buffer, grad_ax_limits[2][1] - sample_buffer] 

    # low_sample_limits = [31., 55., 88.]
    # high_sample_limits = [35., 60., 92.] 
    print(f"low limits: {low_sample_limits} high limits: {high_sample_limits} buffer: {sample_buffer} {grad_ax_limits[0][0]} test: {grad_ax_limits[0][0] + sample_buffer}")
    print(f"plot limits: {plot_ax_limits}")
    for i in range(1):
        #========================== Sampling ==============================
        ## Large Gaussian in center of remind map.
        # prior_dist = Normal(loc=torch.tensor([30.,90.,15.]), scale=torch.tensor([20.,20.,20.]))

        ## Small gaussian in corner of remind map.
        # prior_dist = Normal(loc=torch.tensor([12.,-3.]), scale=torch.tensor([1.,1.]))

        ## Two small gaussians in opposing corners of remind map.
        # sigma = 5.
        # radii_list = [[sigma, sigma],] * 2
        # prior_dist = mixture_of_gaussians(num_comp=2, mu_list=[[12.,-3.], [-5, -18] ], sigma_list=radii_list,)

        # Uniform distribution
        prior_dist = Uniform(low=torch.tensor(low_sample_limits), high=torch.tensor(high_sample_limits))
        # prior_dist = Uniform(low=torch.tensor([50., 70., 40.]), high=torch.tensor([170., 220., 130.]))

        particles_0 = prior_dist.sample((num_particles,))
        particles = particles_0.clone().cpu().numpy()
        particles = torch.from_numpy(particles)

        # Load model
        from Bayesian_Hilbert_Maps import bhmlib
        bhm_path = Path(bhmlib.__path__[0]).resolve()
        model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_remind_3D_res2_final.pt'

        if not lim_func:
            grad_ax_limits = None
        model = BayesianHilbertMap(model_file, grad_ax_limits, dim=3, device=device_, limit_scale=limit_scale)
        
        #================== Kernel ===========================

        # kernel = RBF(hessian_scale=1.0, analytic_grad=True, median_heuristic=False, bandwidth=1.0,)

        kernel = RBF_Anisotropic(hessian_scale=1.0, analytic_grad=analytic_grads, median_heuristic=median_heuristic, bandwidth=bandwidth,)

        #================== Optimizer ===========================

        # optimizer = torch.optim.SGD([particles], lr=step_size)

        optimizer = torch.optim.Adam([particles], lr=step_size)

        # optimizer = torch.optim.LBFGS([particles], lr=step_size, max_iter=100, max_eval=20 * 1.25, tolerance_change=1e-9, history_size=25, line_search_fn=None,)

        # optimizer = FullBatchLBFGS([particles], lr=step_size, history_size=25, line_search='None',)

        #================== SVGD ===========================

        svgd = SVGD(kernel=kernel, kernel_structure=None, repulsive_scaling=repulsive_scaling, geom_metric_type='fisher', verbose=True,)

        ## Optimize
        (particles, p_hist, pw_dists, pw_dists_scaled) = svgd.apply(particles, model, iters, use_analytic_grads=analytic_grads, optimizer=optimizer,)

        print(f"\nMean Est.: {particles.mean(0)} \t Std Est.: {particles.std(0)}")

        #=============================================

        
        particles_ = particles.detach().cpu().numpy()
        particles_ = np.transpose(particles_)
        particles_ = np.concatenate((particles_, np.ones((1, np.shape(particles_)[1]))))
        particles_ = np.matmul(transform, particles_)
        particles_ = np.transpose(particles_[:,0:3])
        # old_particles = np.loadtxt("./remind_001_samples.txt")
        # particles_ = np.vstack((old_particles[:,0:3], particles_))
        np.savetxt("./remind_001_samples.txt", particles_, fmt='%9f ')

        fig_prename = f"./figures/{optimizer.__class__.__name__}_{kernel.__class__.__name__}_np_{num_particles}_iters_{iters}_eps_{step_size}_repulsive_{repulsive_scaling}_bw_{bandwidth}_limfunc_{lim_func}_analytic_{analytic_grads}"
        kernel_optim_str = f"{optimizer.__class__.__name__} {kernel.__class__.__name__} np={num_particles} eps={step_size}"


        plot_graph_2D_slices(particles.detach(), model.log_prob, ax_limits=plot_ax_limits, to_numpy=True,
            save_path=f"{fig_prename}_slice.png" , case_name=kernel_optim_str, dtype=dtype) 

        # Make 2D slice images of vacancy log prob, particles, and particle gradients
        plot_graph_2D_gradient_slices(particles.detach(), model.log_prob, model.grad_log_p, svgd.phi, ax_limits=plot_ax_limits, to_numpy=True,
            save_path=f"{fig_prename}_grad.png" , case_name=kernel_optim_str, dtype=dtype) 

        # Make movie of 2D projections of particles moving 
        create_movie_2D_slices(p_hist, model.log_prob, to_numpy=True, save_path=f"{fig_prename}_slice.mp4" ,
            ax_limits=plot_ax_limits, case_name=kernel_optim_str, dtype=dtype)
        
        # Make 3D plot of the movement of all particles
        create_trace_3D(p_hist, model.log_prob, to_numpy=True, save_path=f"{fig_prename}_trace.png",
            ax_limits=plot_ax_limits, case_name=kernel_optim_str)


def test_brain_2D():
    ###### Params ######
    num_particles = 500
    iters = 500

    analytic_grads = False

    median_heuristic = False
    if median_heuristic:
        bandwidth = -1.
    else:
        bandwidth = 250.

    repulsive_scaling = 0.
    if analytic_grads:
        repulsive_scaling = -repulsive_scaling
    step_size = 1.

    lim_func = False


    # Sample intial particles
    torch.manual_seed(5432876)

    grad_ax_limits = [[35., 180.],[55., 230.],[68., 72.]]
    plot_buffer = 5.
    sample_buffer = 0.5

    plot_ax_limits = [[0., 0.], [0., 0.], [0., 0.]]
    for limit_ind in range(len(plot_ax_limits)):
        plot_ax_limits[limit_ind][0] = grad_ax_limits[limit_ind][0] - plot_buffer
        plot_ax_limits[limit_ind][1] = grad_ax_limits[limit_ind][1] + plot_buffer 

    low_sample_limits = [grad_ax_limits[0][0] + sample_buffer, grad_ax_limits[1][0] + sample_buffer, grad_ax_limits[2][0] + sample_buffer]
    high_sample_limits = [grad_ax_limits[0][1] - sample_buffer, grad_ax_limits[1][1] - sample_buffer, grad_ax_limits[2][1] - sample_buffer] 

    # low_sample_limits = [31., 55., 88.]
    # high_sample_limits = [35., 60., 92.] 
    print(f"low limits: {low_sample_limits} high limits: {high_sample_limits} buffer: {sample_buffer} {grad_ax_limits[0][0]} test: {grad_ax_limits[0][0] + sample_buffer}")
    print(f"plot limits: {plot_ax_limits}")

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
    model_file = bhm_path / 'Outputs' / 'saved_models' / 'bhm_remind_test_log_res1.5_iter200.pt'
    if not lim_func:
        grad_ax_limits = None
    model = BayesianHilbertMap(model_file, grad_ax_limits, dim=3, device=device_)
    
    #================== Kernel ===========================

    # kernel = RBF(hessian_scale=1.0, analytic_grad=True, median_heuristic=False, bandwidth=1.0,)

    kernel = RBF_Anisotropic(hessian_scale=1.0, analytic_grad=analytic_grads, median_heuristic=median_heuristic, bandwidth=bandwidth,)

    #================== Optimizer ===========================

    # optimizer = torch.optim.SGD([particles], lr=step_size)

    optimizer = torch.optim.Adam([particles], lr=step_size)

    # optimizer = torch.optim.LBFGS([particles], lr=step_size, max_iter=100, max_eval=20 * 1.25, tolerance_change=1e-9, history_size=25, line_search_fn=None,)

    # optimizer = FullBatchLBFGS([particles], lr=step_size, history_size=25, line_search='None',)

    #================== SVGD ===========================

    svgd = SVGD(kernel=kernel, kernel_structure=None, repulsive_scaling=repulsive_scaling, geom_metric_type='fisher', verbose=True,)

    ## Optimize
    (particles, p_hist, pw_dists, pw_dists_scaled) = svgd.apply(particles, model, iters, use_analytic_grads=analytic_grads, optimizer=optimizer,)

    print(f"\nMean Est.: {particles.mean(0)} \t Std Est.: {particles.std(0)}")

    #=============================================
    fig_prename = f"./figures/{optimizer.__class__.__name__}_{kernel.__class__.__name__}_np_{num_particles}_iters_{iters}_eps_{step_size}_repulsive_{repulsive_scaling}_bw_{bandwidth}_limfunc_{lim_func}_analytic_{analytic_grads}"
    kernel_optim_str = f"{optimizer.__class__.__name__} {kernel.__class__.__name__} np={num_particles} eps={step_size}"

    # Plot Graph
    plot_graph_2D(particles.detach(), np.array([]), model.log_prob, edge_vals=None, edge_coll_thresh=None, edge_coll_pts=None, 
                  ax_limits=plot_ax_limits, to_numpy=True, save_path=f"{fig_prename}_2D.png",)

    # Make movie
    create_movie_2D(p_hist, model.log_prob, to_numpy=True, save_path=f"{fig_prename}_2D.mp4",
                    ax_limits=plot_ax_limits, opt=optimizer.__class__.__name__, kernel_base_type=kernel.__class__.__name__, num_particles=num_particles, eps=step_size,)