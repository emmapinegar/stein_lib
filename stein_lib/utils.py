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
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

num_levels = 20
slice_buffer = 5

def get_jacobian(gradient, X,):
    """
    Returns the Jacobian matrix, given the gradient
    Parameters
    ----------
    gradient : (Tensor)
        Of shape [dim, batch]
    X : (Tensor)
        of shape [dim, batch]
    Returns
    -------
    J : (Tensor)
        Jacobian, of shape [dim, dim, batch]
    """
    dg_dXi = [torch.autograd.grad(gradient[:, i].sum(), X, retain_graph=True,)[0] for i in range(gradient.shape[1])]
    J = torch.stack(dg_dXi, dim=1)
    return J


def calc_pw_distances(X):
    """
    Returns the pairwise distances between particles.
    Parameters
    ----------
    X : Tensor
        Points. of shape [dim, batch]
    """
    XX = X.matmul(X.t())
    pairwise_dists_sq = -2 * XX + XX.diag().unsqueeze(1) + XX.diag().unsqueeze(0)
    pw_dists = torch.sqrt(pairwise_dists_sq)
    return pw_dists


def calc_scaled_pw_distances(X, M):
    """
    Returns the metric-scaled / anisotropic pairwise distances between particles.
    Parameters
    ----------
    X : Tensor
        Points. of shape [dim, batch]
    M : Tensor
        Metric. of shape [dim, dim]
    """
    X_M_Xt = X @ M @ X.t()
    pw_dists_sq = -2 * X_M_Xt + X_M_Xt.diag().unsqueeze(1) + X_M_Xt.diag().unsqueeze(0)
    pw_dists = torch.sqrt(pw_dists_sq)
    return pw_dists


def plot_graph_2D(particles, edges, log_prob, edge_vals=None, edge_coll_pts=None, edge_coll_thresh=None, save_path='/tmp/graph.png', to_numpy=False, ax_limits=[[-4, 4],[4, 4]],):

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()

    ngrid = 100
    x = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
    if len(ax_limits) > 2:
        z = np.array([(ax_limits[2][1] - ax_limits[2][0])//2 + ax_limits[2][0]])
        X, Y, Z = np.meshgrid(x,y,z)
        grid = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y),np.ndarray.flatten(Z))) 
        particles = particles[np.where(np.logical_and(particles[:,2] >= z[0], particles[:,2] < z[0] + slice_buffer))[0], :]       
    else:
        X, Y = np.meshgrid(x,y)
        grid = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y)),)

    if to_numpy:
        grid = torch.from_numpy(grid)
        z = log_prob(grid.t()).cpu().numpy()
        Z = np.exp(z).reshape(ngrid, ngrid)
        particles = particles.detach().cpu().numpy()
    else:
        Z = np.exp(log_prob(grid)).reshape(ngrid, ngrid)

    print(f"x: {np.shape(X.flatten())} y: {np.shape(Y)} z: {np.shape(Z)}")
    print(X)
    plt.contourf(X.reshape(ngrid, ngrid), Y.reshape(ngrid, ngrid), Z, num_levels)

    for i in range(edges.shape[0]):
        node_pair = edges[i]
        color = 'k'
        if edge_vals is not None:
            if edge_vals[i] < edge_coll_thresh:
                color = 'b'
        plt.plot(particles[node_pair, 0], particles[node_pair, 1], color, markersize=1,)

    if edge_coll_pts is not None:
        plt.plot(edge_coll_pts[:, 0], edge_coll_pts[:, 1], 'go', markersize=2)

    xlim = ax_limits[0]
    ylim = ax_limits[1]
    plt.plot(particles[:, 0], particles[:, 1], 'ro', markersize=3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.savefig(save_path)
    plt.show()


def create_movie_2D(particle_hist, log_prob, save_path="/tmp/stein_movie.mp4", ax_limits=[[-4, 4],[4, 4]], 
                    to_numpy=False, kernel_base_type=None, opt=None, num_particles=None, eps=None,):

    k_type = kernel_base_type,
    if kernel_base_type == 'RBF_Anisotropic':
        k_type = 'RBF_H'

    case_name = '{}-{} (np = {}, eps = {})'.format(
        opt,
        k_type,
        num_particles,
        eps,
    )

    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.set_title(case_name + '\n' + str(0) + '$ ^{th}$ iteration')

    ngrid = 100
    x = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
    if len(ax_limits) > 2:
        z = np.array([(ax_limits[2][1] - ax_limits[2][0])//2 + ax_limits[2][0]])
        X, Y, Z = np.meshgrid(x,y,z)
        grid = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y),np.ndarray.flatten(Z))) 
        new_particle_hist = []
        for time_ind in range(len(particle_hist)):
            particles = particle_hist[time_ind]
            particles = particles[np.where(np.logical_and(particles[:,2] >= z, particles[:,2] < z + slice_buffer))[0], :]  
            new_particle_hist += [particles]    
        particle_hist = new_particle_hist 
    else:
        X, Y = np.meshgrid(x,y)
        grid = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y)),)

    if to_numpy:
        grid = torch.from_numpy(grid)
        z = log_prob(grid.t()).cpu().numpy()
        Z = np.exp(z).reshape(ngrid, ngrid)
    else:
        Z = np.exp(log_prob(grid),).reshape(ngrid, ngrid)

    plt.contourf(X.reshape(ngrid, ngrid), Y.reshape(ngrid, ngrid), Z, num_levels)
    xlim = ax_limits[0]
    ylim = ax_limits[1]
    p_start = particle_hist[0]
    particles = plt.plot(p_start[:, 0], p_start[:, 1], 'ro', markersize=3)
    n_iter = len(particle_hist)

    def _init():  # only required for blitting to give a clean slate.
        # ax.set_title(str(0) + '$ ^{th}$ iteration')
        ax.set_title(case_name + '\n' + str(0) + '$ ^{th}$ iteration')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return particles

    def _animate(i):
        # ax.set_title(str(i) + '$ ^{th}$ iteration')
        ax.set_title(case_name + '\n' + str(i) + '$ ^{th}$ iteration')
        pos = particle_hist[i]
        particles[0].set_xdata(pos[:, 0])
        particles[0].set_ydata(pos[:, 1])
        return particles

    ani = animation.FuncAnimation(fig, _animate, frames=n_iter, init_func=_init, interval=100,)

    ani.save(save_path)
    plt.show()


def plot_graph_2D_slices(particles, log_prob, save_path='/tmp/graph.png', to_numpy=False, ax_limits=[[-4,4],[4,4]],):

    if to_numpy:
        particles = particles.detach().cpu().numpy()

    fig = plt.figure(figsize=(15,5))
    # ax = plt.gca()

    ngrid = 100
    X = []
    Y = []
    Z = []
    # C = []
    slice_grids = []
    slice_particles = []

    # slice 1, xy plane
    x1 = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y1 = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
    z1 = np.array([(ax_limits[2][1] - ax_limits[2][0])//2 + ax_limits[2][0]])
    X1, Y1, Z1 = np.meshgrid(x1,y1,z1)
    X += [X1]; Y+= [Y1]; Z += [Z1]
    slice_grids += [np.vstack((np.ndarray.flatten(X1), np.ndarray.flatten(Y1),np.ndarray.flatten(Z1)))]
    slice_particles += [particles[np.where(np.logical_and(particles[:,2] >= z1[0], particles[:,2] < z1[0] + slice_buffer))[0], 0:2]]      

    # slice 2, xz plane
    z2 = np.linspace(ax_limits[2][0], ax_limits[2][1], ngrid)
    y2 = np.array([(ax_limits[1][1] - ax_limits[1][0])//2 + ax_limits[1][0]])
    X2, Y2, Z2 = np.meshgrid(x1, y2, z2)
    X += [X2]; Y+= [Z2]; Z += [Y2]
    slice_grids += [np.vstack((np.ndarray.flatten(X2), np.ndarray.flatten(Y2),np.ndarray.flatten(Z2)))]
    slice_particles += [particles[np.where(np.logical_and(particles[:,1] >= y2[0], particles[:,1] < y2[0] + slice_buffer))[0], 0:3:2]] 

    # slice 3, yz plane
    x3 = np.array([(ax_limits[0][1] - ax_limits[0][0])//2 + ax_limits[0][0]])
    X3, Y3, Z3 = np.meshgrid(x3, y1, z2)
    X += [Y3]; Y+= [Z3]; Z += [X3]
    slice_grids += [np.vstack((np.ndarray.flatten(X3), np.ndarray.flatten(Y3),np.ndarray.flatten(Z3)))]
    slice_particles += [particles[np.where(np.logical_and(particles[:,0] >= x3[0], particles[:,0] < x3[0] + slice_buffer))[0], 1:]] 

    for slice_ind in range(len(X)):
        plt.subplot(1,len(X),slice_ind+1)
        if to_numpy:
            grid = torch.from_numpy(slice_grids[slice_ind])
            c = log_prob(grid.t()).cpu().numpy()
            C = np.exp(c).reshape(ngrid, ngrid)
        else:
            C = np.exp(log_prob(grid)).reshape(ngrid, ngrid)
        plt.scatter(X[slice_ind], Y[slice_ind], c=C, s=1.5)
        # plt.contourf(X[slice_ind].reshape(ngrid, ngrid), Y[slice_ind].reshape(ngrid, ngrid), C, num_levels, vmax=1, vmin=0)
        # xlim = ax_limits[0]
        # ylim = ax_limits[1]
        plt.plot(slice_particles[slice_ind][:, 0], slice_particles[slice_ind][:, 1], 'ro', markersize=3)
        plt.axis('equal')
        plt.colorbar()

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

    plt.savefig(save_path)
    plt.close()    


def plot_graph_2D_gradient_slices(particles, log_prob, grad_log_prob, phi, save_path='/tmp/graph.png', to_numpy=False, ax_limits=[[-4,4],[4,4]],):

    if to_numpy:
        particles = particles.detach().cpu().numpy()

    fig = plt.figure(figsize=(15,5))
    # ax = plt.gca()

    ngrid = 50
    X = []
    Y = []
    Z = []
    # C = []
    slice_grids = []
    slice_particles = []

    # slice 1, xy plane
    x1 = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y1 = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
    z1 = np.array([(ax_limits[2][1] - ax_limits[2][0])//2 + ax_limits[2][0]])
    X1, Y1, Z1 = np.meshgrid(x1,y1,z1)
    X += [X1]; Y+= [Y1]; Z += [Z1]
    slice_grids += [np.vstack((np.ndarray.flatten(X1), np.ndarray.flatten(Y1),np.ndarray.flatten(Z1)))]
    slice_particles += [particles[np.where(np.logical_and(particles[:,2] >= z1[0], particles[:,2] < z1[0] + slice_buffer))[0], :]]      

    # slice 2, xz plane
    z2 = np.linspace(ax_limits[2][0], ax_limits[2][1], ngrid)
    y2 = np.array([(ax_limits[1][1] - ax_limits[1][0])//2 + ax_limits[1][0]])
    X2, Y2, Z2 = np.meshgrid(x1, y2, z2)
    X += [X2]; Y+= [Y2]; Z += [Z2]
    slice_grids += [np.vstack((np.ndarray.flatten(X2), np.ndarray.flatten(Y2),np.ndarray.flatten(Z2)))]
    slice_particles += [particles[np.where(np.logical_and(particles[:,1] >= y2[0], particles[:,1] < y2[0] + slice_buffer))[0], :]] 

    # slice 3, yz plane
    x3 = np.array([(ax_limits[0][1] - ax_limits[0][0])//2 + ax_limits[0][0]])
    X3, Y3, Z3 = np.meshgrid(x3, y1, z2)
    X += [X3]; Y+= [Y3]; Z += [Z3]
    slice_grids += [np.vstack((np.ndarray.flatten(X3), np.ndarray.flatten(Y3),np.ndarray.flatten(Z3)))]
    slice_particles += [particles[np.where(np.logical_and(particles[:,0] >= x3[0], particles[:,0] < x3[0] + slice_buffer))[0], :]] 
    ax = fig.add_subplot(projection='3d')
    for slice_ind in range(1): #range(len(X)):
        
        # plt.subplot(1,len(X),slice_ind+1)
        if to_numpy:
            grid = torch.from_numpy(slice_grids[slice_ind])
            c = grad_log_prob(grid.t()).cpu().numpy()
            particles_t = torch.tensor(slice_particles[slice_ind])
            print(np.shape(slice_grids[slice_ind]))
            print(particles_t.size())
            particles_log = log_prob(grid.t())
            particles_grad = grad_log_prob(particles_t)
            particles_phi, dists_sq = phi(particles_t, particles_grad, dlog_lh=particles_grad)
            particles_phi = 500*particles_phi.cpu().numpy()
            particles_grad = particles_grad.cpu().numpy()
            particles_log = np.exp(particles_log.cpu().numpy())
            # C = np.exp(c).reshape(ngrid, ngrid)
        else:
            c = grad_log_prob(grid)
            particles_t = torch.tensor(slice_particles[slice_ind])
            particles_grad = grad_log_prob(particles_t)
            particles_phi, dists_sq = phi(particles_t, particles_grad, dlog_lh=particles_grad)            
            # C = np.exp(grad_log_prob(grid)).reshape(ngrid, ngrid)
        print(np.shape(c))
        C = np.linalg.norm(c, axis=1)
        print(np.shape(C))
        # plt.scatter(X[slice_ind], Y[slice_ind], c=C, s=1.5)
        # plt.contour(X[slice_ind].reshape(ngrid, ngrid), Y[slice_ind].reshape(ngrid, ngrid), C.reshape(ngrid,ngrid), num_levels, linewidths=1)
        # xlim = ax_limits[0]
        # ylim = ax_limits[1]
        ax.scatter(slice_grids[slice_ind][0,:], slice_grids[slice_ind][1, :], slice_grids[slice_ind][2, :], s=2, c=particles_log)
        ax.scatter(slice_particles[slice_ind][:, 0], slice_particles[slice_ind][:, 1], slice_particles[slice_ind][:, 2], s=4, c='r')
        ax.quiver(slice_particles[slice_ind][:, 0], slice_particles[slice_ind][:, 1], slice_particles[slice_ind][:, 2], particles_phi[:,0], particles_phi[:,1], particles_phi[:,2])

        # plt.plot(slice_particles[slice_ind][:, 0], slice_particles[slice_ind][:, 1], 'ro', markersize=3)
        # plt.axis('equal')
        # plt.colorbar()

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
    ax.set_xlim(ax_limits[0][0], ax_limits[0][1])
    ax.set_ylim(ax_limits[1][0], ax_limits[1][1])
    ax.set_zlim(ax_limits[2][0], ax_limits[2][1])
    plt.savefig(save_path)
    plt.show() 


def create_movie_2D_slices(particle_hist, log_prob, save_path="/tmp/stein_movie.mp4", ax_limits=[[-4, 4],[4, 4]], to_numpy=False,
        kernel_base_type=None, opt=None, num_particles=None, eps=None,):

    k_type = kernel_base_type,
    if kernel_base_type == 'RBF_Anisotropic':
        k_type = 'RBF_H'

    case_name = '{}-{} (np = {}, eps = {})'.format(opt, k_type, num_particles, eps,)

    fig = plt.figure(figsize=(15,5))
    # ax = plt.gca()
    plt.suptitle(case_name + '\n' + str(0) + '$ ^{th}$ iteration')

    ngrid = 100

    X = []
    Y = []
    Z = []
    # C = []
    slice_grids = []
    slice_particles = []
    plot_particles = []


    x = np.linspace(ax_limits[0][0], ax_limits[0][1], ngrid)
    y = np.linspace(ax_limits[1][0], ax_limits[1][1], ngrid)
    z = np.linspace(ax_limits[2][0], ax_limits[2][1], ngrid)
    x_slice = np.array([(ax_limits[0][1] - ax_limits[0][0])//2 + ax_limits[0][0]])
    y_slice = np.array([(ax_limits[1][1] - ax_limits[1][0])//2 + ax_limits[1][0]])
    z_slice = np.array([(ax_limits[2][1] - ax_limits[2][0])//2 + ax_limits[2][0]])

    # slice 1, xy plane
    X1, Y1, Z1 = np.meshgrid(x, y, z_slice)
    X += [X1]; Y += [Y1]; Z += [Z1]
    slice_grids += [np.vstack((np.ndarray.flatten(X1), np.ndarray.flatten(Y1),np.ndarray.flatten(Z1)))]

    new_particle_hist = []
    for time_ind in range(len(particle_hist)):
        particles = particle_hist[time_ind]
        particles = particles[np.where(np.logical_and(particles[:,2] >= z_slice[0], particles[:,2] < z_slice[0] + slice_buffer))[0], 0:2]  
        new_particle_hist += [particles]    
    slice_particles += [new_particle_hist]

    # slice 2, xz plane
    X2, Y2, Z2 = np.meshgrid(x, y_slice, z)
    X += [X2]; Y += [Z2]; Z += [Y2]
    slice_grids += [np.vstack((np.ndarray.flatten(X2), np.ndarray.flatten(Y2),np.ndarray.flatten(Z2)))]

    new_particle_hist = []
    for time_ind in range(len(particle_hist)):
        particles = particle_hist[time_ind]
        particles = particles[np.where(np.logical_and(particles[:,1] >= y_slice[0], particles[:,1] < y_slice[0] + slice_buffer))[0], 0:3:2]  
        new_particle_hist += [particles]    
    slice_particles += [new_particle_hist]    

    # slice 3, yz plane
    X3, Y3, Z3 = np.meshgrid(x_slice, y, z)
    X += [Y3]; Y += [Z3]; Z += [X3]
    slice_grids += [np.vstack((np.ndarray.flatten(X3), np.ndarray.flatten(Y3),np.ndarray.flatten(Z3)))]

    new_particle_hist = []
    for time_ind in range(len(particle_hist)):
        particles = particle_hist[time_ind]
        particles = particles[np.where(np.logical_and(particles[:,0] >= x_slice[0], particles[:,0] < x_slice[0] + slice_buffer))[0], 1:]  
        new_particle_hist += [particles]    
    slice_particles += [new_particle_hist]

    for slice_ind in range(len(X)):
        plt.subplot(1,len(X),slice_ind+1)
        if to_numpy:
            grid = torch.from_numpy(slice_grids[slice_ind])
            c = log_prob(grid.t()).cpu().numpy()
            # C = c.reshape(ngrid, ngrid)
            C = np.exp(c).reshape(ngrid, ngrid)
        else:
            C = np.exp(log_prob(slice_grids[slice_ind]),).reshape(ngrid, ngrid)
            # C = log_prob(slice_grids[slice_ind]).reshape(ngrid, ngrid)

        plt.contourf(X[slice_ind].reshape(ngrid, ngrid), Y[slice_ind].reshape(ngrid, ngrid), C, num_levels, vmax=1, vmin=0)
        # xlim = ax_limits[0]
        # ylim = ax_limits[1]
        plt.axis('equal')
        # plt.colorbar()
        p_start =  slice_particles[slice_ind][0]
        particles = plt.plot(p_start[:, 0], p_start[:, 1], 'ro', markersize=3)
        plot_particles += [particles]
        n_iter = len(slice_particles[slice_ind])

    def _init():  # only required for blitting to give a clean slate.
        # ax.set_title(str(0) + '$ ^{th}$ iteration')
        plt.suptitle(case_name + '\n' + str(0) + '$ ^{th}$ iteration')
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        return plot_particles

    def _animate(i):
        # ax.set_title(str(i) + '$ ^{th}$ iteration')
        plt.suptitle(case_name + '\n' + str(i) + '$ ^{th}$ iteration')
        for slice_ind in range(len(X)):
            plt.subplot(1, len(X), slice_ind+1)

            pos = slice_particles[slice_ind][i]
            plot_particles[slice_ind][0].set_xdata(pos[:, 0])
            plot_particles[slice_ind][0].set_ydata(pos[:, 1])
        return plot_particles

    ani = animation.FuncAnimation(fig, _animate, frames=n_iter, init_func=_init, interval=200,)

    ani.save(save_path)
    plt.show()


def create_movie_3D(particle_hist, log_prob, save_path="/tmp/stein_movie.mp4", ax_limits=[[-4, 4],[4, 4]], to_numpy=False,
        kernel_base_type=None, opt=None, num_particles=None, eps=None,):

    k_type = kernel_base_type,
    if kernel_base_type == 'RBF_Anisotropic':
        k_type = 'RBF_H'

    case_name = '{}-{} (np = {}, eps = {})'.format(opt, k_type, num_particles, eps,)

    fig = plt.figure(figsize=(10,10))
    # ax = plt.gca()
    plt.suptitle(case_name + '\n' + str(0) + '$ ^{th}$ iteration')

    ngrid = 100

    X = []
    Y = []
    Z = []
    # C = []
    slice_grids = []
    slice_particles = []
    plot_particles = []

    particle_hist = np.array(particle_hist)
    # print(np.shape(particle_hist))
    # print(np.shape(particle_hist)[0])
    # print(np.shape(particle_hist)[1])
    # print(np.shape(particle_hist)[2])
    # print(particle_hist)

    ax = fig.add_subplot(projection='3d')
    # print(np.shape(particle_hist)[1])
    opacities = np.linspace(0.2, 1.0, np.shape(particle_hist)[0])
    for particle_ind in range(np.shape(particle_hist)[1]):
        # print(np.shape(particle_hist[:,particle_ind,0]))
        ax.scatter(particle_hist[:,particle_ind,0], particle_hist[:,particle_ind,1], particle_hist[:,particle_ind,2], s=4*opacities,)
        ax.plot(particle_hist[:,particle_ind,0], particle_hist[:,particle_ind,1], particle_hist[:,particle_ind,2], linewidth=0.5)
    ax.set_xlim(ax_limits[0][0], ax_limits[0][1])
    ax.set_ylim(ax_limits[1][0], ax_limits[1][1])
    ax.set_zlim(ax_limits[2][0], ax_limits[2][1])
    
    # def _init():  # only required for blitting to give a clean slate.
    #     # ax.set_title(str(0) + '$ ^{th}$ iteration')
    #     plt.suptitle(case_name + '\n' + str(0) + '$ ^{th}$ iteration')
    #     # plt.xlim(xlim)
    #     # plt.ylim(ylim)
    #     return plot_particles

    # def _animate(i):
    #     # ax.set_title(str(i) + '$ ^{th}$ iteration')
    #     plt.suptitle(case_name + '\n' + str(i) + '$ ^{th}$ iteration')
    #     for slice_ind in range(len(X)):
    #         plt.subplot(1, len(X), slice_ind+1)

    #         pos = slice_particles[slice_ind][i]
    #         plot_particles[slice_ind][0].set_xdata(pos[:, 0])
    #         plot_particles[slice_ind][0].set_ydata(pos[:, 1])
    #     return plot_particles

    # ani = animation.FuncAnimation(fig, _animate, frames=n_iter, init_func=_init, interval=100,)

    # ani.save(save_path)
    plt.show()