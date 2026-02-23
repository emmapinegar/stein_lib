"""
Getting a Bayesian Hilbert map for a brain environment for stein sampling
"""
import os
import time
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pl
from pathlib import Path
from Bayesian_Hilbert_Maps.bhmlib.BHM.pytorch.bhm_pytorch import BHM_PYTORCH

dtype = pt.float32
device = pt.device("cpu") # pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
obsarrfile = './../steerable-needle-planner/scripts/envs/ReMIND_obstacles_001.npy'

def make_map_2D():
    def getPartitions(cell_max_min, nPartx1, nPartx2):
        """
        :param cell_max_min: The size of the entire area
        :param nPartx1: How many partitions along the longitude
        :param nPartx2: How many partitions along the latitude
        :return: a list of all partitions
        """
        width = cell_max_min[1] - cell_max_min[0]
        height = cell_max_min[3] - cell_max_min[2]
        cell_max_min_segs = []
        for x in range(nPartx1):
            for y in range(nPartx2):
                seg_i = (cell_max_min[0] + width / nPartx1 * x, cell_max_min[0] + width / nPartx1 * (x + 1), \
                        cell_max_min[2] + height / nPartx2 * y, cell_max_min[2] + height / nPartx2 * (y + 1))
                cell_max_min_segs.append(seg_i)

        return cell_max_min_segs

    def load_parameters(case):
        parameters = \
            {'remind': \
                ( os.path.abspath('./remind_001_obstacles.txt'),
                (4, 4), #hinge point resolution
                [[-80, 80], [-80, 80]], #area [min1, max1, min2, max2]
                36000,
                None,
                0.15, #gamma
                ),
            }

        return parameters[case]

    # Settings
    pt.set_default_dtype(dtype)
    pt.set_default_device(device)

    dataset = 'remind'
    save_path = Path("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/")   # can be None
    save_iter = 1
    plot_iter = 1
    buffer = 15
    avg_z = 82
    q_resolution = 2

    # Read the file
    (fn_train, cell_resolution, cell_max_min, skip, thresh, gamma,) = load_parameters(dataset)

    transform = np.loadtxt(fn_train, max_rows=4)
    limits = np.loadtxt(fn_train, skiprows=4, max_rows=1)
    # obspoints = np.transpose(obspoints)
    # # obspoints = np.concatenate((obspoints, np.ones((1,np.shape(obspoints)[1]))))
    # limits = np.array([[0,0,0,1], [obspoints[0,0], 0, 0, 1], [obspoints[0,0], obspoints[1,0], 0, 1], [obspoints[0,0], 0, obspoints[2,0], 1], [0, obspoints[1,0], 0, 1], [0, obspoints[1,0], obspoints[2,0], 1], [0, 0, obspoints[2,0], 1], [obspoints[0,0], obspoints[1,0], obspoints[2,0], 1]])

    obsarr = np.load(obsarrfile)
    obs_inds = np.where(obsarr == 0)
    free_inds = np.where(obsarr == 1)
    obspoints = np.concatenate((obs_inds, np.ones((1,np.shape(obs_inds)[1]))))
    freepoints = np.concatenate((free_inds, np.ones((1,np.shape(free_inds)[1]))))
    # obspoints = np.matmul(transform, obspoints)
    # freepoints = np.matmul(transform, freepoints)
    obspoints[3,:] = 0

    z_min = 82
    z_max = 83
    cell_max_min = [[np.min(obspoints[0,:]) - buffer, np.max(obspoints[0,:]) + buffer], [np.min(obspoints[1,:]) - buffer, np.max(obspoints[1,:]) + buffer]]
    

    limit_inds = np.where(np.logical_and(np.logical_and(freepoints[2,:] >= z_min, freepoints[2,:] < z_max),np.logical_and(np.logical_and(freepoints[1,:] >= cell_max_min[1][0], freepoints[1,:] <= cell_max_min[1][1]),np.logical_and(freepoints[0,:] >= cell_max_min[0][0], freepoints[0,:] <= cell_max_min[0][1]))))[0]
    freepoints = freepoints[:,limit_inds]
    # limit_inds = np.where(np.logical_and(np.logical_and(obspoints[2,:] >= z_min, obspoints[2,:] < z_max),np.logical_and(np.logical_and(obspoints[1,:] >= cell_max_min[1][0], obspoints[1,:] <= cell_max_min[1][1]),np.logical_and(obspoints[0,:] >= cell_max_min[0][0], obspoints[0,:] <= cell_max_min[0][1]))))[0]
    limit_inds = np.where(np.logical_and(obspoints[2,:] >= z_min, obspoints[2,:] < z_max))[0]
    obspoints = obspoints[:,limit_inds]

    points = np.concatenate((np.transpose(obspoints), np.transpose(freepoints)))
    num_points = np.shape(points)[0]
    shuffle_inds = np.arange(num_points)
    np.random.shuffle(shuffle_inds)
    points = points[shuffle_inds, :]

    g = pt.tensor(points, dtype=dtype)
    X_train = g[:, 0:2]
    Y_train = g[:, 3].reshape(-1, 1)


    min_t = 0
    max_t = num_points//skip + 1

    print(f"limits: {limits} {cell_max_min} num points: {num_points}")

    bhm_mdl = BHM_PYTORCH(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=None, nIter=1, torch_kernel_func=True, device=device)
    for ith_scan in range(0, max_t):

        # extract data points of the ith scan
        print_str = f"{ith_scan}th scan: N={skip}"
        X_new = X_train[ith_scan*skip:ith_scan*skip+skip, :]
        y_new = Y_train[ith_scan*skip:ith_scan*skip+skip]

        X, y = X_new, y_new

        print(print_str)

        # Fit the model
        t1 = time.time()
        bhm_mdl.fit(X, y)
        t2 = time.time()

        if ith_scan % plot_iter == 0:
            img_name = f"./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_2D_{ith_scan:03d}.pdf"
            plot_bhm_all_slices(X, y, bhm_mdl, cell_max_min, q_resolution, avg_z, img_name)

        if save_path is not None and ith_scan % save_iter == 0:
            print('Saving map...')
            filename = 'bhm_{}_2D_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
            bhm_mdl.save(save_path, filename)


def make_map_ND():
    def getPartitions(cell_max_min, nPartx1, nPartx2, nPartx3):
        """
        :param cell_max_min: The size of the entire area
        :param nPartx1: How many partitions along the longitude
        :param nPartx2: How many partitions along the latitude
        :return: a list of all partitions
        """
        width = cell_max_min[0][1] - cell_max_min[0][0]
        height = cell_max_min[1][1] - cell_max_min[1][0]
        depth = cell_max_min[2][1] - cell_max_min[2][0]
        cell_max_min_segs = []
        for x in range(nPartx1):
            for y in range(nPartx2):
                for z in range(nPartx3):
                    seg_i = (cell_max_min[0][0] + width / nPartx1 * x, cell_max_min[0][0] + width / nPartx1 * (x + 1),
                            cell_max_min[1][0] + height / nPartx2 * y, cell_max_min[1][0] + height / nPartx2 * (y + 1),
                            cell_max_min[2][0] + depth / nPartx3 * z, cell_max_min[2][0] + depth / nPartx3 * (z + 1),)
                    cell_max_min_segs.append(seg_i)

        return cell_max_min_segs

    def load_parameters(case):
        parameters = \
            {'remind': \
                ( os.path.abspath('./remind_001_obstacles.txt'),
                (6, 6, 6), #hinge point resolution
                [[-80, 80], [-80, 80], [-80, 80]], #area [min1, max1, min2, max2]
                30000,
                None,
                0.10, #gamma
                ),
            }

        return parameters[case]

    # Settings
    pt.set_default_dtype(dtype)
    pt.set_default_device(device)

    dataset = 'remind'
    save_path = Path("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/")   # can be None
    save_iter = 10
    plot_iter = 10
    buffer = 10
    avg_z = 82
    q_resolution = 2

    # Read the file
    (fn_train, cell_resolution, cell_max_min, skip, thresh, gamma,) = load_parameters(dataset)

    transform = np.loadtxt(fn_train, max_rows=4)
    limits = np.loadtxt(fn_train, skiprows=4, max_rows=1)
    # obspoints = np.transpose(obspoints)
    # obspoints = np.concatenate((obspoints, np.ones((1,np.shape(obspoints)[1]))))
    # limits = np.array([[0,0,0,1], [obspoints[0,0], 0, 0, 1], [obspoints[0,0], obspoints[1,0], 0, 1], [obspoints[0,0], 0, obspoints[2,0], 1], [0, obspoints[1,0], 0, 1], [0, obspoints[1,0], obspoints[2,0], 1], [0, 0, obspoints[2,0], 1], [obspoints[0,0], obspoints[1,0], obspoints[2,0], 1]])

    obsarr = np.load(obsarrfile)
    obs_inds = np.where(obsarr == 0)
    free_inds = np.where(obsarr == 1)
    obspoints = np.concatenate((obs_inds, np.ones((1,np.shape(obs_inds)[1]))))
    freepoints = np.concatenate((free_inds, np.ones((1,np.shape(free_inds)[1]))))
    # obspoints = np.matmul(transform, obspoints)
    # freepoints = np.matmul(transform, freepoints)
    obspoints[3,:] = 0

    cell_max_min = [[np.min(obspoints[0,:]) - buffer, np.max(obspoints[0,:]) + buffer], [np.min(obspoints[1,:]) - buffer, np.max(obspoints[1,:]) + buffer], [np.min(obspoints[2,:]) - buffer, np.max(obspoints[2,:]) + buffer]]
    

    limit_inds = np.where(np.logical_and(np.logical_and(freepoints[2,:] >= cell_max_min[2][0], freepoints[2,:] <= cell_max_min[2][1]),np.logical_and(np.logical_and(freepoints[1,:] >= cell_max_min[1][0], freepoints[1,:] <= cell_max_min[1][1]),np.logical_and(freepoints[0,:] >= cell_max_min[0][0], freepoints[0,:] <= cell_max_min[0][1]))))[0]
    freepoints = freepoints[:,limit_inds]

    points = np.concatenate((np.transpose(obspoints), np.transpose(freepoints)))
    num_points = np.shape(points)[0]
    shuffle_inds = np.arange(num_points)
    np.random.shuffle(shuffle_inds)
    points = points[shuffle_inds, :]

    g = pt.tensor(points, dtype=dtype)
    X_train = g[:, 0:3]
    Y_train = g[:, 3].reshape(-1, 1)


    min_t = 0
    max_t = num_points//skip + 1

    print(f"limits: {limits} {cell_max_min} num points: {num_points}")

    bhm_mdl = BHM_PYTORCH(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=None, nIter=1, torch_kernel_func=True, device=device)
    for ith_scan in range(0, max_t):

        # extract data points of the ith scan
        print_str = f"{ith_scan}th scan: N={skip}"
        X_new = X_train[ith_scan*skip:ith_scan*skip+skip, :]
        y_new = Y_train[ith_scan*skip:ith_scan*skip+skip]

        X, y = X_new, y_new

        print(print_str)

        # Fit the model
        t1 = time.time()
        bhm_mdl.fit(X, y)
        t2 = time.time()

        if ith_scan % plot_iter == 0:
            img_name = f"./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_3D_res{cell_resolution[0]}_gamma{gamma}_{ith_scan:03d}.pdf"
            plot_index = pt.where(pt.logical_and(X_train[:,2] >= avg_z, X_train[:,2] < avg_z + 1))[0]
            X = X_train[plot_index,:]
            y = Y_train[plot_index]
            plot_bhm_all_slices(X, y, bhm_mdl, cell_max_min, q_resolution, avg_z, img_name)

        if save_path is not None and ith_scan % save_iter == 0:
            print('Saving map...')
            filename = f"bhm_{dataset}_3D_res{cell_resolution[0]}_gamma{gamma}_iter{ith_scan:03d}.pt"
            bhm_mdl.save(save_path, filename)

    if save_path is not None:
        print('Saving map...')
        filename = f"bhm_{dataset}_3D_res{cell_resolution[0]}_gamma{gamma}_final.pt"
        bhm_mdl.save(save_path, filename)


def make_map_3D():
    def getPartitions(cell_max_min, nPartx1, nPartx2):
        """
        :param cell_max_min: The size of the entire area
        :param nPartx1: How many partitions along the longitude
        :param nPartx2: How many partitions along the latitude
        :return: a list of all partitions
        """
        width = cell_max_min[1] - cell_max_min[0]
        height = cell_max_min[3] - cell_max_min[2]
        cell_max_min_segs = []
        for x in range(nPartx1):
            for y in range(nPartx2):
                seg_i = (cell_max_min[0] + width / nPartx1 * x, cell_max_min[0] + width / nPartx1 * (x + 1), \
                        cell_max_min[2] + height / nPartx2 * y, cell_max_min[2] + height / nPartx2 * (y + 1))
                cell_max_min_segs.append(seg_i)

        return cell_max_min_segs

    def load_parameters(case):
        parameters = \
            {'remind': \
                ( os.path.abspath('./remind_001_obstacles.txt'),
                (6, 6, 6), #hinge point resolution
                (-80, 80, -80, 80, -80, 80), #area [min1, max1, min2, max2]
                32000,
                None,
                0.08, #gamma
                ),

            }

        return parameters[case]

    base_path =  Path(__file__).resolve().parents[2]


    pt.set_default_dtype(pt.float64)

    dataset = 'remind'
    save_path = Path("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/")   # an be None
    save_iter = 25
    plot_iter = 5
    q_resolution = 2
    avg_z = 82

    device = pt.device("cpu")

    # Read the file
    (fn_train, cell_resolution, cell_max_min, skip, thresh, gamma,) = load_parameters(dataset)


    transform = np.loadtxt(fn_train, max_rows=4)

    limit_points = np.loadtxt(fn_train, skiprows=4, max_rows=1)
    print(np.shape(limit_points))
    limit_points = np.transpose(limit_points).reshape(-1,1)
    print(np.shape(limit_points))
    limit_points = np.concatenate((limit_points, np.ones((1,np.shape(limit_points)[1]))))

    limits = np.array([[0,0,0,1], [limit_points[0,0], 0, 0, 1], [limit_points[0,0], limit_points[1,0], 0, 1], [limit_points[0,0], 0, limit_points[2,0], 1], [0, limit_points[1,0], 0, 1], [0, limit_points[1,0], limit_points[2,0], 1], [0, 0, limit_points[2,0], 1], [limit_points[0,0], limit_points[1,0], limit_points[2,0], 1]])
    # obspoints = obspoints[:,1:]
    # obspoints = np.matmul(transform, obspoints)


    obsarr = np.load('./../steerable-needle-planner/scripts/envs/ReMIND_obstacles_001.npy')
    # print(np.shape(obsarr))
    # print(obsarr)

    obs_inds = np.where(obsarr == 1)
    free_inds = np.where(obsarr == 0)

    obspoints = np.concatenate((obs_inds, np.ones((1,np.shape(obs_inds)[1]))))
    freepoints = np.concatenate((free_inds, np.ones((1,np.shape(free_inds)[1]))))

    # obspoints = np.matmul(transform, obspoints)
    # freepoints = np.matmul(transform, freepoints)
    hinge_point_buffer = 10 #2*cell_resolution[0]
    cell_max_min = (np.min(freepoints[0,:]) - hinge_point_buffer, np.max(freepoints[0,:]) + hinge_point_buffer, np.min(freepoints[1,:]) - hinge_point_buffer, np.max(freepoints[1,:]) + hinge_point_buffer, np.min(freepoints[2,:]) - hinge_point_buffer, np.max(freepoints[2,:]) + hinge_point_buffer)
    freepoints[3,:] = 0

    limit_inds = np.where(np.logical_and(np.logical_and(obspoints[2,:] >= cell_max_min[4], obspoints[2,:] <= cell_max_min[5]),np.logical_and(np.logical_and(obspoints[1,:] >= cell_max_min[2], obspoints[1,:] <= cell_max_min[3]),np.logical_and(obspoints[0,:] >= cell_max_min[0], obspoints[0,:] <= cell_max_min[1]))))
    limit_inds = limit_inds[0].reshape(-1,)
    obspoints = obspoints[:,limit_inds]
    print(f"x min: {np.min(obspoints[0,:]):.04f} max: {np.max(obspoints[0,:]):.04f} x min: {np.min(obspoints[1,:]):.04f} max: {np.max(obspoints[1,:]):.04f} z min: {np.min(obspoints[2,:]):.04f} max: {np.max(obspoints[2,:]):.04f} ")

    points = np.concatenate((np.transpose(obspoints), np.transpose(freepoints)))
    print(f"points: {np.shape(points)} limits: {cell_max_min}")

    shuffle_inds = np.arange(np.shape(points)[0])
    np.random.shuffle(shuffle_inds)
    points = points[shuffle_inds,:]
    print(np.shape(shuffle_inds))
    g = points
    g = pt.tensor(g, device=device)
    X_train = g[:, 0:3]
    Y_train = g[:, 3].reshape(-1, 1)


    min_t = 0 #round(cell_max_min[4])
    max_t = np.shape(points)[0]//skip + 1 #round(cell_max_min[5]) - min_t
    print(max_t)
    print(skip)
    ith_scan = 0
    # ith_scan_indx = pt.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip)
    # X = X_train[ith_scan_indx, :]

    bhm_mdl = BHM_PYTORCH(gamma=gamma, grid=None, cell_resolution=cell_resolution, X=pt.tensor(np.transpose(freepoints), device=device), nIter=1, device=device, torch_kernel_func=True, cell_max_min=cell_max_min) # cell_max_min=cell_max_min,
    # bhm_mdl.load("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/" + 'bhm_remind_res1_iter114.pt')

    for ith_scan in range(0, max_t):

        # extract data points of the ith scan
        print_str = f"{ith_scan}th scan: N={skip}"
        X_new = X_train[ith_scan*skip:ith_scan*skip + skip, :]
        y_new = Y_train[ith_scan*skip:ith_scan*skip + skip]

        X, y = X_new, y_new
        if X.size()[0] > 0:
            print(print_str)

            # Fit the model
            t1 = time.time()
            bhm_mdl.fit(X, y)
            t2 = time.time()

            if save_path is not None and ith_scan % save_iter == 0:
                print('Saving map...')
                filename = 'bhm_{}_test_log_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
                bhm_mdl.save(save_path, filename)

            if ith_scan % plot_iter == 0:
                # plot_bhm_log_slice(X, y, X_train, Y_train, bhm_mdl, cell_max_min, ith_scan, q_resolution, dataset, save_path, device)
                img_name = f"./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_test_log_{ith_scan:03d}.png"
                plot_bhm_all_slices(X, y, bhm_mdl, cell_max_min, q_resolution, avg_z, img_name)


    if save_path is not None :
        print('Saving map...')
        filename = 'bhm_{}_test_log_res{}_final.pt'.format(dataset, q_resolution)
        bhm_mdl.save(save_path, filename) 



def plot_bhm_slice(X, y, X_train, Y_train, bhm_mdl, cell_max_min, plot_index, q_resolution, dataset, save_path, device, colormap='plasma'):
    avg_z = pt.round(pt.mean(X[:,2]))
    print(f"Plotting bhm at z = {avg_z}")
    ith_scan_indx_ = pt.logical_and(X_train[:,2] >= avg_z , X_train[:,2] < avg_z + 1)
    X = X_train[ith_scan_indx_, :]
    y = Y_train[ith_scan_indx_]
    plot_train_points = True
    avg_z = avg_z.cpu().numpy()
    if plot_train_points:
        Xq = X_train[ith_scan_indx_, :]
    else:
        # query the model
        buffer = 6
        xx, yy, zz= np.meshgrid(np.arange(cell_max_min[0]-buffer, cell_max_min[1]+buffer, q_resolution),
                             np.arange(cell_max_min[2]-buffer, cell_max_min[3]+buffer, q_resolution),
                             np.arange(avg_z, avg_z+1, q_resolution))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis], zz.ravel()[:, np.newaxis]))
        Xq = pt.tensor(grid, device=device)
    # Predict
    t3 = time.time()
    yq = bhm_mdl.predict(Xq)
    t4 = time.time()

    Xq = Xq.cpu().numpy()
    yq = yq.cpu().numpy()
    X = X.cpu().numpy()
    y = y.cpu().numpy()

    # print(f"Fit time: {t2 - t1:.2f} Pred time: {t4 - t3:.2f} iter time: {t4 - t1:.2f} \tPlotting...\n")

    pl.figure(figsize=(18,5))
    pl.subplot(131)
    pl.scatter(X[:, 0], X[:, 1], c=y[:], cmap=colormap, s=5, vmin=0, vmax=1)
    pl.axis('equal')
    pl.title('Laser hit points at t={}'.format(avg_z))
    pl.colorbar()
    pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])
    
    pl.subplot(132)
    pl.title('SBHM at t={}'.format(avg_z))
    pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap=colormap, s=5, vmin=0, vmax=1)
    pl.axis('equal')
    pl.colorbar()
    pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])

    if plot_train_points:
        y_diff = y.reshape(-1,) - yq.reshape(-1,)
        pl.subplot(133)
        pl.scatter(Xq[:, 0], Xq[:, 1], c=y_diff, cmap=colormap, s=5, vmin=-1, vmax=1)
        pl.axis('equal')
        pl.colorbar()
        pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])

    pl.savefig(os.path.abspath('./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_test_{:03d}.png'.format(plot_index)), bbox_inches='tight')
    pl.close()



def plot_bhm_log_slice(X, y, X_train, Y_train, bhm_mdl, cell_max_min, plot_index, q_resolution, dataset, save_path, device, colormap='plasma'):
    avg_z = pt.round(pt.mean(X[:,2])) - 2
    print(f"Plotting bhm at z = {avg_z}")
    ith_scan_indx_ = pt.logical_and(X_train[:,2] >= avg_z , X_train[:,2] < avg_z + 1)
    X = X_train[ith_scan_indx_, :]
    y = Y_train[ith_scan_indx_]
    plot_train_points = False
    avg_z = avg_z.cpu().numpy()



    X = X.cpu().numpy()
    y = y.cpu().numpy()


    # print(f"Fit time: {t2 - t1:.2f} Pred time: {t4 - t3:.2f} iter time: {t4 - t1:.2f} \tPlotting...\n")

    pl.figure(figsize=(18,5))
    pl.subplot(131)
    pl.scatter(X[:, 0], X[:, 1], c=y[:], cmap=colormap, s=5, vmin=0, vmax=1)
    pl.axis('equal')
    pl.title('Laser hit points at t={}'.format(avg_z))
    pl.colorbar()
    pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])
    
    if plot_train_points:
        X = pt.tensor(X, device=device)
    else:
        # query the model
        buffer = 2
        xx, yy, zz= np.meshgrid(np.arange(cell_max_min[0]-buffer, cell_max_min[1]+buffer, q_resolution),
                             np.arange(cell_max_min[2]-buffer, cell_max_min[3]+buffer, q_resolution),
                             np.arange(avg_z, avg_z+1, q_resolution))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis], zz.ravel()[:, np.newaxis]))
        X = pt.tensor(grid, device=device)
        print(X.size())

    # Predict
    t3 = time.time()
    y = bhm_mdl.predict(X)
    t4 = time.time()
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    
    print(f"min: {np.min(y)} max: {np.max(y)}")
    pl.subplot(132)
    pl.title('SBHM at t={}'.format(avg_z))
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, s=5, vmin=0, vmax=1)
    pl.axis('equal')
    pl.colorbar()
    pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])

    X = pt.tensor(X, device=device)
    y = bhm_mdl.log_prob_vacancy(X)
    X = X.cpu().numpy()
    y = y.cpu().numpy()
    print(f"min: {np.min(y)} max: {np.max(y)}")
    pl.subplot(133)
    pl.title('SBHM log at t={}'.format(avg_z))
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, s=5, vmin=-5, vmax=0)
    pl.axis('equal')
    pl.colorbar()
    pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])

    # if plot_train_points:
    #     y_diff = y.reshape(-1,) - yq.reshape(-1,)
    #     pl.subplot(133)
    #     pl.scatter(Xq[:, 0], Xq[:, 1], c=y_diff, cmap=colormap, s=5, vmin=-1, vmax=1)
    #     pl.axis('equal')
    #     pl.colorbar()
    #     pl.xlim([cell_max_min[0]-2, cell_max_min[1]+2]); pl.ylim([cell_max_min[2]-2, cell_max_min[3]+2])

    pl.savefig(os.path.abspath('./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_test_log_{:03d}.png'.format(plot_index)), bbox_inches='tight')
    pl.close()



def plot_bhm_all_slices(X, y, bhm_mdl, cell_max_min, q_resolution, avg_z, save_path, colormap='plasma'):
    print(f"Plotting bhm at z = {avg_z}")

    y = y.cpu().numpy()

    yq = bhm_mdl.predict(X)
    yq = yq.cpu().numpy()

    y_diff = y.reshape(-1,) - yq.reshape(-1,)

    ylog = bhm_mdl.log_prob_vacancy(X)
    ylog = ylog.cpu().numpy()

    X = X.cpu().numpy()


    pl.figure(figsize=(18,10))
    pl.subplot(231)
    pl.title(f"Ground truth at z={avg_z}")
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap, s=1, vmin=0, vmax=1)
    pl.colorbar()
    pl.axis('equal')
    # pl.xlim([cell_max_min[0][0], cell_max_min[0][1]]); pl.ylim([cell_max_min[1][0], cell_max_min[1][1]])
    

    pl.subplot(232)
    pl.title(f"SBHM at z={avg_z}")
    pl.scatter(X[:, 0], X[:, 1], c=yq, cmap=colormap, s=1, vmin=0, vmax=1)
    pl.colorbar()
    pl.axis('equal')
    # pl.xlim([cell_max_min[0][0], cell_max_min[0][1]]); pl.ylim([cell_max_min[1][0], cell_max_min[1][1]])


    pl.subplot(233)
    pl.title(f"y - yq at z={avg_z}")
    pl.scatter(X[:, 0], X[:, 1], c=y_diff, cmap=colormap, s=1, vmin=-1, vmax=1)
    pl.colorbar()
    pl.axis('equal')
    # pl.xlim([cell_max_min[0][0], cell_max_min[0][1]]); pl.ylim([cell_max_min[1][0], cell_max_min[1][1]])


    pl.subplot(234)
    pl.title(f"log(1 - yq) at z={avg_z}")
    pl.scatter(X[:, 0], X[:, 1], c=ylog, cmap=colormap, s=1, vmin=-10, vmax=0)
    pl.colorbar()
    pl.axis('equal')
    # pl.xlim([cell_max_min[0][0], cell_max_min[0][1]]); pl.ylim([cell_max_min[1][0], cell_max_min[1][1]])

    Xq = bhm_mdl.calc_plotting_grid(q_resolution, cell_max_min[0:2][:])
    Xq = pt.autograd.Variable(Xq, requires_grad=True)
    
    if len(cell_max_min) > 2:
        print(f"Xq: {Xq.size()} test: {Xq.size()[0]}")
        Xq = pt.stack((Xq[:,0].reshape(-1,1), Xq[:,1].reshape(-1,1), avg_z * pt.ones(Xq.size()[0]).reshape(-1,1)), dim=1).squeeze()
        print(f"Xq: {Xq.size()}")

    grad_analytic = bhm_mdl.grad_log_p_vacancy(Xq, sub_limits=True)
    grad_analytic = grad_analytic.detach().cpu().numpy()

    grad_numerical = bhm_mdl.numerical_grad_log_p(Xq, sub_limits=False)
    grad_numerical = grad_numerical.detach().cpu().numpy()

    Xq = Xq.detach().cpu().numpy()

    pl.subplot(235)
    pl.title(f"analytic grad(log(1 - yq)) at z={avg_z}")
    pl.quiver(Xq[:, 0], Xq[:, 1], grad_analytic[:,0], grad_analytic[:,1])
    # pl.colorbar()
    pl.axis('equal')
    # pl.xlim([cell_max_min[0][0], cell_max_min[0][1]]); pl.ylim([cell_max_min[1][0], cell_max_min[1][1]])    

    pl.subplot(236)
    pl.title(f"numerical grad(log(1 - yq)) at z={avg_z}")
    pl.quiver(Xq[:, 0], Xq[:, 1], grad_numerical[:,0], grad_numerical[:,1])
    # pl.colorbar()
    pl.axis('equal')
    # pl.xlim([cell_max_min[0][0], cell_max_min[0][1]]); pl.ylim([cell_max_min[1][0], cell_max_min[1][1]])

    pl.savefig(save_path, bbox_inches='tight')
    pl.close()


if __name__=='__main__':
    # make_map_2D()
    make_map_ND()