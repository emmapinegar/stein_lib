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
                (-80, 80, -80, 80), #area [min1, max1, min2, max2]
                2,
                None,
                0.2, #gamma
                ),

            }

        return parameters[case]

    base_path =  Path(__file__).resolve().parents[2]

    # Settings
    # dtype = pt.float32
    device = pt.device("cpu")
    # pt.set_default_dtype(pt.FloatTensor)
    pt.set_default_dtype(pt.float64)
    # dataset =  'kitti1'
    dataset = 'remind'
    save_path = Path("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/")   # an be None
    save_iter = 1
    plot_iter = 1
    #device = pt.device("cuda:0") # Uncomment this to run on GPU

    # Read the file
    (fn_train,
    cell_resolution,
    cell_max_min,
    skip,
    thresh,
    gamma,) = load_parameters(dataset)

    #read data
    # g = pd.read_csv(fn_train, delimiter=',').values

    transform = np.loadtxt(fn_train, max_rows=4)

    obspoints = np.loadtxt(fn_train, skiprows=4)
    obspoints = np.transpose(obspoints)
    obspoints = np.concatenate((obspoints, np.ones((1,np.shape(obspoints)[1]))))

    limits = np.array([[0,0,0,1], [obspoints[0,0], 0, 0, 1], [obspoints[0,0], obspoints[1,0], 0, 1], [obspoints[0,0], 0, obspoints[2,0], 1], [0, obspoints[1,0], 0, 1], [0, obspoints[1,0], obspoints[2,0], 1], [0, 0, obspoints[2,0], 1], [obspoints[0,0], obspoints[1,0], obspoints[2,0], 1]])
    print(limits)
    obspoints = obspoints[:,1:]
    obspoints = np.matmul(transform, obspoints)

    min_voxel = np.matmul(transform, np.transpose(limits))
    print(min_voxel)
    print(min_voxel[0,:])
    # cell_max_min = (np.min(min_voxel[0,:]), np.max(min_voxel[0,:]), np.min(min_voxel[1,:]), np.max(min_voxel[1,:]), np.min(min_voxel[2,:]), np.max(min_voxel[2,:]))
    # cell_max_min = (np.min(obspoints[0,:]) - 5, np.max(obspoints[0,:]) + 5, np.min(obspoints[1,:]) - 5, np.max(obspoints[1,:]) + 5, np.min(obspoints[2,:]) - 5, np.max(obspoints[2,:]) + 5)


    obsarr = np.load('./../steerable-needle-planner/scripts/envs/ReMIND_obstacles_001.npy')
    print(np.shape(obsarr))
    print(obsarr)

    obs_inds = np.where(obsarr == 1)
    free_inds = np.where(obsarr == 0)
    print(np.shape(obs_inds))
    obspoints = np.concatenate((obs_inds, np.ones((1,np.shape(obs_inds)[1]))))
    freepoints = np.concatenate((free_inds, np.ones((1,np.shape(free_inds)[1]))))
    print(np.shape(obspoints))
    obspoints = np.matmul(transform, obspoints)
    freepoints = np.matmul(transform, freepoints)

    cell_max_min = (np.min(freepoints[0,:]) - 1, np.max(freepoints[0,:]) + 1, np.min(freepoints[1,:]) - 1, np.max(freepoints[1,:]) + 1, np.min(freepoints[2,:]) - 1, np.max(freepoints[2,:]) + 1)
    freepoints[3,:] = 0

    limit_inds = np.where(np.logical_and(np.logical_and(obspoints[2,:] >= cell_max_min[4] - 5, obspoints[2,:] < cell_max_min[5] + 5),np.logical_and(np.logical_and(obspoints[1,:] >= cell_max_min[2] - 5, obspoints[1,:] < cell_max_min[3] + 5),np.logical_and(obspoints[0,:] >= cell_max_min[0] - 5, obspoints[0,:] < cell_max_min[1] + 5))))[0]
    obspoints = obspoints[:,limit_inds]

    points = np.concatenate((np.transpose(obspoints), np.transpose(freepoints)))
    print(np.shape(points))
    # shuffle_inds = np.arange(np.shape(points)[0])
    # np.random.shuffle(shuffle_inds)
    # print(shuffle_inds)
    # print(np.shape(shuffle_inds))
    # points = points[shuffle_inds, :]
    # print(np.shape(points))

    print(np.shape(points))
    print(cell_max_min)
    g = points


    print('shapes:', np.shape(g))
    g = pt.tensor(g)
    X_train = g[:, 0:3]
    Y_train = g[:, 3].reshape(-1, 1)


    min_t = (round(cell_max_min[5]) + 5 - round(cell_max_min[4]))//2 + round(cell_max_min[4])
    max_t = 1 #round(cell_max_min[5]) + 5 - min_t
    print(max_t)
    print(skip)
    for ith_scan in range(0, max_t, skip):

        # extract data points of the ith scan
        # ith_scan_indx = X_train[:, 0] == ith_scan
        ith_scan_indx = np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip)
        print_str = f"{ith_scan}th scan: N={pt.sum(ith_scan_indx)}"
        # ith_scan_indx = np.where(np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip))
        X_new = X_train[ith_scan_indx, :]
        y_new = Y_train[ith_scan_indx]

        X, y = X_new, y_new
        if ith_scan == 0:
            # get all data for the first scan and initialize the model
            X, y = X_new, y_new
            bhm_mdl = BHM_PYTORCH(
                gamma=gamma,
                grid=None,
                cell_resolution=cell_resolution,
                cell_max_min=cell_max_min,
                X=X[:,0:2],
                nIter=1,
            )


        print(print_str)

        # Fit the model
        t1 = time.time()
        bhm_mdl.fit(X[:,0:2], y)
        t2 = time.time()

        q_resolution = 1
        if ith_scan % plot_iter == 0:
            ones_ = np.where(np.logical_and(X[:,2] >= ith_scan + min_t + skip//2, X[:,2] < ith_scan + min_t + skip//2 + 1))
            print(np.shape(ones_))        
            Xq = X[ones_[0],0:2]
            yq = bhm_mdl.predict(Xq)
            yq = yq.cpu().numpy()
            y_diff = y[ones_].reshape(-1,) - yq.reshape(-1,)
            # print(f"Fit time: {t2 - t1:.2f} Pred time: {t4 - t3:.2f} iter time: {t4 - t1:.2f} \tPlotting...\n")

            Xq = Xq.cpu().numpy()

            pl.figure(figsize=(18,5))
            pl.subplot(131)

            # ones_ = np.where(y ==1)
            # pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
            pl.scatter(X[ones_, 0], X[ones_, 1], c=y[ones_], cmap='jet', s=5, vmin=0, vmax=1)
            pl.axis('equal')
            pl.title('Laser hit points at t={}'.format(np.unique(ith_scan + min_t + skip//2)))
            pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
            pl.subplot(132)
            pl.title('SBHM at t={}'.format(np.unique(ith_scan + min_t + skip//2)))
            # pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=10, marker='8',edgecolors='')
            pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1)

            pl.colorbar()
            pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])

            pl.subplot(133)
            pl.scatter(Xq[:, 0], Xq[:, 1], c=y_diff, cmap='jet', s=5, vmin=-1, vmax=1)
            pl.colorbar()
            pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
            pl.savefig(os.path.abspath('./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_2D_{:03d}.png'.format(ith_scan)), bbox_inches='tight')
            pl.close()

        if save_path is not None and \
        ith_scan % save_iter == 0:
            print('Saving map...')
            filename = 'bhm_{}_2D_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
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
                (5, 5, 5), #hinge point resolution
                (-80, 80, -80, 80, -80, 80), #area [min1, max1, min2, max2]
                20000,
                None,
                0.13, #gamma
                ),

            }

        return parameters[case]

    base_path =  Path(__file__).resolve().parents[2]


    pt.set_default_dtype(pt.float64)

    dataset = 'remind'
    save_path = Path("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/")   # an be None
    save_iter = 25
    plot_iter = 10
    q_resolution = 1.5
    # colormap = 'inferno'

    # if (pt.cuda.is_available()):
    #     print("cuda is available!!")
    #     device = pt.device("cuda:0") 
    # else:
    #     print("cuda is NOT available!!")
    #     device = pt.device("cpu")

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
    hinge_point_buffer = 2*cell_resolution[0]
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

    bhm_mdl = BHM_PYTORCH(gamma=gamma, grid=None, cell_resolution=cell_resolution, X=pt.tensor(np.transpose(freepoints), device=device), nIter=1, device=device, torch_kernel_func=True) # cell_max_min=cell_max_min,
    # bhm_mdl.load("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/" + 'bhm_remind_res1_iter114.pt')

    for ith_scan in range(0, max_t):

        # extract data points of the ith scan
        # ith_scan_indx = X_train[:, 0] == ith_scan
        # ith_scan_indx = pt.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip)
        # ith_scan_indx = [ith_scan*skip:ith_scan*skip + skip]
        print_str = f"{ith_scan}th scan: N={skip}"
        # ith_scan_indx = np.where(np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip))
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
                plot_bhm_log_slice(X, y, X_train, Y_train, bhm_mdl, cell_max_min, ith_scan, q_resolution, dataset, save_path, device)

    if save_path is not None :
        print('Saving map...')
        filename = 'bhm_{}_test_log_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
        bhm_mdl.save(save_path, filename) 

    old_scan_max = ith_scan + 1

    # points_ = np.transpose(obspoints)
    # g = points_
    # g = pt.tensor(g, device=device)
    # X_train_ = g[:, 0:3]
    # Y_train_ = g[:, 3].reshape(-1, 1)
    min_t = 0
    max_t = np.shape(points)[0]
    sections = 100
    skip = max_t // sections
    print(f"points: {np.shape(points)} cell limits: {cell_max_min} num points per round: {skip}")

    for ith_scan in range(0, sections):

        # extract data points of the ith scan
        # ith_scan_indx = X_train[:, 0] == ith_scan
        # ith_scan_indx = np.logical_and(X_train_[:,2] >= ith_scan + min_t, X_train_[:,2] < ith_scan + min_t + skip)
        print_str = f"{ith_scan}th scan: N={pt.sum(ith_scan_indx)}"
        ith_scan_indx = pt.where(Y_train[ith_scan*skip + min_t:ith_scan*skip + min_t + skip] == 1)[0] + ith_scan*skip + min_t
        print(f"y: {ith_scan_indx.size()} {ith_scan_indx}")
        if ith_scan_indx.size()[0] > 0:
            X_new = X_train[ith_scan_indx, :]
            y_new = Y_train[ith_scan_indx]


            # X_new = X_train[ith_scan*skip:(ith_scan+1)*skip, :]
            # y_new = Y_train[ith_scan*skip:(ith_scan+1)*skip]        

            X, y = X_new, y_new
    
            print(print_str)

            # Fit the model
            t1 = time.time()
            bhm_mdl.fit(X, y)
            t2 = time.time()

            if save_path is not None and ith_scan % save_iter == 0:
                print('Saving map...')
                filename = 'bhm_{}_test_log_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan+old_scan_max)
                bhm_mdl.save(save_path, filename)  

        
            if ith_scan % plot_iter == 0:
                plot_bhm_log_slice(X, y, X_train, Y_train, bhm_mdl, cell_max_min, ith_scan+old_scan_max, q_resolution, dataset, save_path, device)

          

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


if __name__=='__main__':
    make_map_3D()