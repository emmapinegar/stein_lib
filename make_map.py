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
from Bayesian_Hilbert_Maps.bhmlib.BHM.pytorch.bhmtorch_cpu import BHM3D_PYTORCH, BHM2D_PYTORCH



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
            bhm_mdl = BHM2D_PYTORCH(
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
            pl.savefig(os.path.abspath('./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_{:03d}.png'.format(ith_scan)), bbox_inches='tight')
            pl.close()

        if save_path is not None and \
        ith_scan % save_iter == 0:
            print('Saving map...')
            filename = 'bhm_{}_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
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
                (4, 4, 8), #hinge point resolution
                (-80, 80, -80, 80, -80, 80), #area [min1, max1, min2, max2]
                2,
                None,
                0.1, #gamma
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
    save_iter = 25
    plot_iter = 15
    #device = pt.device("cuda:0") # Uncomment this to run on GPU

    # Read the file
    (fn_train, cell_resolution, cell_max_min, skip, thresh, gamma,) = load_parameters(dataset)

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

    # min_voxel = np.matmul(transform, np.transpose(limits))
    # print(min_voxel)
    # print(min_voxel[0,:])
    # cell_max_min = (np.min(min_voxel[0,:]), np.max(min_voxel[0,:]), np.min(min_voxel[1,:]), np.max(min_voxel[1,:]), np.min(min_voxel[2,:]), np.max(min_voxel[2,:]))
    # cell_max_min = (np.min(obspoints[0,:]) - 5, np.max(obspoints[0,:]) + 5, np.min(obspoints[1,:]) - 5, np.max(obspoints[1,:]) + 5, np.min(obspoints[2,:]) - 5, np.max(obspoints[2,:]) + 5)


    obsarr = np.load('./../steerable-needle-planner/scripts/envs/ReMIND_obstacles_001.npy')
    # print(np.shape(obsarr))
    # print(obsarr)

    obs_inds = np.where(obsarr == 1)
    free_inds = np.where(obsarr == 0)
    # print(np.shape(obs_inds))
    obspoints = np.concatenate((obs_inds, np.ones((1,np.shape(obs_inds)[1]))))
    freepoints = np.concatenate((free_inds, np.ones((1,np.shape(free_inds)[1]))))
    # print(np.shape(obspoints))
    obspoints = np.matmul(transform, obspoints)
    freepoints = np.matmul(transform, freepoints)
    hinge_point_buffer = -cell_resolution[0]//2
    cell_max_min = (np.min(freepoints[0,:]) - hinge_point_buffer, np.max(freepoints[0,:]) + hinge_point_buffer, np.min(freepoints[1,:]) - hinge_point_buffer, np.max(freepoints[1,:]) + hinge_point_buffer, np.min(freepoints[2,:]) - hinge_point_buffer, np.max(freepoints[2,:]) + hinge_point_buffer)
    freepoints[3,:] = 0

    limit_inds = np.where(np.logical_and(np.logical_and(obspoints[2,:] >= cell_max_min[4] - 5, obspoints[2,:] < cell_max_min[5] + 5),np.logical_and(np.logical_and(obspoints[1,:] >= cell_max_min[2] - 5, obspoints[1,:] < cell_max_min[3] + 5),np.logical_and(obspoints[0,:] >= cell_max_min[0] - 5, obspoints[0,:] < cell_max_min[1] + 5))))
    print(limit_inds)
    limit_inds = limit_inds[0].reshape(-1,)
    print(f"before: {np.shape(obspoints)} after: {np.shape(limit_inds)}")
    obspoints = obspoints[:,limit_inds]
    print(f"x min: {np.min(obspoints[0,:])} max: {np.max(obspoints[0,:])} x min: {np.min(obspoints[1,:])} max: {np.max(obspoints[1,:])} z min: {np.min(obspoints[2,:])} max: {np.max(obspoints[2,:])} ")

    points = np.concatenate((np.transpose(obspoints), np.transpose(freepoints)))
    print(np.shape(points))
    # shuffle_inds = np.arange(np.shape(points)[0])
    # np.random.shuffle(shuffle_inds)
    # print(shuffle_inds)
    # print(np.shape(shuffle_inds))
    # points = points[shuffle_inds, :]
    # print(np.shape(points))

    print(cell_max_min)
    g = points


    print('shapes:', np.shape(g))
    g = pt.tensor(g)
    X_train = g[:, 0:3]
    Y_train = g[:, 3].reshape(-1, 1)


    min_t = round(cell_max_min[4])
    max_t = round(cell_max_min[5]) + 5 - min_t
    print(max_t)
    print(skip)
    ith_scan = 0
    ith_scan_indx = np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip)
    X = X_train[ith_scan_indx, :]

    bhm_mdl = BHM3D_PYTORCH(gamma=gamma,grid=None,cell_resolution=cell_resolution,cell_max_min=cell_max_min, X=X, nIter=1,)
    bhm_mdl.load("./Bayesian_Hilbert_Maps/bhmlib/Outputs/saved_models/" + 'bhm_remind_res1_iter114.pt')

    # for ith_scan in range(0, max_t, skip):

    #     # extract data points of the ith scan
    #     # ith_scan_indx = X_train[:, 0] == ith_scan
    #     ith_scan_indx = np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip)
    #     print_str = f"{ith_scan}th scan: N={pt.sum(ith_scan_indx)}"
    #     # ith_scan_indx = np.where(np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip))
    #     X_new = X_train[ith_scan_indx, :]
    #     y_new = Y_train[ith_scan_indx]

    #     X, y = X_new, y_new
    #     if ith_scan == 0:
    #         # get all data for the first scan and initialize the model
    #         X, y = X_new, y_new
    #         bhm_mdl = BHM3D_PYTORCH(
    #             gamma=gamma,
    #             grid=None,
    #             cell_resolution=cell_resolution,
    #             cell_max_min=cell_max_min,
    #             X=X,
    #             nIter=1,
    #         )
    #     # else:
    #     #     # information filtering
    #     #     q_new = bhm_mdl.predict(X_new).reshape(-1, 1)
    #     #     print(q_new)
    #     #     print(y_new)
    #     #     print(pt.absolute(q_new - y_new))
    #     #     info_val_indx = pt.absolute(q_new - y_new) > thresh
    #     #     info_val_indx = info_val_indx.flatten()
    #     #     X, y = X_new[info_val_indx, :], y_new[info_val_indx]
    #     #     print_str += f" {X.shape[0]/X_new.shape[0]*100:.2f}% points were used"

    #     print(print_str)

    #     # Fit the model
    #     t1 = time.time()
    #     bhm_mdl.fit(X, y)
    #     t2 = time.time()

    #     q_resolution = 2
    #     if ith_scan % plot_iter == 0:
    #         # query the model
    #         # xx, yy, zz= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
    #         #                      np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution),
    #         #                      np.arange(ith_scan + min_t + skip//2, ith_scan + min_t + skip//2 +1, q_resolution))
    #         # grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis], zz.ravel()[:, np.newaxis]))
    #         # Xq = pt.tensor(grid)
    #         # # Predict
    #         # t3 = time.time()
    #         # yq = bhm_mdl.predict(Xq)
    #         # t4 = time.time()

    #         ones_ = np.where(np.logical_and(X[:,2] >= ith_scan + min_t + skip//2, X[:,2] < ith_scan + min_t + skip//2 + 1))
    #         print(np.shape(ones_))        
    #         Xq = X[ones_[0],:]
    #         yq = bhm_mdl.predict(Xq)
    #         yq = yq.cpu().numpy()
    #         y_diff = y[ones_].reshape(-1,) - yq.reshape(-1,)
    #         # print(f"Fit time: {t2 - t1:.2f} Pred time: {t4 - t3:.2f} iter time: {t4 - t1:.2f} \tPlotting...\n")

    #         Xq = Xq.cpu().numpy()

    #         pl.figure(figsize=(18,5))
    #         pl.subplot(131)

    #         # ones_ = np.where(y ==1)
    #         # pl.scatter(X[ones_, 0], X[ones_, 1], c='r', cmap='jet', s=5, edgecolors='')
    #         pl.scatter(X[ones_, 0], X[ones_, 1], c=y[ones_], cmap='jet', s=5, vmin=0, vmax=1)
    #         pl.axis('equal')
    #         pl.title('Laser hit points at t={}'.format(np.unique(ith_scan + min_t + skip//2)))
    #         pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
    #         pl.subplot(132)
    #         pl.title('SBHM at t={}'.format(np.unique(ith_scan + min_t + skip//2)))
    #         # pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=10, marker='8',edgecolors='')
    #         pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1)

    #         pl.colorbar()
    #         pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])

    #         pl.subplot(133)
    #         pl.scatter(Xq[:, 0], Xq[:, 1], c=y_diff, cmap='jet', s=5, vmin=-1, vmax=1)
    #         pl.colorbar()
    #         pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
    #         pl.savefig(os.path.abspath('./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_test_{:03d}.png'.format(ith_scan)), bbox_inches='tight')
    #         pl.close()

    #     if save_path is not None and \
    #     ith_scan % save_iter == 0:
    #         print('Saving map...')
    #         filename = 'bhm_{}_test_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan)
    #         bhm_mdl.save(save_path, filename)


    # shuffle_inds = np.random.shuffle(np.arange(np.shape(points)[0]))
    points_ = np.transpose(obspoints)
    # points = points[shuffle_inds, :]

    
    g = points_
    g = pt.tensor(g)
    X_train_ = g[:, 0:3]
    Y_train_ = g[:, 3].reshape(-1, 1)
    max_t = np.shape(points_)[0]
    sections = 50
    skip = np.shape(points_)[0] // sections
    print(f"points: {np.shape(points_)} cell limits: {cell_max_min} num points per round: {skip}")

    for ith_scan in range(0, sections):

        # extract data points of the ith scan
        # ith_scan_indx = X_train[:, 0] == ith_scan
        # ith_scan_indx = np.logical_and(X_train_[:,2] >= ith_scan + min_t, X_train_[:,2] < ith_scan + min_t + skip)
        print_str = f"{ith_scan}th scan: N={pt.sum(ith_scan_indx)}"
        # ith_scan_indx = np.where(np.logical_and(X_train[:,2] >= ith_scan + min_t, X_train[:,2] < ith_scan + min_t + skip))
        X_new = X_train_[ith_scan*skip:(ith_scan+1)*skip, :]
        y_new = Y_train_[ith_scan*skip:(ith_scan+1)*skip]

        X, y = X_new, y_new
        # if ith_scan == 0:
        #     # get all data for the first scan and initialize the model
        #     X, y = X_new, y_new
        #     bhm_mdl = BHM3D_PYTORCH(
        #         gamma=gamma,
        #         grid=None,
        #         cell_resolution=cell_resolution,
        #         cell_max_min=cell_max_min,
        #         X=X,
        #         nIter=1,
        #     )
        # else:
        #     # information filtering
        #     q_new = bhm_mdl.predict(X_new).reshape(-1, 1)
        #     print(q_new)
        #     print(y_new)
        #     print(pt.absolute(q_new - y_new))
        #     info_val_indx = pt.absolute(q_new - y_new) > thresh
        #     info_val_indx = info_val_indx.flatten()
        #     X, y = X_new[info_val_indx, :], y_new[info_val_indx]
        #     print_str += f" {X.shape[0]/X_new.shape[0]*100:.2f}% points were used"

        print(print_str)

        # Fit the model
        t1 = time.time()
        bhm_mdl.fit(X, y)
        t2 = time.time()

        q_resolution = 1
        if save_path is not None and \
        ith_scan % save_iter == 0:
            print('Saving map...')
            filename = 'bhm_{}_test_res{}_iter{:03d}.pt'.format(dataset, q_resolution, ith_scan+114)
            bhm_mdl.save(save_path, filename)  

        
        if ith_scan % plot_iter == 0:
            # query the model
            # xx, yy, zz= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
            #                      np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution),
            #                      np.arange(ith_scan + min_t + skip//2, ith_scan + min_t + skip//2 +1, q_resolution))
            # grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis], zz.ravel()[:, np.newaxis]))
            # Xq = pt.tensor(grid)
            # # Predict
            # t3 = time.time()
            # yq = bhm_mdl.predict(Xq)
            # t4 = time.time()
            avg_z = np.average(X[:,2].detach().cpu().numpy())
            print(avg_z)
            ith_scan_indx_ = np.logical_and(X_train[:,2] >= avg_z , X_train[:,2] < avg_z + 2)

            X = X_train[ith_scan_indx_, :]
            y = Y_train[ith_scan_indx_]
            ones_ = np.where(np.logical_and(X[:,2] >= avg_z, X[:,2] < avg_z + 1))
            print(np.shape(ones_)) 
            if np.shape(ones_)[1]:       
                Xq = X[ones_[0],:]
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
                pl.title('Laser hit points at t={}'.format(avg_z))
                pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
                pl.subplot(132)
                pl.title('SBHM at t={}'.format(avg_z))
                # pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=10, marker='8',edgecolors='')
                pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1)

                pl.colorbar()
                pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])

                pl.subplot(133)
                pl.scatter(Xq[:, 0], Xq[:, 1], c=y_diff, cmap='jet', s=5, vmin=-1, vmax=1)
                pl.colorbar()
                pl.xlim([cell_max_min[0], cell_max_min[1]]); pl.ylim([cell_max_min[2], cell_max_min[3]])
                pl.savefig(os.path.abspath('./Bayesian_Hilbert_Maps/bhmlib/Outputs/images/remind_test_{:03d}.png'.format(ith_scan)), bbox_inches='tight')
                pl.close()

          

    if save_path is not None :
        print('Saving map...')
        filename = 'bhm_{}_test_res{}_final.pt'.format(dataset, q_resolution)
        bhm_mdl.save(save_path, filename) 


if __name__=='__main__':
    make_map_3D()