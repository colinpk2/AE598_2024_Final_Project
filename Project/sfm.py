import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rerun as rr
from scipy.spatial.transform import Rotation

import symforce
# symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from symforce.opt.noise_models import PseudoHuberNoiseModel
from symforce.opt.noise_models import BarronNoiseModel
import sym

import csv


"""
FUNCTIONS THAT YOU NEED TO MODIFY
"""

"""
Functions for two-view reconstruction.
"""
def write_csv(file, a):
    with open(file, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in a:
            writer.writerow({'x': i[0], 'y': i[1], 'z': i[2]})
def KNN(pts, idx, k):
    '''
    Find k-nearest neighbors
    param pts: list of points to search (including the target point)
    param idx: index of point to find neighbors
    param k: number of neighbors
    return: returns a tuple of 2 lists: list of pts corresponding to k neighbors and list of indexes in pts 
    '''

    pi = pts[idx]
    nbs = []
    dist = []
    for i in range(len(pts)):
        dist.append([np.linalg.norm(pi-pts[i]),i])
    dist = np.array(dist)

    dist_sort = dist[dist[:,0].argsort()]

    knn = dist_sort[1:k+1]
    for nb in knn:
        pt_idx = int(nb[1])
        nbs.append(pts[pt_idx])
    
    return nbs, knn[:,1]

def local_norm(pts):
    p_ = np.mean(pts, axis=0)
    k = len(pts)
    M = np.zeros((len(p_), len(p_)))
    for pt in pts:
        M += (1/k)*np.outer((pt-p_),(pt-p_))
    eigval, eigvec = np.linalg.eig(M)
    n = eigvec[np.argmin(eigval)]
    n/=np.linalg.norm(n)
    return n, p_

def project_outliers(pts, k):
    b = pts.copy()
    for i in range(len(b)):
        q, idxs = KNN(b, i, k)
        n, p_ = local_norm(q)
        disp = np.dot((b[i]-p_), n)
        b[i] -= n*disp
        # for j in range(len(q)):
        #     idx = int(idxs[j])
        #     disp = np.dot((b[idx]-b[i]), n)
        #     b[idx] -= n*disp
    return b
def avg_dist(pi,q):
    '''
    computes density of point and neighbors
    param pi: point to find density at
    param q: list of k-nearest neighbors
    return density at point pi
    '''
    
    sum_ = 0
    for pt in q:
        sum_+=np.linalg.norm(pt-pi)
    return sum_/len(q)

def p_outlier(pts, idx, k, thresh):
    '''
    computes local density of point pi to find probability of being an outlier
    param pi: point to find local density at
    param q: list of k-nearest neighbors
    return outlier probability of point pi and if it is likely an outlier
    '''
    pi = pts[idx]
    q, idxs = KNN(pts,idx,k)
    di = avg_dist(pi,q)
    # print("density: ", di)
    sum_ = 0
    for pt in q:
        sum_+=np.e**(-np.linalg.norm(pt-pi)/di)
    p = 1-sum_/len(q)
    return p, p<thresh*di

def remove_outliers(a, k=10, thresh=0.1):
    a_ = []
    for i in range(len(a)):
        p, outlier = p_outlier(a,i,k, thresh)
        if not outlier:
            a_.append(a[i])
    return np.array(a_)


def apply_transform(R_inB_ofA, p_inB_ofA, p_inA):
    p_inB = np.row_stack([
        (R_inB_ofA @ p_inA_i + p_inB_ofA) for p_inA_i in p_inA
    ])
    return p_inB

def project(K, R_inB_ofA, p_inB_ofA, p_inA, warn=True):
    p_inB = apply_transform(R_inB_ofA, p_inB_ofA, p_inA)
    if not np.all(p_inB[:, 2] > 0):
        if warn:
            print('WARNING: non-positive depths')
    q = np.row_stack([K @ p_inB_i / p_inB_i[2] for p_inB_i in p_inB])
    return q[:, 0:2]

def projection_error(K, R_inB_ofA, p_inB_ofA, p_inA, b, warn=True):
    b_pred = project(K, R_inB_ofA, p_inB_ofA, p_inA, warn=warn)
    return np.linalg.norm(b_pred - b, axis=1)

def twoview_triangulate(alpha, beta, R_inB_ofA, p_inB_ofA):
    # INPUTS (alpha, beta, R_inB_ofA, p_inB_ofA)
    #  alpha        normalized coordinates of points in image A
    #  beta         normalized coordinates of points in image B
    #  R_inB_ofA    orientation of frame A in frame B
    #  p_inB_ofA    position of frame A in frame B
    #
    # OUTPUTS (p_inA, p_inB, mask)
    #  p_inA        triangulated points in frame A
    #  p_inB        triangulated points in frame B
    #  mask         1d array of length equal to number of triangulated points,
    #               with a "1" for each point that has positive depth in both
    #               frames and with a "0" otherwise
    p_inA = []
    mask = []
    for i in range(len(alpha)):
        # Create matrices corresponding to 
        b_skew = skew(beta[i])
        b1 = -b_skew@p_inB_ofA
        b2 = np.zeros(alpha[i].shape[-1])
        b = np.vstack([b1, b2]).reshape((6))

        a1 = b_skew@R_inB_ofA
        a2 = skew(alpha[i])
        a = np.vstack([a1, a2])
        p_inAi = np.linalg.pinv(a) @ b
        p_inA.append(p_inAi)

        # Calculate p inBi
        p_inBi = R_inB_ofA @ p_inAi + p_inB_ofA

        # Check if the points PinA and PinB have the correct orentation
        if p_inAi[-1] > 0 and p_inBi[-1] > 0:
            mask.append(1)

    return np.array(p_inA), np.sum(np.array(mask))

def getE(a, b, K, rng, threshold=1e-3, num_iters=1000):
    """
    INPUTS
    
    -- required --
    a is n x 2
    b is n x 2
    K is 3 x 3
    rng is a random number generator
    
    -- optional --
    threshold is the max error (e.g., epipolar or sampson) for inliers
    num_iters is the number of RANSAC iterations
    
    OUTPUTS

    E is 3 x 3
    num_inliers is a scalar
    mask is length n (truthy value for each match that is an inlier, falsy otherwise)
    """

    a_til = np.column_stack([a, np.ones(a.shape[0],)])
    b_til = np.column_stack([b, np.ones(b.shape[0],)])
    alpha = np.array([np.linalg.inv(K)@ai for ai in a_til])
    beta = np.array([np.linalg.inv(K)@bi for bi in b_til])

    E_best = None
    count_best = 0
    mask_best = None
    e_hat = skew(np.array([0,0,1]))
    for j in range(num_iters):
        # sample 8 matching points
        I = rng.choice(len(alpha), 8, replace=False)
        A = np.array([alpha[i] for i in I])
        B = np.array([beta[i] for i in I])
        # estimate Essential matrix
        M = np.row_stack([np.kron(A[i], B[i]).T for i in range(len(A))])
        Udp, Sdp, VdpT = np.linalg.svd(M)
        Edp = VdpT[-1].reshape(3,3).T

        # Normalize the Essential matrix
        Ep = Edp * np.sqrt(2)/np.linalg.norm(Edp)
        # Correct Essential matrix
        Up, Sp, VpT = np.linalg.svd(Ep)

        U = np.column_stack([Up.T[0], Up.T[1], np.linalg.det(Up)*Up.T[2]])
        V = np.column_stack([VpT[0], VpT[1], np.linalg.det(VpT.T)*VpT[2]])
        E = U@np.diag([1,1,0])@(V.T)
        # Calculate epipolar distances
        count = 0
        mask = []
        for i in range(len(alpha)):
            dAi = np.abs(beta[i].T@E@alpha[i]/np.linalg.norm(e_hat@E@alpha[i]))
            dBi = np.abs(alpha[i].T@E.T@beta[i]/np.linalg.norm(e_hat@E.T@beta[i]))
            # validate if epipolar distance is below prescribed threshold
            if dAi < threshold and dBi < threshold:
                count += 1
                mask.append(1)
            else:
                mask.append(0)
        # Replace previous values if current estimate provides better epipolar distances
        if count > count_best:
            count_best = count
            mask_best = np.array(mask)
            E_best = E

    num_inlier = count_best
    return E_best, num_inlier, mask_best

def decomposeE(a, b, K, E):
    """
    INPUTS
    
    -- required --
    a is n x 2
    b is n x 2
    K is 3 x 3
    E is 3 x 3
    
    OUTPUTS

    R_inB_ofA is 3 x 3
    p_inB_ofA is length 3
    p_inA is n x 3
    """
    K_inv = np.linalg.inv(K)
    a_til = np.column_stack([a, np.ones(a.shape[0],)])
    b_til = np.column_stack([b, np.ones(b.shape[0],)])
    alpha = np.array([K_inv@ai for ai in a_til])
    beta = np.array([K_inv@bi for bi in b_til])

    # Normalize the Essential matrix

    # Correct Essential matrix
    U, S, VT = np.linalg.svd(E)
    u1,u2,u3 = U.T
    v1,v2,v3 = VT
    DU = np.linalg.det(U)
    DV = np.linalg.det(VT.T)

    U = np.column_stack([u1,u2,DU*u3])
    V = np.column_stack([v1,v2,DV*v3])

    W = np.array([[0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]])
    R_inBofA_a, p_inBofA_a = [U@W.T@V.T, U.T[2]]
    R_inBofA_b, p_inBofA_b = [U@W@V.T, -U.T[2]]
    #Solutions Corresponding to negative E
    R_inBofA_c, p_inBofA_c = [U@W.T@V.T, -U.T[2]]
    R_inBofA_d, p_inBofA_d = [U@W@V.T, U.T[2]]

    R = np.array([R_inBofA_a, R_inBofA_b, R_inBofA_c, R_inBofA_d])
    P = np.array([p_inBofA_a, p_inBofA_b, p_inBofA_c, p_inBofA_d])
    sums = []
    p = []
    # Perform Triangulation and return the Essential matrix
    for i in range(len(R)):
        p_A, mask = twoview_triangulate(alpha, beta, R[i], P[i])
        p.append(p_A)
        sums.append(mask)
    max_index = sums.index(max(sums))
    p_inA = p[max_index]

    return R[max_index], P[max_index], p_inA


"""
Functions for resectioning and triangulation.
"""

def resection(p_inA, c, K, rng, threshold=1., num_iters=1000):
    """
    INPUTS
    
    -- required --
    p_inA is n x 3
    c is n x 2
    K is 3 x 3
    rng is a random number generator
    
    -- optional --
    threshold is the max error (e.g., reprojection) for inliers
    num_iters is the number of RANSAC iterations
    
    OUTPUTS

    R_inC_ofA is 3 x 3
    p_inC_ofA is length 3
    num_inliers is a scalar
    mask is length n (truthy value for each match that is an inlier, falsy otherwise)
    """
    mask = 0
    R_inC_ofA_best = None
    p_inC_ofA_best = None
    count_best = 0
    mask_best = None
    c_til = np.column_stack([c, np.ones(c.shape[0],)])
    gamma = np.array([np.linalg.inv(K)@ci for ci in c_til])
    
    for j in range(num_iters):
        I = rng.choice(len(gamma), 6, replace=False)
        G = np.array([gamma[i] for i in I])
        PA = np.array([p_inA[i] for i in I])
        M = np.row_stack([np.block([np.kron(PA[i], skew(G[i])), skew(G[i])]) for i in range(len(I))])


        # Find solution
        U, S, VT = np.linalg.svd(M)
        x = VT[-1]
        R_inC_ofA_temp = x[:9].reshape(3,3).T
        

        # Correct solution
        # - Should have the right scale
        x_inC_ofA = x[:3]

        x /= np.linalg.norm(x_inC_ofA)

        # - Should be right-handed
        x *= np.linalg.det(R_inC_ofA_temp) 
        # - Should be a rotation matrix
        # R_inC_ofA_temp = x[:9].reshape(3,3).T

        U, S, VT = np.linalg.svd(R_inC_ofA_temp)
        
        R_inC_ofA = np.linalg.det(U@VT)*U@VT
        p_inC_ofA = x[-3:]

        e = projection_error(K, R_inC_ofA, p_inC_ofA, p_inA, c)
        count = 0
        mask = []
        for i in e:
            if i < threshold:
                count += 1
                mask.append(1)
            else:
                mask.append(0)
        if count > count_best:
            count_best = count
            mask_best = np.array(mask)
            R_inC_ofA_best, p_inC_ofA_best = R_inC_ofA, p_inC_ofA
    
    num_inliers = sum(mask_best)
    
    return R_inC_ofA_best, p_inC_ofA_best, num_inliers, mask_best

# def triangulate(track, views, K):
#     """
#     INPUTS
#     track is **one** track of matches to triangulate
#     views is the list of all views
#     K is 3 x 3
    
#     OUTPUTS

#     p_inA is length 3
#     """


#     # Compute relative pose
#     R_inB_ofC = R_inB_ofA @ R_inC_ofA.T
#     p_inB_ofC = p_inB_ofA - R_inB_ofA @ R_inC_ofA.T @ p_inC_ofA

#     # Compute normalized image coordinates
#     c_til = np.column_stack([c, np.ones(c.shape[0],)])
#     b_til = np.column_stack([b, np.ones(b.shape[0],)])
#     beta = np.array([np.linalg.inv(K)@bi for bi in b_til])
#     gamma = np.array([np.linalg.inv(K)@ci for ci in c_til])
#     # Compute depths in both images and verify (e.g., assert) all are positive
#     L = []
#     for i in range(len(c)):
#         u = skew(beta[i])@R_inB_ofC@gamma[i]
#         v = -skew(beta[i])@p_inB_ofC
#         L_ci = u.T@v/(u.T@u) 
#         L_bi = np.array([0,0,1])@(L_ci*R_inB_ofC@gamma[i] + p_inB_ofC)
#         if L_ci < 0 or L_bi < 0:
#             return None, None
#         else:
#             L.append([L_ci, L_bi])
#     # Compute points in frame C
#     p_inC = np.array([L[i][0]*gamma[i] for i in range(len(L))])

#     # Compute points in frame A
#     p_inA = np.array([R_inC_ofA.T @ (p_inC[i] - p_inC_ofA)for i in range(len(L))])

#     return p_inA

def triangulate(track, views, K):
    """
    INPUTS
    track is **one** track of matches to triangulate
    views is the list of all views
    K is 3 x 3
    
    OUTPUTS

    p_inA is length 3
    """
    M1 = []
    M2 = []
    Kinv = np.linalg.inv(K)
    for i in track['matches']:
        view = i['view_id']
        feature = i['feature_id']
        R_ofA = views[view]['R_inB_ofA']
        p_ofA = views[view]['p_inB_ofA']
        pt2d = views[view]['pts'][feature]['pt2d']
        beta = Kinv@np.array([*pt2d, 1])
        Bhat = skew(beta)
        M1.append(-Bhat@p_ofA)
        M2.append(Bhat@R_ofA)
    M1 = np.block(M1)
    M2 = np.vstack(M2)
    p_inA = np.linalg.pinv(M2)@M1

    return p_inA

"""
Functions for optimization
"""

def get_optimizer(views, tracks, K):
    """
    Returns a symforce optimizer and a set of initial_values that
    allow you to solve the bundle adjustment problem corresponding
    to the given views and tracks.
    """

    # Create data structures
    initial_values = Values(
        fx=K[0,0],                                                              # <-- FIXME
        fy=K[1,1],                                                              # <-- FIXME
        cx=K[0,2],                                                              # <-- FIXME
        cy=K[1,2],                                                              # <-- FIXME
        tracks=[],
        epsilon=sym.epsilon,
    )
    optimized_keys = []
    factors = []

    # For each view that has a pose estimate, add this pose estimate as an initial
    # value and (if not the first view) as an optimized key.
    print(f'Iterate over {len(views)} views:')
    for i, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
            continue
        
        initial_values[f'T_inB{i}_ofA'] = sym.Pose3(
            R=sym.Rot3.from_rotation_matrix(view['R_inB_ofA']),                                          # <-- FIXME
            t=view['p_inB_ofA'],                                                  # <-- FIXME
        )

        if i > 0:
            optimized_keys.append(f'T_inB{i}_ofA')
            print(f' T_inB{i}_ofA has an initial value and is an optimized key')
        else:
            print(f' T_inB{i}_ofA has an initial value')

    # Add a factor to fix the scale (the relative distance between frames B0 and
    # B1 will be something close to one).
    print(f'T_inB{1}_ofA has an sf_scale_residual factor')
    factors = [
        Factor(
            residual=sf_scale_residual,
            keys=[
                f'T_inB{1}_ofA',
                'epsilon',
            ],
        )
    ]

    # For each valid track, add its 3d point as an initial value and an optimized
    # key, and then, for each match in this track, add its 2d point as an initial
    # value and add a factor to penalize reprojection error.
    print(f'Iterate over {len(tracks)} tracks:')
    for i_track, track in enumerate(tracks):
        if not track['valid']:
            continue
        
        if (i_track == 0) or (i_track == len(tracks) - 1):
            print(f' track {i_track}:')
            print(f'  track_{i_track}_p_inA has an initial value and is an optimized key')
        elif (i_track == 1):
            print('\n ...\n')
        initial_values[f'track_{i_track}_p_inA'] = track['p_inA']              # <-- FIXME
        optimized_keys.append(f'track_{i_track}_p_inA')

        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            if (i_track == 0) or (i_track == len(tracks) - 1):
                print(f'  track_{i_track}_b_{view_id} has an initial value and an sf_projection_residual factor')
            initial_values[f'track_{i_track}_b_{view_id}'] = views[view_id]['pts'][feature_id]['pt2d']    # <-- FIXME
            factors.append(Factor(
                residual=sf_projection_residual,
                keys=[
                    f'T_inB{view_id}_ofA',
                    f'track_{i_track}_p_inA',
                    f'track_{i_track}_b_{view_id}',
                    'fx',
                    'fy',
                    'cx',
                    'cy',
                    'epsilon',
                ],
            ))
    
    # Create optimizer
    optimizer = Optimizer(
        factors=factors,
        optimized_keys=optimized_keys,
        debug_stats=True,
        params=Optimizer.Params(
            iterations=100,
            use_diagonal_damping=True,
            lambda_down_factor=0.1,
            lambda_up_factor=5.,
            early_exit_min_reduction=1e-4,
        ),
    )

    return optimizer, initial_values






"""
FUNCTIONS THAT YOU DO NOT NEED TO MODIFY
"""

def myprint(M):
    """
    Prints either a scalar or a numpy array with four digits after the
    decimal and with consistent spacing so it is easier to read.
    """
    if M.shape:
        with np.printoptions(linewidth=150, formatter={'float': lambda x: f'{x:10.4f}'}):
            print(M)
    else:
        print(f'{M:10.4f}')


def apply_transform(R_inB_ofA, p_inB_ofA, p_inA):
    """
    Returns p_inB.
    """
    p_inB = np.row_stack([
        (R_inB_ofA @ p_inA_i + p_inB_ofA) for p_inA_i in p_inA
    ])
    return p_inB

def project(K, R_inB_ofA, p_inB_ofA, p_inA, warn=False):
    """
    Returns the projection (n x 2) of p_inA (n x 3) into an image
    taken from a calibrated camera at frame B.
    """
    p_inB = apply_transform(R_inB_ofA, p_inB_ofA, p_inA)
    if not np.all(p_inB[:, 2] > 0):
        if warn:
            print('WARNING: non-positive depths')
    q = np.row_stack([K @ p_inB_i / p_inB_i[2] for p_inB_i in p_inB])
    return q[:, 0:2]

def projection_error(K, R_inB_ofA, p_inB_ofA, p_inA, b, warn=False):
    """
    Returns the projection error - the difference between the projection
    of p_inA into an image taken from a calibrated camera at frame B, and
    the observed image coordinates b.

    If p_inA is n x 3 and b is n x 2, then the error is length n.

    If p_inA is length 3 and b is length 2, then the error is a scalar.
    """
    if len(b.shape) == 1:
        b_pred = project(K, R_inB_ofA, p_inB_ofA, np.reshape(p_inA, (1, -1)), warn=warn).flatten()
        return np.linalg.norm(b_pred - b)
    elif len(b.shape) == 2:
        b_pred = project(K, R_inB_ofA, p_inB_ofA, p_inA, warn=warn)
        return np.linalg.norm(b_pred - b, axis=1)
    else:
        raise Exception(f'b has bad shape: {b.shape}')
    
def skew(v):
    """
    Returns the 3 x 3 skew-symmetric matrix that corresponds
    to v, a vector of length 3.
    """
    assert(type(v) == np.ndarray)
    assert(v.shape == (3,))
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])

def show_results(views, tracks, K, show_pose_estimates=True, show_reprojection_errors=True):
    """
    Show the pose estimates (text) and reprojection errors (both text and plots)
    corresponding to views and tracks.
    """

    # Pose estimates
    if show_pose_estimates:
        print('POSE ESTIMATES')
        for i_view, view in enumerate(views):
            if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
                continue

            R_inA_ofB = view['R_inB_ofA'].T
            p_inA_ofB = - view['R_inB_ofA'].T @ view['p_inB_ofA']
            s = f' [R_inA_ofB{i_view}, p_inA_ofB{i_view}] = '
            s += np.array2string(
                np.column_stack([R_inA_ofB, p_inA_ofB]),
                formatter={'float': lambda x: f'{x:10.4f}'},
                prefix=s,
            )
            print(s)
    
    # Get reprojection errors
    e = [[] for view in views]
    for track in tracks:
        if not track['valid']:
            continue
        
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            e[view_id].append(
                projection_error(
                    K,
                    view['R_inB_ofA'],
                    view['p_inB_ofA'],
                    track['p_inA'],
                    view['pts'][feature_id]['pt2d'],
                )
            )
    
    # Show reprojection errors
    if show_reprojection_errors:
        print('\nREPROJECTION ERRORS')

        # Text
        for i_view, (e_i, view) in enumerate(zip(e, views)):
            if len(e_i) == 0:
                assert((view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None))
                continue

            assert(not ((view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None)))
            print(f' Image {i_view:2d} ({len(e_i):5d} points) : (mean, std, max, min) =' + \
                  f' ({np.mean(e_i):6.2f}, {np.std(e_i):6.2f}, {np.max(e_i):6.2f}, {np.min(e_i):6.2f})')
        
        # Figure
        bins = np.linspace(0, 5, 50)
        counts = [len(e_i) for e_i in e if len(e_i) > 0]
        max_count = np.max(counts)
        num_views = len(counts)
        num_cols = 3
        num_rows = (num_views // num_cols) + 1
        fig = plt.figure(figsize=(num_cols * 4, num_rows * 2), tight_layout=True)
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Error [pixels]")
        plt.ylabel("Pixels [count]")

        index = 0
        for i_view, e_i in enumerate(e):
            if len(e_i) == 0:
                continue

            index += 1
            ax = fig.add_subplot(num_rows, num_cols, index)
            ax.hist(e_i, bins, label=f'Image {i_view}')
            ax.set_xlim([bins[0], bins[-1]])
            ax.set_ylim([0, max_count])
            ax.legend()
            ax.grid()

        plt.show()

def is_duplicate_match(mA, matches):
    """
    Returns True if the match mA shares either a queryIdx or a trainIdx
    with any match in the list matches, False otherwise.
    """
    for mB in matches:
        if (mA.queryIdx == mB.queryIdx) or (mA.trainIdx == mB.trainIdx):
            return True
    return False

def get_good_matches(descA, descB, threshold=0.5):
    """
    Returns a list of matches that satisfy the ratio test with
    a given threshold. Makes sure these matches are unique - no
    feature in either image will be part of more than one match.
    """

    # Create a brute-force matcher
    bf = cv2.BFMatcher(
        normType=cv2.NORM_L2,
        crossCheck=False,       # <-- IMPORTANT - must be False for kNN matching
    )

    # Find the two best matches between descriptors
    matches = bf.knnMatch(descA, descB, k=2)

    # Find the subset of good matches
    good_matches = []
    for m, n in matches:
        if m.distance / n.distance < threshold:
            good_matches.append(m)

    # Sort the good matches by distance (smallest first)
    sorted_matches = sorted(good_matches, key = lambda m: m.distance)

    # VERY IMPORTANT - Eliminate duplicate matches
    unique_matches = []
    for sorted_match in sorted_matches:
        if not is_duplicate_match(sorted_match, unique_matches):
            unique_matches.append(sorted_match)
    
    # Return good matches, sorted by distance (smallest first)
    return unique_matches

def remove_track(track_to_remove, tracks):
    """
    Removes the given track from a list of tracks. (This is non-trivial
    because tracks are dictionaries for which we don't have a built-in notion
    of equality. An alternative would be to create a class for tracks, and to
    define an equality operator for that class.)
    """

    i_to_remove = [i_track for i_track, track in enumerate(tracks) if track is track_to_remove]
    assert(len(i_to_remove) == 1)
    del tracks[i_to_remove[0]]

def add_next_view(views, tracks, K, matching_threshold=0.5):
    """
    Updates views and tracks after matching the next available image.
    """
    iC = None
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
            iC = i_view
            break
    
    if iC is None:
        raise Exception('all views have been added')
    
    print(f'ADDING VIEW {iC}')

    for iB in range(0, iC):

        viewB = views[iB]
        viewC = views[iC]

        # Get good matches
        matches = get_good_matches(viewB['desc'], viewC['desc'], threshold=matching_threshold)
        num_matches = len(matches)
        print('')
        print(f'matching image {iB} with image {iC} with threshold {matching_threshold}:')
        print(f' {num_matches:4d} good matches found')
        
        num_created = 0
        num_added_to_B = 0
        num_added_to_C = 0
        num_merged_trivial = 0
        num_merged_nontrivial = 0
        num_invalid = 0

        for m in matches:
            # Get the corresponding points
            ptB = viewB['pts'][m.queryIdx]
            ptC = viewC['pts'][m.trainIdx]

            # Get the corresponding tracks (if they exist)
            trackB = ptB['track']
            trackC = ptC['track']

            # Create, extend, or merge tracks
            if (trackC is None) and (trackB is None):
                num_created += 1
                # Create a new track
                track = {
                    'p_inA': None,
                    'valid': True,
                    'matches': [
                        {'view_id': iB, 'feature_id': m.queryIdx},
                        {'view_id': iC, 'feature_id': m.trainIdx},
                    ],
                }
                tracks.append(track)
                ptB['track'] = track
                ptC['track'] = track
            elif (trackC is not None) and (trackB is None):
                num_added_to_C += 1
                # Add ptB to trackC
                track = trackC
                trackC['matches'].append({'view_id': iB, 'feature_id': m.queryIdx})
                ptB['track'] = track
            elif (trackC is None) and (trackB is not None):
                num_added_to_B += 1
                # Add ptC to trackB
                track = trackB
                trackB['matches'].append({'view_id': iC, 'feature_id': m.trainIdx})
                ptC['track'] = track
            elif (trackC is not None) and (trackB is not None):
                
                # If trackB and trackC are identical, then nothing further needs to be done
                if trackB is trackC:
                    num_merged_trivial += 1
                    s = f'       trivial merge - ({iB:2d}, {m.queryIdx:4d}) ({iC:2d}, {m.trainIdx:4d}) - '
                    for track_m in trackB['matches']:
                        s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                    print(s)
                    continue
                
                num_merged_nontrivial += 1

                s = f'       non-trivial merge - matches ({iB:2d}, {m.queryIdx:4d}) ({iC:2d}, {m.trainIdx:4d})\n'
                s += '                           track one '
                for track_m in trackB['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                s += '\n'
                s += '                           track two '
                for track_m in trackC['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                print(s)

                # Merge
                # - triangulated point
                if trackB['p_inA'] is None:
                    if trackC['p_inA'] is None:
                        p_inA = None
                    else:
                        p_inA = trackC['p_inA']
                else:
                    if trackC['p_inA'] is None:
                        p_inA = trackB['p_inA']
                    else:
                        # FIXME: May want to re-triangulate rather than averaging
                        p_inA = 0.5 * (trackB['p_inA'] + trackC['p_inA'])
                trackC['p_inA'] = p_inA
                # - valid
                valid = trackB['valid'] and trackC['valid']
                trackC['valid'] = valid
                # - matches and points
                for trackB_m in trackB['matches']:
                    # ONLY add match to track if it isn't already there (duplicate matches
                    # can happen if we closed a loop)
                    if (trackB_m['view_id'] != iC) or (trackB_m['feature_id'] != m.trainIdx):
                        trackC['matches'].append(trackB_m)
                    
                    # ALWAYS update track of point corresponding to match
                    views[trackB_m['view_id']]['pts'][trackB_m['feature_id']]['track'] = trackC
                track = trackC

                # Remove the leftover track
                remove_track(trackB, tracks)
                
                s = '                           => '
                for track_m in track['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                s += f' - {str(track["valid"])}'
                print(s)

            else:
                raise Exception('Should never get here!')
            
            # Check if track is self-consistent (i.e., check that it does not contain two
            # matches from the same image)
            view_ids = [track_m['view_id'] for track_m in track['matches']]
            if len(set(view_ids)) != len(view_ids):
                num_invalid += 1
                track['valid'] = False
                s = f'       FOUND INCONSISTENT - '
                for track_m in track['matches']:
                    s += f'({track_m["view_id"]:2d}, {track_m["feature_id"]:4d}) '
                print(s)

        print(f' {num_created:4d} tracks created')
        print(f' {num_added_to_C:4d} tracks in C extended with point in B')
        print(f' {num_added_to_B:4d} tracks in B extended with point in C')
        print(f' {num_merged_trivial:4d} tracks merged (trivial)')
        print(f' {num_merged_trivial:4d} tracks merged (non-trivial)')
        print(f' {num_invalid:4d} inconsistent tracks')
    
    return iC

def store_results(views, tracks, K, result, max_reprojection_err=1.):
    """
    Updates views and tracks given the result from optimization.
    """

    # Get pose estimates
    num_views = 0
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view['p_inB_ofA'] is None):
            continue

        T_inB_ofA = result.optimized_values[f'T_inB{i_view}_ofA'].to_homogenous_matrix()
        R_inB_ofA = T_inB_ofA[0:3, 0:3]
        p_inB_ofA = T_inB_ofA[0:3, 3]
        view['R_inB_ofA'] = R_inB_ofA
        view['p_inB_ofA'] = p_inB_ofA
        num_views += 1

    # Get position estimates
    num_invalid_old = 0
    num_invalid_new = 0
    num_valid = 0
    for i_track, track in enumerate(tracks):
        if not track['valid']:
            num_invalid_old += 1
            continue
        
        p_inA = result.optimized_values[f'track_{i_track}_p_inA']
        track['p_inA'] = p_inA
        valid = track['valid']
        for match in track['matches']:
            view_id = match['view_id']
            feature_id = match['feature_id']
            view = views[view_id]
            R_inB_ofA = view['R_inB_ofA']
            p_inB_ofA = view['p_inB_ofA']
            p_inB = R_inB_ofA @ p_inA + p_inB_ofA
            b = views[view_id]['pts'][feature_id]['pt2d']
            e = projection_error(K, R_inB_ofA, p_inB_ofA, p_inA, b)
            
            # Remain valid if depth is positive
            valid = valid and p_inB[2] > 0.
            
            # Remain valid if reprojection error is below threshold
            valid = valid and e < max_reprojection_err
        
        track['valid'] = valid
        if valid:
            num_valid += 1
        else:
            num_invalid_new += 1

    # Show diagnostics
    print(f'{num_views:6d} views with updated pose estimate')
    print(f'{num_valid:6d} valid tracks with updated position estimate')
    print(f'{num_invalid_old:6d} already invalid tracks')
    print(f'{num_invalid_new:6d} newly invalid tracks')

def get_match_with_view_id(matches, view_id):
    """
    From a list of matches, return the one with the given view_id.
    """
    for match in matches:
        if match['view_id'] == view_id:
            return match
    return None

def get_pt2d_from_match(views, match):
    """
    Get image coordinates of the given match.
    """
    return views[match['view_id']]['pts'][match['feature_id']]['pt2d']

def sf_projection(
    T_inC_ofW: sf.Pose3,
    p_inW: sf.V3,
    fx: sf.Scalar,
    fy: sf.Scalar,
    cx: sf.Scalar,
    cy: sf.Scalar,
    epsilon: sf.Scalar,
) -> sf.V2:
    """
    Symbolic function that projects a point into an image. (If the depth
    of this point is non-positive, then the projection will be pushed far
    away from the image center.)
    """
    p_inC = T_inC_ofW * p_inW
    z = sf.Max(p_inC[2], epsilon)   # <-- if depth is non-positive, then projection
                                    #     will be pushed far away from image center
    return sf.V2(
        fx * (p_inC[0] / z) + cx,
        fy * (p_inC[1] / z) + cy,
    )

def sf_projection_residual(
    T_inC_ofW: sf.Pose3,
    p_inW: sf.V3,
    q: sf.V2,
    fx: sf.Scalar,
    fy: sf.Scalar,
    cx: sf.Scalar,
    cy: sf.Scalar,
    epsilon: sf.Scalar,  
) -> sf.V2:
    """
    Symbolic function that computes the difference between a projected point
    and an image point. This error is "whitened" so that taking its norm will
    be equivalent to applying a robust loss function (Geman-McClure).
    """
    q_proj = sf_projection(T_inC_ofW, p_inW, fx, fy, cx, cy, epsilon)
    unwhitened_residual = sf.V2(q_proj - q)
    
    noise_model = BarronNoiseModel(
        alpha=-2,
        scalar_information=1,
        x_epsilon=epsilon,
        alpha_epsilon=epsilon,
    )
    
    return noise_model.whiten_norm(unwhitened_residual)

def sf_scale_residual(
    T_inC_ofW: sf.Pose3,
    epsilon: sf.Scalar,
) -> sf.V1:
    """
    Symbolic function that computes the relative distance between two frames.
    """
    return sf.V1(T_inC_ofW.t.norm() - 1)

def copy_results(views, tracks):
    """
    Returns a deep copy of views and tracks so that you can store intermediate results.
    """
    # Copy views (except for references to tracks)
    views_copy = []
    for view in views:
        # Copy the view
        view_copy = {
            'frame_id': view['frame_id'],
            'img': view['img'].copy(),
            'R_inB_ofA': None if view['R_inB_ofA'] is None else view['R_inB_ofA'].copy(),
            'p_inB_ofA': None if view['p_inB_ofA'] is None else view['p_inB_ofA'].copy(),
            'pts': [],
            'desc': view['desc'].copy(),
        }

        # Copy all points in the view
        for pt in view['pts']:
            view_copy['pts'].append({
                'pt2d': pt['pt2d'].copy(),
                'track': None,
            })
        
        # Append view copy to list of views
        views_copy.append(view_copy)

    # Copy tracks
    tracks_copy = []
    for track in tracks:
        # Copy the track
        track_copy = {
            'p_inA': None if track['p_inA'] is None else track['p_inA'].copy(),
            'valid': track['valid'],
            'matches': [],
        }

        # Copy all matches in the track
        for match in track['matches']:
            track_copy['matches'].append({
                'view_id': match['view_id'],
                'feature_id': match['feature_id'],
            })
        
        # Append track copy to list of tracks
        tracks_copy.append(track_copy)
    
    # Insert references to tracks into views
    for track in tracks_copy:
        for match in track['matches']:
            pt = views_copy[match['view_id']]['pts'][match['feature_id']]
            if pt['track'] is None:
                pt['track'] = track
            else:
                assert(pt['track'] is track)
    
    return views_copy, tracks_copy

def visualize_results(views, tracks, K, fps):
    """
    Visualizes results with rerun.
    """
    
    # Get index of current view
    i_view = len(views)
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view ['p_inB_ofA'] is None):
            break
    i_view -= 1
    
    # Set current time
    rr.set_time_seconds('stable_time', float(views[i_view]['frame_id']) / float(fps))

    # Show triangulated points
    p_inA = np.array([track['p_inA'] for track in tracks if track['valid'] and track['p_inA'] is not None])
    rr.log(
        '/results/triangulated_points',
        rr.Points3D(p_inA),
    )

    # Show camera frames
    for i_view, view in enumerate(views):
        if (view['R_inB_ofA'] is None) or (view ['p_inB_ofA'] is None):
            continue
        
        R_inB_ofA = view['R_inB_ofA'].copy()
        p_inB_ofA = view['p_inB_ofA'].copy()
        R_inA_ofB = R_inB_ofA.T
        p_inA_ofB = -R_inB_ofA.T @ p_inB_ofA
        rr.log(
            f'/results/camera_{i_view}',
            rr.Transform3D(
                translation=p_inA_ofB,
                rotation=rr.Quaternion(xyzw=Rotation.from_matrix(R_inA_ofB).as_quat()),
            )
        )

