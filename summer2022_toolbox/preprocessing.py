import os
import copy
import pickle
import numpy as np
import open3d as o3d
from .read_object import *
from tqdm import tqdm

def preprocess_and_save(folder, ref_name, save_path):
    """
    Unify object sizes and orientation to the reference object
    @param    folder: (in form of (data)/train/... and (data)/test/...)
    @param    ref_name: name of the reference object
    @param    save_path: preprocessed data will be saved here as 'train.txt' and 'test.txt'
    """
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print("Data path: " + folder)
    print("Reference Object: " + ref_name)
    print("Preprocessed data will be saved at: " + save_path)
    train_folder = folder + "train/"
    test_folder = folder + "test/"

    train_filenames = listFileNames(train_folder)
    test_filenames = listFileNames(test_folder)

    print(":::Start preprocessing train data...")
    train_all = preprocessAll(train_filenames, folder + ref_name)
    print(":::Finished preprocessing train data...")
    print(":::Start preprocessing test data...")
    test_all = preprocessAll(test_filenames, folder + ref_name)
    print(":::Finished preprocessing test data...")

    train_save_filename = open(save_path + 'train.txt', 'wb')
    pickle.dump(train_all, train_save_filename)

    test_save_filename = open(save_path + 'test.txt', 'wb')
    pickle.dump(test_all, test_save_filename)
    print("Preprocessed data are saved at: " + save_path)

def scale_point_cloud(pcd, target_size):
    """
    scale the data to let the feature with maxmium value to target_size
    and the data is move to the min = 0
    @param pcd: original open3D.PointCloud
    @param target_size: int
    @return: scaled open3D.PointCloud
    """
    X = np.asarray(pcd.points)
    all_scale = np.array(
        [np.max(X[:, 0]) - np.min(X[:, 0]), np.max(X[:, 1]) - np.min(X[:, 1]), np.max(X[:, 2]) - np.min(X[:, 2])])
    max_scale = np.max(all_scale)
    X = X / max_scale * target_size
    pcd_scale = o3d.geometry.PointCloud()
    pcd_scale.points = o3d.utility.Vector3dVector(X)

    return pcd_scale


def prepare_point_cloud(pcd, target_size=1000, voxel_size=20):
    """
    preprocessing point cloud
    @param pcd: original open3D.PointCloud
    @param target_size: int
    @param voxel_size: int
    @return: downsampled open3D.PointCloud, FPFH of this point cloud
    """
    #     print(":: Scale to a max size %d" % target_size)
    pcd_scale = scale_point_cloud(pcd, target_size)

    #     print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd_scale.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #     print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(
                                                                   radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def GCP_registration(source_down, target_down, source_fpfh,
                     target_fpfh, voxel_size=20):
    """
    Reference: Open3D tutorial documents http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    Apply Global Registration to the source data
    @param    folder: (in form of (data)/train/... and (data)/test/...)
    @param    ref_name: name of the reference object
    @param    save_path: preprocessed data will be saved here as 'train.txt' and 'test.txt'
    """
    distance_threshold = voxel_size * 1.2
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.cpu.pybind.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999))

    return result


def ICP_registration(source_down, target_down, threshold, result_ransac):
    """
    Reference: Open3D tutorial documents http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
    Apply Global Registration to the source data
    @param    folder: (in form of (data)/train/... and (data)/test/...)
    @param    ref_name: name of the reference object
    @param    save_path: preprocessed data will be saved here as 'train.txt' and 'test.txt'
    """
    return o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10000))

def preprocess(data, target_down, target_fpfh, ICP_threshold = 15, target_size=1000, voxel_size=20, n_iter = 20):
    source_down, source_fpfh = prepare_point_cloud(data, target_size, voxel_size)
    transformations = []
    scores = []
    for i in range(n_iter):
        GCP_transformation = GCP_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        ICP_transformation = ICP_registration(source_down, target_down, ICP_threshold, GCP_transformation)
        transformations.append(ICP_transformation)
        scores.append(ICP_transformation.fitness)

    max_idx = scores.index(max(scores))
    Max_Transformation = transformations[max_idx]

    source_temp = copy.deepcopy(source_down)
    source_temp.transform(Max_Transformation.transformation)
    return source_temp

def listFileNames(folder):
    """Walk through every files in a directory"""
    filenames = []
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            filenames.append(os.path.abspath(os.path.join(dirpath, filename)))

    return filenames

def preprocessAll(filenames, ref_name, target_size=1000, voxel_size=20):
    """
    filenames: list of filenames (output of listFileNames)
    return list of scaled and downsampled data
    """
    reference = read_pointcloud(ref_name)
    reference_down, reference_fpfh = prepare_point_cloud(reference, target_size, voxel_size)

    pcd_all = []
    for filename in tqdm(filenames):
        pcd = read_pointcloud(filename)
        pcd_after = preprocess(pcd, reference_down, reference_fpfh)
        if (np.asarray(pcd_after.points).shape[0]) > 500:
            pcd_all.append(np.asarray(pcd_after.points).T)

    return pcd_all
