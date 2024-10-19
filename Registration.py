import copy
import open3d as o3d
import numpy as np

def draw_registration_results(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.5, 0.5, 0.5])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):

    source = copy.deepcopy(pcd_1)
    target = copy.deepcopy(pcd_2)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    #draw_registration_results(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

pcd_1 = o3d.io.read_point_cloud("GS_FARO_Scan_038_5mm.pts")
pcd_2 = o3d.io.read_point_cloud("GS_FARO_Scan_039_5mm.pts")


voxel_size = 0.05  # means 5cm for this dataset
source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
    voxel_size)

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)

result_icp = refine_registration(source_down, target_down,source_fpfh,target_fpfh,voxel_size)
print(result_icp)
print(result_icp.transformation)
#draw_registration_results(source, target, result_icp.transformation)

pcd_1_transformed = copy.deepcopy(pcd_1)
pcd_1_transformed.transform(result_icp.transformation)
pcd_1_transformed.paint_uniform_color([0, 0.651, 0.929])
o3d.visualization.draw_geometries([pcd_1, pcd_2, pcd_1_transformed])

#threshold  = 0.5

#o3d.visualization.draw_geometries([pcd_1, pcd_2])

#draw_registration_results(pcd_1, pcd_2)
print("dsdsad")

#evaluation = o3d.pipelines.registration.evaluate_registration(pcd_1, pcd_2, 2)
#print(evaluation)

# print("Apply point-to-point ICP")
# reg_p2p = o3d.pipelines.registration.registration_icp(
#     pcd_1, pcd_2, threshold, np.eye(4),
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
#
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)