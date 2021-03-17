import open3d as o3d
import numpy as np
import copy


def visualize(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.light_on = False
    opt.point_size = 2
    vis.run()
    vis.destroy_window()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def feature_extraction(pcd, voxel_size, gamma=False):
    if gamma:
        color = np.asarray(pcd.colors)
        color = np.minimum(np.power(color, 0.6), 1)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    print(":: Estimate normal with search radius %.3f." % voxel_size)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=30))

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud('/home/hvt/Code/ACSC/experiments/RGBMAP_v2/RGBMap_avia1.pcd')
    # target = o3d.io.read_point_cloud('/home/hvt/Code/ACSC/experiments/RGBMAP_v2/RGBMap_avia2.pcd')
    target = o3d.io.read_point_cloud('/home/hvt/Code/ACSC/experiments/RGBMAP_v2/RGBMap_avia3.pcd')
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = feature_extraction(source, voxel_size)
    target_down, target_fpfh = feature_extraction(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, initial_guess):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, initial_guess,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def mapping(pcd_list, voxel_size):
    for frame_idx in range(len(pcd_list) - 1):
        # 读取数据
        if frame_idx == 0:
            map_pcd = pcd_list[frame_idx]
            map_down, map_fpfh = feature_extraction(map_pcd, voxel_size=voxel_size, gamma=True)

        source_pcd = pcd_list[frame_idx + 1]
        source_down, source_fpfh = feature_extraction(source_pcd, voxel_size=voxel_size, gamma=True)

        # 全局ICP
        result_ransac = execute_global_registration(source_down, map_down,
                                                    source_fpfh, map_fpfh,
                                                    voxel_size)
        print(result_ransac)

        # refine结果
        result_icp = refine_registration(source_pcd, map_pcd, source_fpfh, map_fpfh,
                                         voxel_size, initial_guess=result_ransac.transformation)
        print(result_icp)
        # draw_registration_result(source_pcd, map_pcd,
        #                          result_icp.transformation)

        # 更新map
        map_pcd += source_pcd.transform(result_icp.transformation)
        map_down, map_fpfh = feature_extraction(map_pcd, voxel_size=voxel_size)

    return map_pcd


if __name__ == "__main__":
    voxel_size = 0.3  # means 5cm for the dataset

    # 点云数据路径
    # 每帧之间需要有相互
    data_list = [
        '/home/hvt/Code/ACSC/experiments/RGBMAP_v2/RGBMap_avia1.pcd',
        '/home/hvt/Code/ACSC/experiments/RGBMAP_v2/RGBMap_avia2.pcd',
        '/home/hvt/Code/ACSC/experiments/RGBMAP_v2/RGBMap_avia3.pcd',
    ]

    # 读取点云数据集
    pcds = [o3d.io.read_point_cloud(path) for path in data_list]
    map_pcd = mapping(pcd_list=pcds, voxel_size=voxel_size)

    # 绘制地图
    visualize(map_pcd)

    o3d.io.write_point_cloud("mapping.pcd", map_pcd)
