import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import traceback
import time

file_path = "test_data/1727320101-665925967.pcd"

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud loaded with {len(pcd.points)} points.")
    return pcd

def downsample_point_cloud(pcd, voxel_size=0.2):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points.")
    return downsampled_pcd

def remove_radius_outliers(pcd, nb_points=6, radius=1.2):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    ror_pcd = pcd.select_by_index(ind)
    print(f"After Radius Outlier Removal: {len(ror_pcd.points)} points remain.")
    return ror_pcd

def segment_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=2000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    road_pcd = pcd.select_by_index(inliers)
    non_road_pcd = pcd.select_by_index(inliers, invert=True)
    
    print(f"Road points: {len(road_pcd.points)}, Non-road points: {len(non_road_pcd.points)}")
    return road_pcd, non_road_pcd

def cluster_dbscan(pcd, eps=0.3, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    return labels, max_label

def assign_cluster_colors(pcd, labels):
    try:
        if len(labels) != len(pcd.points):
            raise ValueError("The number of labels does not match the number of points in the point cloud.")

        unique_labels = np.unique(labels)

        label_mapping = {}
        idx = 0
        for label in unique_labels:
            if label != -1:
                label_mapping[label] = idx
                idx += 1
        max_label = idx - 1
        print(f"Max label after mapping: {max_label}")

        mapped_labels = np.array([label_mapping[label] if label != -1 else -1 for label in labels])

        num_clusters = max_label + 1
        colors = np.zeros((len(labels), 3))

        if num_clusters > 0:
            hsv_colors = np.zeros((num_clusters, 3))
            hsv_colors[:, 0] = np.linspace(0, 1, num_clusters, endpoint=False)
            hsv_colors[:, 1] = 1
            hsv_colors[:, 2] = 1
            rgb_colors = plt.cm.hsv(hsv_colors[:, 0])

            for i in range(num_clusters):
                colors[mapped_labels == i] = rgb_colors[i][:3]
        else:
            colors[:] = [1, 0, 0]

        colors[mapped_labels == -1] = [0, 0, 0]

        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd, mapped_labels

    except Exception as e:
        print(f"Error in assign_cluster_colors: {e}")
        traceback.print_exc()
        return pcd, labels
def filter_clusters(pcd, labels,
                    min_points_in_cluster=5,
                    max_points_in_cluster=40,
                    min_z_value=-1.5,
                    max_z_value=2.5,
                    min_height=0.5,
                    max_height=2.0,
                    max_distance=30.0):
    try:
        print("Entering filter_clusters()")
        bboxes = []
        unique_labels = np.unique(labels)
        print(f"Labels dtype: {labels.dtype}")
        for i in unique_labels:
            if i == -1:
                continue  
            cluster_indices = np.where(labels == i)[0]
            num_points = len(cluster_indices)
            print(f"Processing cluster {i} with {num_points} points")

            if min_points_in_cluster > max_points_in_cluster:
                min_points_in_cluster, max_points_in_cluster = max_points_in_cluster, min_points_in_cluster

            if not (min_points_in_cluster <= num_points <= max_points_in_cluster):
                print(f"Cluster {i} skipped due to point count")
                continue
            else:
                print(f"Cluster {i} accepted")

            cluster_pcd = pcd.select_by_index(cluster_indices)
            print("Selected cluster points")
            points = np.asarray(cluster_pcd.points)
            print(f"Converted points to numpy array with shape {points.shape}")

            if points.size == 0:
                print(f"Cluster {i} has no points after selection.")
                continue

            if np.isnan(points).any() or np.isinf(points).any():
                print(f"Cluster {i} has NaN or Inf values in points.")
                continue

            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            print(f"Cluster {i} Z-range: {z_min} to {z_max}")

            if not (min_z_value <= z_min <= max_z_value):
                print(f"Cluster {i} skipped due to Z-range")
                continue
            else:
                print(f"Cluster {i} passed Z-range filter")

            height_diff = z_max - z_min
            print(f"Cluster {i} height difference: {height_diff}")
            if not (min_height <= height_diff <= max_height):
                print(f"Cluster {i} skipped due to height difference")
                continue
            else:
                print(f"Cluster {i} passed height difference filter")

            distances = np.linalg.norm(points, axis=1)
            max_dist = distances.max()
            print(f"Cluster {i} max distance from origin: {max_dist}")
            if max_dist > max_distance:
                print(f"Cluster {i} skipped due to max distance")
                continue
            else:
                print(f"Cluster {i} passed max distance filter")

            # 클러스터 포인트 수 확인
            print(f"Cluster {i} has {len(cluster_pcd.points)} points.")

            try:
                print(f"Attempting to create oriented bounding box for Cluster {i}")
                obb = cluster_pcd.get_oriented_bounding_box()
                print(f"Oriented bounding box created for Cluster {i}")
                obb.color = (0, 1, 0)  # 초록색
                bboxes.append(obb)
                print(f"Cluster {i}: Added oriented bounding box.")
            except Exception as e:
                print(f"Error adding oriented bounding box for Cluster {i}: {e}")
                traceback.print_exc()
                continue

        print(f"Total {len(bboxes)} bounding boxes created.")
        return bboxes
    except Exception as e:
        print(f"An error occurred in filter_clusters(): {e}")
        traceback.print_exc()

def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=800, height=600)
        vis.add_geometry(pcd)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print(f"Visualization error: {e}")
        traceback.print_exc()

def main():
    try:
        print("Starting main() function")
        start_time = time.time()
        original_pcd = load_point_cloud(file_path)
        
        downsampled_pcd = downsample_point_cloud(original_pcd, voxel_size=0.2)
        
        ror_pcd = remove_radius_outliers(downsampled_pcd, nb_points=6, radius=1.2)
        
        road_pcd, non_road_pcd = segment_plane(ror_pcd)
        
        print(f"Number of non-road points: {len(non_road_pcd.points)}")
        if len(non_road_pcd.points) == 0:
            print("No non-road points to cluster!")
            return

        labels, max_label = cluster_dbscan(non_road_pcd, eps=0.3, min_points=10)
        if max_label < 0:
            print("No clusters found!")
            return

        print(f"Number of labels: {len(labels)}")
        print(f"Number of points in non_road_pcd: {len(non_road_pcd.points)}")
        if len(labels) != len(non_road_pcd.points):
            print("Mismatch in labels and points!")
            return

        non_road_pcd, labels = assign_cluster_colors(non_road_pcd, labels)

        bboxes = filter_clusters(
            non_road_pcd,
            labels,
            min_points_in_cluster=5,
            max_points_in_cluster=40,
            min_z_value=-1.5,
            max_z_value=2.5,
            min_height=0.5,
            max_height=2.0,
            max_distance=30.0
        )
        print(f"Total bounding boxes: {len(bboxes)}")
        if not bboxes:
            print("No bounding boxes to visualize!")
            return

        print("Calling visualization function...")
        visualize_with_bounding_boxes(non_road_pcd, bboxes)
        print("Visualization function call completed.")
        print("Finished main() function")
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
