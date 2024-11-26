import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import time
from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위한 모듈

# PCD 파일이 저장된 디렉토리 경로
pcd_directory = "C:/Users/yewon/Downloads/COSE416_HW1_tutorial/COSE416_HW1_tutorial/data/01_straight_walk/pcd"
output_directory = "./output_frames_matplotlib"

# 출력 디렉토리 생성
os.makedirs(output_directory, exist_ok=True)

# Bounding Box 및 포인트 클라우드 시각화 함수 (이미지 저장 포함)
def visualize_and_save_matplotlib(pcd, bounding_boxes, output_path, point_size=1.0):
    try:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 포인트 클라우드 그리기
        points = np.asarray(pcd.points)
        ax.scatter(points[:,0], points[:,1], points[:,2], c='blue', s=point_size)

        # 바운딩 박스 그리기
        for bbox in bounding_boxes:
            min_bound = bbox.get_min_bound()
            max_bound = bbox.get_max_bound()
            # AABB 코너 계산
            corners = np.array([
                [min_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
            ])

            # 바운딩 박스의 면을 그리기 위한 선 연결
            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)
            ]

            for edge in edges:
                ax.plot([corners[edge[0],0], corners[edge[1],0]],
                        [corners[edge[0],1], corners[edge[1],1]],
                        [corners[edge[0],2], corners[edge[1],2]], color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud with Bounding Boxes')
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved visualization to {output_path}")
    except Exception as e:
        print(f"Visualization error with Matplotlib: {e}")

# 포인트 클라우드 로드 함수
def load_point_cloud(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"Point cloud loaded with {len(pcd.points)} points.")
        return pcd
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        traceback.print_exc()
        return None

# 다운샘플링 함수
def downsample_point_cloud(pcd, voxel_size=0.2):
    try:
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points.")
        return downsampled_pcd
    except Exception as e:
        print(f"Error during downsampling: {e}")
        traceback.print_exc()
        return pcd

# 반경 기반 이상치 제거 함수
def remove_radius_outliers(pcd, nb_points=6, radius=1.2):
    try:
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        ror_pcd = pcd.select_by_index(ind)
        print(f"After Radius Outlier Removal: {len(ror_pcd.points)} points remain.")
        return ror_pcd
    except Exception as e:
        print(f"Error during radius outlier removal: {e}")
        traceback.print_exc()
        return pcd

# 평면 세분화 함수
def segment_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=2000):
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        road_pcd = pcd.select_by_index(inliers)
        non_road_pcd = pcd.select_by_index(inliers, invert=True)

        print(f"Road points: {len(road_pcd.points)}, Non-road points: {len(non_road_pcd.points)}")
        return road_pcd, non_road_pcd
    except Exception as e:
        print(f"Error during plane segmentation: {e}")
        traceback.print_exc()
        return pcd, o3d.geometry.PointCloud()

# DBSCAN 클러스터링 함수
def cluster_dbscan(pcd, eps=0.3, min_points=10):
    try:
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

        max_label = labels.max()
        print(f"Point cloud has {max_label + 1} clusters")
        return labels, max_label
    except Exception as e:
        print(f"Error during DBSCAN clustering: {e}")
        traceback.print_exc()
        return np.array([]), -1

# 클러스터 색상 할당 함수
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

# 클러스터 필터링 및 Bounding Box 생성 함수
def filter_clusters(pcd, labels,
                    min_points_in_cluster=5,
                    max_points_in_cluster=40,
                    min_z_value=-1.5,
                    max_z_value=2.5,
                    min_height=0.5,
                    max_height=2.0,
                    max_distance=50.0):  # max_distance 완화
    try:
        print("Entering filter_clusters()")
        bboxes = []
        unique_labels = np.unique(labels)
        print(f"Labels dtype: {labels.dtype}")
        
        for i in unique_labels:
            if i == -1:
                continue  # 노이즈는 건너뜁니다.
            cluster_indices = np.where(labels == i)[0]
            num_points = len(cluster_indices)
            print(f"Processing cluster {i} with {num_points} points")

            if num_points < min_points_in_cluster:
                print(f"Cluster {i} skipped due to insufficient points.")
                continue

            cluster_pcd = pcd.select_by_index(cluster_indices)

            # 데이터 검증
            points = np.asarray(cluster_pcd.points)
            if len(points) == 0:
                print(f"Cluster {i} has no points. Skipping...")
                continue

            if np.isnan(points).any() or np.isinf(points).any():
                print(f"Cluster {i} contains NaN or Inf values. Skipping...")
                continue

            print(f"Cluster {i} points: {points[:5]}... (showing first 5 points)")

            # Z-range와 height 필터링
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            height_diff = z_max - z_min
            if not (min_z_value <= z_min <= max_z_value):
                print(f"Cluster {i} skipped due to Z-range. z_min={z_min}, z_max={z_max}")
                continue
            if not (min_height <= height_diff <= max_height):
                print(f"Cluster {i} skipped due to height difference. height_diff={height_diff}")
                continue

            # 거리 필터링
            distances = np.linalg.norm(points, axis=1)
            max_dist = distances.max()
            if max_dist > max_distance:
                print(f"Cluster {i} skipped due to max distance. max_dist={max_dist}")
                continue

            # Bounding Box 생성
            try:
                print(f"Attempting to create axis-aligned bounding box for Cluster {i}")
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)  # 빨간색
                bbox.scale(1.5, bbox.get_center())
                bboxes.append(bbox)
                print(f"Cluster {i}: Bounding box created successfully.")
            except Exception as e:
                print(f"Error creating AABB for Cluster {i}: {e}")
                traceback.print_exc()
                continue

        print(f"Total {len(bboxes)} bounding boxes created.")
        return bboxes
    except Exception as e:
        print(f"An error occurred in filter_clusters(): {e}")
        traceback.print_exc()


# 메인 함수
def main():
    try:
        print("Starting main() function")
        start_time = time.time()
        
        # PCD 파일 목록 가져오기
        pcd_files = sorted([f for f in os.listdir(pcd_directory) if f.endswith('.pcd')])
        if not pcd_files:
            print("No PCD files found in the specified directory.")
            return
        
        for idx, file_name in enumerate(pcd_files):
            print(f"\nProcessing {file_name} ({idx + 1}/{len(pcd_files)})")
            file_path = os.path.join(pcd_directory, file_name)

            # 포인트 클라우드 로드
            original_pcd = load_point_cloud(file_path)
            if original_pcd is None:
                print(f"Failed to load {file_name}. Skipping...")
                continue

            # 다운샘플링
            downsampled_pcd = downsample_point_cloud(original_pcd, voxel_size=0.2)

            # 반경 기반 이상치 제거
            ror_pcd = remove_radius_outliers(downsampled_pcd, nb_points=6, radius=1.2)

            # 평면 세분화
            road_pcd, non_road_pcd = segment_plane(ror_pcd)

            print(f"Number of non-road points: {len(non_road_pcd.points)}")
            if len(non_road_pcd.points) == 0:
                print("No non-road points to cluster! Skipping...")
                continue

            # DBSCAN 클러스터링
            labels, max_label = cluster_dbscan(non_road_pcd, eps=0.3, min_points=10)
            if max_label < 0:
                print("No clusters found! Skipping...")
                continue

            print(f"Number of labels: {len(labels)}")
            print(f"Number of points in non_road_pcd: {len(non_road_pcd.points)}")
            if len(labels) != len(non_road_pcd.points):
                print("Mismatch in labels and points! Skipping...")
                continue

            # 클러스터 색상 할당
            non_road_pcd, labels = assign_cluster_colors(non_road_pcd, labels)

            # 클러스터 필터링 및 Bounding Box 생성
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
                print("No bounding boxes to visualize! Skipping...")
                continue

            # 시각화 및 이미지 저장
            output_path = os.path.join(output_directory, f"frame_{idx:04d}.png")
            visualize_and_save_matplotlib(non_road_pcd, bboxes, output_path, point_size=1.0)

        end_time = time.time()
        print(f"\nFinished main() function")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
