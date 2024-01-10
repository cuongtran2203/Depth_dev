from Depth_estimate import Depth_Estimation
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
if __name__=="__main__":


    img=cv2.imread("bike.png")
    model_detection=YOLO('yolov8s.pt')
    model_depth=Depth_Estimation()
    results=model_detection.predict(img,save=True)
    bboxes= results[0].boxes.xyxy.numpy()
    for box in bboxes:
        box=list(map(int,box))
        object_img=img[box[1]:box[3],box[0]:box[2]]
        print(box)
        depth_map,output_points=model_depth.inference(object_img)
        
        cv2.imwrite("depth_map.png",depth_map)
        print(output_points)
        x,y,z=zip(*output_points)
        # Tính toán giới hạn của tất cả các tọa độ
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        min_z, max_z = min(z), max(z)
        # Tạo subplot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #  # Sử dụng RANSAC để ước lượng mô hình mặt phẳng
        # plane_model, inliers = o3d.geometry.plane_from_points(output_points, distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        
        # # Xác định hướng và vị trí từ mô hình mặt phẳng
        # normal_vector = plane_model[0:3]
        # distance_to_origin = plane_model[3]
        
        # print("Estimated Normal Vector:", normal_vector)
        # print("Estimated Distance to Origin:", distance_to_origin)

        # Vẽ điểm 3D trên đồ thị
        ax.scatter(x, y, z, c='r', marker='o')

        # Vẽ bounding box
        box_vertices = np.array([[min_x, min_y, min_z],
                                [max_x, min_y, min_z],
                                [max_x, max_y, min_z],
                                [min_x, max_y, min_z],
                                [min_x, min_y, max_z],
                                [max_x, min_y, max_z],
                                [max_x, max_y, max_z],
                                [min_x, max_y, max_z]])

        ax.plot([box_vertices[i, 0] for i in range(0, 5)], [box_vertices[i, 1] for i in range(0, 5)], [box_vertices[i, 2] for i in range(0, 5)], color='b')
        ax.plot([box_vertices[i, 0] for i in range(4, 8)], [box_vertices[i, 1] for i in range(4, 8)], [box_vertices[i, 2] for i in range(4, 8)], color='b')
        ax.plot([box_vertices[i, 0] for i in range(0, 4)], [box_vertices[i, 1] for i in range(0, 4)], [box_vertices[i, 2] for i in range(0, 4)], color='b', linestyle='dashed')

        # Đặt tên cho các trục
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Hiển thị đồ thị
        plt.show()


