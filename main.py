from Depth_estimate import Depth_Estimation
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        print(output_points)
        x,y,z=zip(*output_points)
        # Tính toán giới hạn của tất cả các tọa độ
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        min_z, max_z = min(z), max(z)
        # Tạo subplot 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

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


