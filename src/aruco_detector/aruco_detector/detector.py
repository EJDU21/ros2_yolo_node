import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as scipy_R
import cv2
import numpy as np
import yaml
import os

from .aruco_config import ArucoConfig


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()
        self.config = ArucoConfig()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.image_pub = self.create_publisher(Image, '/aruco_detector/detected_image', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco_detector/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)
        # /-aruco_info------------------------------------------------------/
        self.aruco_info = {}
        self.create_subscription(
            String,
            "/aruco_info",
            self.aruco_map_callback,
            10)
        # /-----------------------------------------------------------------/

        # map_path = os.path.join(os.path.dirname(__file__), 'aruco_map.yaml')
        # aruco_path = '/workspaces/src/aruco_detector/config/aruco_location.yaml'
        # with open(aruco_path, 'r') as f:
        #     self.marker_map = yaml.safe_load(f)
        # print(self.marker_map)

        # yaml_path = os.path.join(os.path.dirname(__file__), 'map.yaml')
        map_path = '/workspaces/src/aruco_detector/map'
        yaml_path = yaml_path = os.path.join(map_path, 'map01.yaml')
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
        image_path = yaml_path = os.path.join(map_path, map_metadata['image'])
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']
        occupied_thresh = map_metadata.get('occupied_thresh', 0.65)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Can't load map: {image_path}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.flip(img, 0)
        h, w = img.shape

        data = []
        for row in img:
            for pixel in row:
                occ = 100 if pixel < occupied_thresh * 255 else 0
                data.append(occ)

        self.map_msg = OccupancyGrid()
        self.map_msg.header = Header()
        self.map_msg.info.resolution = resolution
        self.map_msg.info.width = w
        self.map_msg.info.height = h
        self.map_msg.info.origin.position.x = origin[0]
        self.map_msg.info.origin.position.y = origin[1]
        self.map_msg.info.origin.position.z = 0.0
        self.map_msg.info.origin.orientation.w = 1.0
        self.map_msg.data = data

        self.timer = self.create_timer(1.0, self.publish_map)

    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_msg.header.frame_id = "map"  
        self.map_pub.publish(self.map_msg)

    def polygon_area(self, pts):
        pts = np.array(pts[0])
        return 0.5 * abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))

    # /-aruco_info------------------------------------------------------/
    def aruco_map_callback(self, msg):
        try:
            self.aruco_info = yaml.safe_load(msg.data)
            self.get_logger().info("[ArucoDetectorNode] 成功接收 aruco_info")
            # 可選：印出 marker info 確認內容
            for marker_id, data in self.aruco_info .items():
                print(f"ID: {marker_id}, x: {data['x']}, y: {data['y']}, theta: {data['theta']}")
        except Exception as e:
            self.get_logger().error(f"[ArucoDetectorNode] 解析 YAML 時出錯: {e}")
    # /-----------------------------------------------------------------/

    def image_callback(self, msg):
        '''
        讀取RGB影像、Aruco_info (id、Loc)並計算camera_pose
        '''

        # /-在讀取RGB影像上畫出 ArUco 的邊框與 id ---------------------------------------------/
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8") #ROS 壓縮影像轉為 OpenCV (bgr8)格式
        if cv_image is None:                                                          #如果轉換失敗，就提早退出。
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)                             # 將RGB影像轉為灰階
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        corners, ids, _ = cv2.aruco.detectMarkers(                                    # 從灰階影像偵測 ArUco 標記位置
            gray, 
            self.config.aruco_dict, 
            parameters=self.config.aruco_params
            )                                                        
        if ids is None:                                                               # 如果沒有偵測到 ArUco，就提早退出。
            return

        gray_float = np.float32(gray) 
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001)
        for i in range(len(corners)):
            cv2.cornerSubPix(
                gray_float,
                corners[i], 
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=criteria
            )
        cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)                         # 在"RGB"影像上畫出偵測到的 ArUco 的邊框與 id

        # /-計算 camera_pose 並 publish --------------------------------------------------/
        for i, marker_id in enumerate(ids.flatten()):                                 # 逐一處理所有偵測到的 ArUco 
            if marker_id not in self.aruco_info or self.polygon_area(corners[i])<1500: # 過濾掉沒有在aruco_info裡的marker_id或是面積太小的 ArUco
                continue

            image_points = corners[i]  # shape: (4, 2)                                # 重排 ArUco 的四個角點順序
            if marker_id not in [3,4,5]:
                reorder = [2, 3, 0, 1]
                image_points = image_points[:, reorder, :]

            success, rvec, tvec, inliersinliers = cv2.solvePnPRansac(                 # 使用 solvePnPRansac 計算相機到 ArUco 的 R,T
                self.config.objp,
                image_points,
                self.config.camera_matrix,
                self.config.dist_coeffs,
                # flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                continue
            cv2.drawFrameAxes(                                                        # 在"RGB"影像上畫出 ArUco 的坐標軸
                cv_image, 
                self.config.camera_matrix, 
                self.config.dist_coeffs, 
                rvec, 
                tvec, 
                self.config.marker_length * 0.5
            )

            # /-計算 camera 到 ArUco marker 的變換矩陣----------------------------------/
            '''
            1. 建立 T_camera_marker : "camera 到 marker"的變換矩陣
            2. 建立 T_marker_camera : "marker 到 camera" 的變換矩陣
            '''
            T_camera_marker = np.eye(4)                                           # 建立 "camera 到 marker" 的變換矩陣 T_camera_marker
            R, _ = cv2.Rodrigues(rvec)                                             # 把從solvePnP 得到的 rvec (旋轉向量)轉為旋轉矩陣 R
            T_camera_marker[0:3, 0:3] = R                                          # 將旋轉矩陣 R 填入 T_camera_marker 的左上角 3x3 區域
            T_camera_marker[0:3, 3] = tvec.flatten()                               # 把從 solvePnP 得到的平移向量 tvec 填入 T_camera_marker 的前三列 
            T_marker_camera = np.linalg.inv(T_camera_marker)                      # 取 T_camera_marker 的逆矩陣

            '''
            1. 從 aruco_info 中取得 aruco 在地圖中的 pose(位置,朝向)
            2. 建立 map 到 ArUco 的變換矩陣 T_map_marker
            '''
            m = self.aruco_info[marker_id]
            theta = m['theta']
            cos_t, sin_t = np.cos(theta), np.sin(theta)    

            R_map_marker = np.array([                                             # 建立 aruco 在地圖中的朝向 R_map_marker
                [cos_t, -sin_t, 0],
                [sin_t,  cos_t, 0],
                [0,      0,     1]
            ])
            location_map_marker = np.array(                                       # 建立 aruco 在地圖中的位置 location_map_marker
                [m['x'], m['y'], 0.0], 
                dtype=np.float32
            )
            T_map_marker = np.eye(4)                                              # 建立 "地圖 到 marker" 的變換矩陣 T_map_marker
            T_map_marker[0:3, 0:3] = R_map_marker                                  # 將 R_map_marker 填入 T_map_marker 的左上角 3x3 區域
            T_map_marker[0:3, 3] = location_map_marker                             # 把 location_map_marker 填入 T_map_marker 的前三列

            '''
            1. 計算 T_map_camera : "map ➝ marker ➝ camera" 的變換矩陣
            2. 計算 T_map_base   : 加上相對機體的偏移的 "最終變換矩陣" 
            '''
            T_map_camera = T_map_marker @ self.config.T_align @ T_marker_camera   # T_map_camera = "地圖 ➝ marker" @ "marker➝ camera"
            T_map_base = T_map_camera  @ np.linalg.inv(self.config.T_camera_base) # 最終變換矩陣  = "T_map_camera" @ "相對機體的偏移"

            # /-Publish 位置pos & 四元數旋轉矩陣 q----------------------------------/
            '''
            * PoseWithCovarianceStamped : ?? (Publish 到 /aruco_detector/pose)
            * TransformStamped          : ?? (Publish 到 /aruco_detector/pose)
            '''
            pos = T_map_base[0:3, 3]                         # 從 T_map_base 的前三列取出位置 pos
            R_matrix = T_map_base[0:3, 0:3]                  # 從 T_map_base 的左上角 3x3 取出旋轉矩陣 R_matrix
            q = scipy_R.from_matrix(R_matrix).as_quat()      # 將旋轉矩陣 R_matrix 轉為四元數 q
            print(f"{pos}")
            # 建立 PoseWithCovarianceStamped
            pose_msg = PoseWithCovarianceStamped()           
            pose_msg.header.frame_id = "map"

            pose_msg.pose.pose.position.x = pos[0]           # 將位置 pos 填入 PoseWithCovarianceStamped 的對應欄位
            pose_msg.pose.pose.position.y = pos[1]
            pose_msg.pose.pose.position.z = 0.0

            pose_msg.pose.pose.orientation.x = q[0]          # 將四元數 q 填入 PoseWithCovarianceStamped 的對應欄位
            pose_msg.pose.pose.orientation.y = q[1]
            pose_msg.pose.pose.orientation.z = q[2]
            pose_msg.pose.pose.orientation.w = q[3]
            print("pose estimatation!!")
            self.pose_pub.publish(pose_msg)                  # Publish PoseWithCovarianceStamped 到 /aruco_detector/pose 
            # 建立 TransformStamped
            t = TransformStamped()                           
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"

            t.transform.translation.x = pos[0]               # 將位置 pos 填入 TransformStamped 的對應欄位
            t.transform.translation.y = pos[1]
            t.transform.translation.z = 0.0

            t.transform.rotation.x = q[0]                    # 將四元數 q 填入 TransformStamped 的對應欄位
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)             # ????發送 TF 資訊到 TF Tree
            break  
        # /-將處理過的影像（已畫出 marker 與座標軸）轉為 ROS 訊息並發佈----------------------------------/
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(img_msg)



    def yaw_to_quaternion(self, yaw):
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        return (qx, qy, qz, qw)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()