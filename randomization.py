import numpy as np
import random


class Randomizer:
    def __init__(self):
        """This class is responsible for randomizing parameters related to creating the environment, including
        creating plants, ground plane, robot, and camera.
        """
        # Constant params
        self.plant_names = []
        self.vertical_row_len_max = 5  # Maximum length of vertical rows (m)
        self.vertical_row_len_min = 3  # Minimum length of vertical rows (m)
        self.horizontal_row_len = 6  # Length of horizontal rows (m)
        self.plant_dist = 0.2  # Distance between individual plants (m)
        self.row_dist = 0.76  # Distance between individual rows (m)
        self.horizontal_gap_min = 1.5
        self.horizontal_gap_max = 2.0
        self.trajectory_num = 1  # Number of routes inside one environment

        # Randomly generated params
        self.plant_name = None  # Used plant variant in the environment
        self.plant_rgb = None  # Used plant color in the environment
        self.horizontal_gap = None  # Gap length between vertical rows and horizontal rows (m)
        self.horizontal_exist = None  # Flag to indicate whether horizontal rows exist
        self.trajectory_list = []  # List encode all trajectory information
        self.ground_rgb = None  # Color for ground
        # Camera parameters will be overwritten whenever a new camera is created
        self.camera_intrinsic = None  # Camera intrinsic matrix 
        self.cam_yaw = None  # Camera yaw angle
        self.cam_x_error = None  # Camera position error for x-axis (world frame)
        self.cam_y_error = None  # Camera position error for y-axis (world frame)
        self.cam_z_error = None  # Camera position error for z-axis (world frame)

        # Initialize the list of plant names
        corn_var_num = 21
        for var_type in range(corn_var_num):
            if var_type != 0 and var_type != 1:
                self.plant_names.append(f"corn_variant_{str(var_type)}")
        sorghum_var_num = 9
        for var_type in range(sorghum_var_num):
            if var_type != 0 and var_type != 1 and var_type != 2:
                self.plant_names.append(f"sorghum_variant_{str(var_type)}")


    @staticmethod
    def _generate_random_green():
        """Generate random green RGB colors for plants.

        :return red_intensity: The R value of RGB code
        :type red_intensity: int
        :return red_intensity: The G value of RGB code
        :type red_intensity: int
        :return red_intensity: The B value of RGB code
        :type red_intensity: int
        """
        red_intensity = random.randint(0, 195)
        green_intensity = 255
        blue_intensity = random.randint(0, 128)
        return red_intensity, green_intensity, blue_intensity

    @staticmethod
    def _generate_random_brown():
        """Generate random brown RGB colors for the ground surface.

        :return red_intensity: The R value of RGB code
        :type red_intensity: int
        :return red_intensity: The G value of RGB code
        :type red_intensity: int
        :return red_intensity: The B value of RGB code
        :type red_intensity: int
        """
        brown_rgb = {
            'burlywood': (222, 184, 135),
            'tan': (210, 180, 140),
            'sandybrown': (244, 164, 96),
            'peru': (205, 133, 63),
            'saddlebrown': (139, 69, 19),
            'sienna': (160, 82, 45),
        }
        brown_names = list(brown_rgb.keys())
        brown_name = random.choice(brown_names)
        red_intensity, green_intensity, blue_intensity = brown_rgb[brown_name]
        return red_intensity, green_intensity, blue_intensity

    @staticmethod
    def _create_camera_intrinsic(img_width, img_height, horizontal_fov, x_offset, y_offset):
        """Construct the camera intrinsic for the camera sensor in every environment.

        :return camera_intrinsic: The camera intrinsic matrix (3, 3)
        :type camera_intrinsic: np.ndarray
        """
        vertical_fov = (img_height / img_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (img_width / 2.0) / np.tan(horizontal_fov / 2.0)  # Focal length
        f_y = (img_height / 2.0) / np.tan(vertical_fov / 2.0)

        camera_intrinsic = np.array([[f_x, 0.0, x_offset],
                                     [0.0, f_y, y_offset],
                                     [0.0, 0.0, 1.0]])
        return camera_intrinsic

    def randomize_plant(self):
        """Randomize plant parameters for the current environment
        """
        # self.plant_name = random.choice(self.plant_names)
        self.plant_name = "sorghum_variant_5"
        self.plant_rgb = self._generate_random_green()
        self.horizontal_gap = random.uniform(1.5, 2.0)

        # Determine whether horizontal rows exist (0.9 probability exist)
        prob = random.uniform(0.0, 1.0)
        if prob > 0.9:
            self.horizontal_exist = False
        else:
            self.horizontal_exist = True

    def randomize_robot(self):
        """Randomize robot-related parameters for the current environment

           Generate trajectory_num number of trajectories
        """
        '''
        trajectory data structure: 
            [[(init_lane, target_lane), (init_x, init_y, init_yaw), [(wp1), (wp2), ...]], [...], ...]
        
        robot_init_lane: The lane no. for robot initial position
        robot_target_lane: The lane no. for robot target position
        robot_init_x: Robot initial x position (m) in world frame
        robot_init_y: Robot initial y position (m) in world frame
        robot_init_yaw: Robot initial yaw angle (degrees) in world frame
        robot_target_x: Robot target x position (m) in world frame
        robot_target_y: Robot target y position (m) in world frame
        '''
        for i in range(self.trajectory_num):
            trajectory = []

            # Generate initial and target lane no.
            direction = random.randint(0, 1)  # 0 is left, 1 is right
            if direction == 0:
                robot_init_lane = random.choice([1, -1, -2, -3])  # 1 and -1 are closest to the central row
                if robot_init_lane < 0:
                    lanes_gap = random.randint(1, 2)
                    robot_target_lane = robot_init_lane + lanes_gap + 2
                else:
                    lanes_gap = 1
                    robot_target_lane = robot_init_lane + 2
                
                # Determine exact initial pose and target position
                robot_init_x = random.uniform(4.5, 4.7)
                if robot_init_lane < 0:
                    robot_init_y = random.uniform(robot_init_lane * self.row_dist + self.row_dist / 4,
                                                  robot_init_lane * self.row_dist + (3 * self.row_dist) / 4)
                else:
                    robot_init_y = random.uniform(robot_init_lane * self.row_dist - self.row_dist / 4,
                                                  robot_init_lane * self.row_dist - (3 * self.row_dist) / 4)
                robot_init_yaw = random.uniform(-15.0, 15.0)
                # Target position only depends on initial position and distance between rows
                robot_target_x = robot_init_x
                robot_target_y = robot_init_y + self.row_dist * (lanes_gap + 1)

                trajectory.append((robot_init_lane, robot_target_lane))
                trajectory.append((robot_init_x, robot_init_y, robot_init_yaw))
                trajectory.append((robot_target_x, robot_target_y))

                # Based on initial lane and target lane, randomize trajectory
                waypoints = []
                num_waypoints = lanes_gap + 1
                if num_waypoints == 3:
                    waypoint_1_x = self.vertical_row_len_max + self.horizontal_gap * (1 / 2)
                    waypoint_1_y = (robot_init_lane + 1) * self.row_dist
                    waypoint_1 = (waypoint_1_x, waypoint_1_y)

                    waypoint_2_x = self.vertical_row_len_max + random.uniform(self.horizontal_gap * (1 / 2),
                                                                              self.horizontal_gap * (3 / 4))
                    waypoint_2_y = waypoint_1_y + self.row_dist
                    waypoint_2 = (waypoint_2_x, waypoint_2_y)

                    waypoint_3_x = self.vertical_row_len_max + random.uniform(self.horizontal_gap * (1 / 4),
                                                                              self.horizontal_gap * (1 / 2))
                    waypoint_3_y = waypoint_2_y + self.row_dist
                    waypoint_3 = (waypoint_3_x, waypoint_3_y)

                    waypoints.append(waypoint_1)
                    waypoints.append(waypoint_2)
                    waypoints.append(waypoint_3)
                    waypoints.append((robot_target_x, robot_target_y))
                    trajectory.append(waypoints)
                elif num_waypoints == 2:
                    waypoint_1_x = self.vertical_row_len_max + self.horizontal_gap * (1 / 2)
                    waypoint_1_y = None
                    if robot_init_lane > 0:
                        waypoint_1_y = robot_init_lane * self.row_dist
                    else:
                        waypoint_1_y = (robot_init_lane + 1) * self.row_dist
                    waypoint_1 = (waypoint_1_x, waypoint_1_y)

                    # Determine the second waypoint
                    waypoint_2_x = self.vertical_row_len_max + random.uniform(self.horizontal_gap * (1 / 4),
                                                                              self.horizontal_gap * (1 / 2))
                    waypoint_2_y = waypoint_1_y + self.row_dist
                    waypoint_2 = (waypoint_2_x, waypoint_2_y)

                    waypoints.append(waypoint_1)
                    waypoints.append(waypoint_2)
                    waypoints.append((robot_target_x, robot_target_y))
                    trajectory.append(waypoints)

                self.trajectory_list.append(trajectory)

            else:
                robot_init_lane = random.choice([-1, 1, 2, 3])
                if robot_init_lane < 0:
                    lanes_gap = 1
                    robot_target_lane = robot_init_lane - 2
                else:
                    lanes_gap = random.randint(1, 2)
                    robot_target_lane = robot_init_lane - lanes_gap - 2
                
                # Determine exact initial pose and target position
                robot_init_x = random.uniform(4.5, 4.7)
                if robot_init_lane < 0:
                    robot_init_y = random.uniform(robot_init_lane * self.row_dist + self.row_dist / 4,
                                                  robot_init_lane * self.row_dist + (3 * self.row_dist) / 4)
                else:
                    robot_init_y = random.uniform(robot_init_lane * self.row_dist - self.row_dist / 4,
                                                  robot_init_lane * self.row_dist - (3 * self.row_dist) / 4)
                robot_init_yaw = random.uniform(-15.0, 15.0)
                # Target position only depends on initial position and distance between rows
                robot_target_x = robot_init_x
                robot_target_y = robot_init_y - self.row_dist * (lanes_gap + 1)

                trajectory.append((robot_init_lane, robot_target_lane))
                trajectory.append((robot_init_x, robot_init_y, robot_init_yaw))
                trajectory.append((robot_target_x, robot_target_y))

                # Based on initial lane and target lane, randomize trajectory
                waypoints = []
                num_waypoints = lanes_gap + 1
                if num_waypoints == 3:
                    waypoint_1_x = self.vertical_row_len_max + self.horizontal_gap * (1 / 2)
                    waypoint_1_y = (robot_init_lane - 1) * self.row_dist
                    waypoint_1 = (waypoint_1_x, waypoint_1_y)

                    waypoint_2_x = self.vertical_row_len_max + random.uniform(self.horizontal_gap * (1 / 2),
                                                                              self.horizontal_gap * (3 / 4))
                    waypoint_2_y = waypoint_1_y - self.row_dist
                    waypoint_2 = (waypoint_2_x, waypoint_2_y)

                    waypoint_3_x = self.vertical_row_len_max + random.uniform(self.horizontal_gap * (1 / 4),
                                                                              self.horizontal_gap * (1 / 2))
                    waypoint_3_y = waypoint_2_y - self.row_dist
                    waypoint_3 = (waypoint_3_x, waypoint_3_y)

                    waypoints.append(waypoint_1)
                    waypoints.append(waypoint_2)
                    waypoints.append(waypoint_3)
                    waypoints.append((robot_target_x, robot_target_y))
                    trajectory.append(waypoints)
                elif num_waypoints == 2:
                    waypoint_1_x = self.vertical_row_len_max + self.horizontal_gap * (1 / 2)
                    waypoint_1_y = None
                    if robot_init_lane > 0:
                        waypoint_1_y = (robot_init_lane - 1) * self.row_dist
                    else:
                        waypoint_1_y = robot_init_lane * self.row_dist
                    waypoint_1 = (waypoint_1_x, waypoint_1_y)

                    # Determine the second waypoint
                    waypoint_2_x = self.vertical_row_len_max + random.uniform(self.horizontal_gap * (1 / 4),
                                                                              self.horizontal_gap * (1 / 2))
                    waypoint_2_y = waypoint_1_y - self.row_dist
                    waypoint_2 = (waypoint_2_x, waypoint_2_y)

                    waypoints.append(waypoint_1)
                    waypoints.append(waypoint_2)
                    waypoints.append((robot_target_x, robot_target_y))
                    trajectory.append(waypoints)

                self.trajectory_list.append(trajectory)

    def randomize_ground(self):
        """Randomize the color of ground plane
        """
        # TODO: Fix ground color for now
        self.ground_rgb = (244, 164, 96)

    def randomize_camera(self):
        """Randomize camera-related params for individual camera
        """
        # Randomized_parameters
        self.cam_x_error = random.uniform(-0.1, 0.1)
        self.cam_y_error = random.uniform(-0.1, 0.1)
        self.cam_z_error = random.uniform(-0.1, 0.1)

        # TODO: Fix some camera parameters for now
        original_img_width = 500
        original_img_height = 500
        horizontal_fov = 90

        # Principal point
        x_offset = 250  # New principal point
        y_offset = 250
        true_x_offset = original_img_width // 2  # Original principal point
        true_y_offset = original_img_height // 2
        x_offset_diff = x_offset - true_x_offset
        y_offset_diff = y_offset - true_y_offset
        capture_img_width = original_img_width + 2 * abs(x_offset_diff)  # Size of captured image
        capture_img_height = original_img_height + 2 * abs(y_offset_diff)

        self.camera_intrinsic = self._create_camera_intrinsic(capture_img_width, capture_img_height, horizontal_fov,
                                                              x_offset, y_offset)  # Generate camera intrinsic matrix

        return (horizontal_fov, original_img_width, original_img_height, capture_img_width,
                capture_img_height, x_offset_diff, y_offset_diff, self.camera_intrinsic)

    def save_config(self):
        """Return plant and environment related config
        """
        config = {
            "plant_name": self.plant_name,
            "plant_rgb": self.plant_rgb,
            "horizontal_gap": self.horizontal_gap,
            "horizontal_exist": self.horizontal_exist,
            "trajectory_num": self.trajectory_num,
            "trajectory_list": self.trajectory_list
        }
        return config
        



