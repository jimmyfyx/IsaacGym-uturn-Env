import numpy as np
import yaml
import csv


class DataRecorder:
    def __init__(self):
        self.env_config = {}
        self.robot_pose = {}

    def record_env_params(self, row_len_list, rand_config):
        # Reformat data structure
        plant_rgb = rand_config['plant_rgb']
        plant_rgb = list(plant_rgb)
        trajectory_list = rand_config['trajectory_list']
        trajectory_dict = {}
        for i in range(len(trajectory_list)):
            trajectory_dict[f'route_{i}'] = {}
            trajectory_dict[f'route_{i}']['init_lane'] = trajectory_list[0][0][0]
            trajectory_dict[f'route_{i}']['target_lane'] = trajectory_list[0][0][1]
            trajectory_dict[f'route_{i}']['init_x'] = trajectory_list[0][1][0]
            trajectory_dict[f'route_{i}']['init_y'] = trajectory_list[0][1][1]
            trajectory_dict[f'route_{i}']['init_yaw'] = trajectory_list[0][1][2]
            trajectory_dict[f'route_{i}']['target_x'] = trajectory_list[0][2][0]
            trajectory_dict[f'route_{i}']['target_y'] = trajectory_list[0][2][1]
            trajectory_dict[f'route_{i}']['waypoints'] = []
            for waypoint in trajectory_list[i][3]:
                trajectory_dict[f'route_{i}']['waypoints'].append(list(waypoint))
        
        print(trajectory_list)
        
        self.env_config ={
            "plant_name": rand_config['plant_name'],
            "plant_rgb": plant_rgb,
            "horizontal_gap": rand_config['horizontal_gap'],
            "horizontal_exist": rand_config['horizontal_exist'],
            "trajectory_num": rand_config['trajectory_num'],
            "trajectory_list": trajectory_dict,
            "row_len_list": row_len_list
        }
    
    def record_camera_params(self, cam_idx, img_width, img_height, camera_intrinsic, cam_x, cam_y, cam_z, cam_yaw):
        cam_config = {
            "img_width": img_width,
            "img_height": img_height,
            "camera_intrinsic": camera_intrinsic.tolist(),
            "cam_x": cam_x,
            "cam_y": cam_y,
            "cam_z": cam_z,
            "cam_yaw": cam_yaw,
        }
        self.env_config[f'camera_{cam_idx}'] = cam_config
    
    def save_env_params(self, env_idx):
        outfile = open(f"data/env{env_idx}/env_config.yaml", "w")
        yaml.dump(self.env_config, outfile)
        outfile.close()
    
    def record_robot_pose(self, trajectory_idx, pose):
        if trajectory_idx not in self.robot_pose.keys():
            self.robot_pose[trajectory_idx] = []
        self.robot_pose[trajectory_idx].append(pose)
    
    def save_robot_pose(self, env_idx):
        for trajectory_idx in self.robot_pose.keys():
            with open(f"data/env{env_idx}/route_{trajectory_idx}/pose.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.robot_pose[trajectory_idx])




        

