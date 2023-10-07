import numpy as np
import csv
import os
import shutil
from isaacgym import gymapi, gymutil
import math
import random

from PIL import Image
from scipy.spatial.transform import Rotation as R

from assets_loader import load_terrasentia, load_ground, load_plant
from randomization import Randomizer
from controller import PIDController
from simulation_handle import SimHandle
from data_recorder import DataRecorder


class Simulator:
    def __init__(self):
        self.randomizer = None
        self.recorder = None

    def run_simulator(self, env_idx):
        """The main function to run simulator
        """
        gym = gymapi.acquire_gym()  # Initialize gym
        args = gymutil.parse_arguments(description="terrasentia_env")  # Parse arguments

        # Create Simulation
        sim_params = gymapi.SimParams()
        sim_params.dt = dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z  # Specify z-up coordinate system
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = False
        sim_params.physx.use_gpu = False
        if args.physics_engine == gymapi.SIM_FLEX:
            pass
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 6
            sim_params.physx.num_velocity_iterations = 0
            sim_params.physx.num_threads = args.num_threads
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")
        sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        # Add Ground Plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        gym.add_ground(sim, plane_params)

        robot_asset = load_terrasentia(gym, sim)  # Load robot asset

        # DOF Initialization for terrasentia
        def clamp(x, min_value, max_value):
            return max(min(x, max_value), min_value)

        dof_names = gym.get_asset_dof_names(robot_asset)  # Get array of DOF names
        dof_props = gym.get_asset_dof_properties(robot_asset)  # Get array of DOF properties
        num_dofs = gym.get_asset_dof_count(robot_asset)  # Create an array of DOF states
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        dof_types = [gym.get_asset_dof_type(robot_asset, i) for i in range(num_dofs)]  # Get list of DOF types
        dof_positions = dof_states['pos']  # Get the position slice of the DOF state array
        # Get the limit-related slices of the DOF properties array
        stiffnesses = dof_props['stiffness']
        dampings = dof_props['damping']
        armatures = dof_props['armature']
        has_limits = dof_props['hasLimits']
        lower_limits = dof_props['lower']
        upper_limits = dof_props['upper']

        # Initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.zeros(num_dofs)
        speeds = np.zeros(num_dofs)
        for i in range(num_dofs):
            if has_limits[i]:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
                # Make sure our default position is in range
                if lower_limits[i] > 0.0:
                    defaults[i] = lower_limits[i]
                elif upper_limits[i] < 0.0:
                    defaults[i] = upper_limits[i]
            else:
                # Set reasonable animation limits for unlimited joints
                # Unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            dof_positions[i] = defaults[i]  # Set DOF position to default

        # Create path to save data
        if not os.path.exists(f"data/env{env_idx}"):
            os.mkdir(f"data/env{env_idx}")
            for i in range(self.randomizer.trajectory_num):
                os.mkdir(f"data/env{env_idx}/route_{i}")
                os.mkdir(f"data/env{env_idx}/route_{i}/rgb")
                os.mkdir(f"data/env{env_idx}/route_{i}/depth")
                for j in range(3):
                    os.mkdir(f"data/env{env_idx}/route_{i}/rgb/camera_{j}")
                    os.mkdir(f"data/env{env_idx}/route_{i}/depth/camera_{j}")
        else:
            raise RuntimeError("The environment index already exist")

        # Create Environment(s)
        num_envs = 1   # Parallel environments not in-use for this application
        num_per_row = 2
        spacing = 20.0
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        envs = []
        actor_handles = []
        print(f"Creating env{env_idx}...")
        for i in range(num_envs):
            # Create environment
            env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            env_handle_list = []

            # Create and place plants
            # Randomize plants-related parameters
            self.randomizer.randomize_plant()
            plant_asset = load_plant(gym, sim, self.randomizer.plant_name)

            # Place 7 vertical rows
            vertical_row_len = 0.0
            step_count = 0
            # Randomize and specify the vertical row length of each row
            row_len_list = []
            cur_len_list = []
            for j in range(7):
                row_len_list.append(random.uniform(self.randomizer.vertical_row_len_min,
                                                   self.randomizer.vertical_row_len_max))
                cur_len_list.append(0.0)
            while vertical_row_len <= self.randomizer.vertical_row_len_max:
                # Randomize plants orientation for each step across all 6 rows
                yaw_angle_list = []
                for j in range(7):
                    yaw_angle_list.append(random.randint(-180, 180))

                # Place plants from left to right for every row in parallel
                for row_num in range(-3, 4, 1):
                    if cur_len_list[row_num + 3] < row_len_list[row_num + 3]:
                        plant_pose = gymapi.Transform()
                        plant_pose.p = gymapi.Vec3(step_count * self.randomizer.plant_dist,
                                                   self.randomizer.row_dist * row_num, 0.0)
                        plant_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                                   np.radians(yaw_angle_list[row_num + 3]))
                        plant_handle = gym.create_actor(env, plant_asset, plant_pose,
                                                        f"vertical_{row_num}_{step_count}", i, 1)
                        plant_r, plant_g, plant_b = self.randomizer.plant_rgb
                        gym.set_rigid_body_color(env, plant_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                                 gymapi.Vec3(plant_r / 255, plant_g / 255, plant_b / 255))
                    cur_len_list[row_num + 3] += self.randomizer.plant_dist
                # Update step
                step_count += 1
                vertical_row_len += self.randomizer.plant_dist

            if self.randomizer.horizontal_exist:
                # Place 3 horizontal rows
                start_x = vertical_row_len + self.randomizer.horizontal_gap + self.randomizer.row_dist
                start_y = 3.0
                horizontal_row_len = 0.0
                step_count = 0
                while horizontal_row_len <= self.randomizer.horizontal_row_len:
                    # Randomize plants orientation for each step across all 3 rows
                    yaw_angle_list = []
                    for j in range(3):
                        yaw_angle_list.append(random.randint(-180, 180))

                    # Place plants from bottom to top for every row in parallel
                    for row_num in range(-1, 2, 1):
                        plant_pose = gymapi.Transform()
                        plant_pose.p = gymapi.Vec3(start_x + self.randomizer.row_dist * row_num,
                                                   start_y - step_count * self.randomizer.plant_dist, 0.0)
                        plant_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                                   np.radians(yaw_angle_list[row_num + 1]))
                        plant_handle = gym.create_actor(env, plant_asset, plant_pose,
                                                        f"horizontal_{row_num}_{step_count}", i, 1)
                        plant_r, plant_g, plant_b = self.randomizer.plant_rgb
                        gym.set_rigid_body_color(env, plant_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                                 gymapi.Vec3(plant_r / 255, plant_g / 255, plant_b / 255))
                    # Update step
                    step_count += 1
                    horizontal_row_len += self.randomizer.plant_dist

            # Create robot actor
            self.randomizer.randomize_robot()
            robot_pose = gymapi.Transform()
            robot_pose.p = gymapi.Vec3(self.randomizer.trajectory_list[0][1][0],
                                       self.randomizer.trajectory_list[0][1][1], 0.1)
            robot_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                       np.radians(self.randomizer.trajectory_list[0][1][2]))
            robot_handle = gym.create_actor(env, robot_asset, robot_pose, "terrasentia", i, 2)

            # Record environment parameters
            rand_config = self.randomizer.save_config()
            self.recorder.record_env_params(row_len_list, rand_config)

            # Create ground
            self.randomizer.randomize_ground()
            ground_asset = load_ground(gym, sim)
            ground_pose = gymapi.Transform()
            ground_pose.p = gymapi.Vec3(10.0, 0.0, 0.0)
            ground_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                        np.radians(-90.0))  # In environment world frame
            ground_handle = gym.create_actor(env, ground_asset, ground_pose, "ground_plane", i, 1)
            ground_r, ground_g, ground_b = self.randomizer.ground_rgb
            gym.set_rigid_body_color(env, ground_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                     gymapi.Vec3(ground_r / 255, ground_g / 255, ground_b / 255))

            # Specify velocity control mode
            props = gym.get_actor_dof_properties(env, robot_handle)
            props["driveMode"] = [gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL,
                                  gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL]
            props["stiffness"] = [1000.0, 0.0, 1000.0, 0.0,
                                  1000.0, 0.0, 1000.0, 0.0]
            props["damping"] = [200.0, 600.0, 200.0, 600.0,
                                200.0, 600.0, 200.0, 600.0]
            gym.set_actor_dof_properties(env, robot_handle, props)
            pos_targets = np.zeros(8).astype('f')
            gym.set_actor_dof_position_targets(env, robot_handle, pos_targets)

            # Add current environment to env list
            env_handle_list.append(robot_handle)
            env_handle_list.append(ground_handle)
            actor_handles.append(env_handle_list)
            envs.append(env)

        camera_handles = []
        for i in range(num_envs):
            # Create 3 cameras
            camera_handles_env = []
            camera_yaw = random.uniform(-10.0, 10.0)  # Randomize camera yaw angle (consistent across all 3 cameras)
            for cam in range(3):
                # For each camera, randomize a set of params independently
                (horizontal_fov, original_img_width, original_img_height, capture_img_width,
                 capture_img_height, x_offset_diff, y_offset_diff, camera_intrinsic) = self.randomizer.randomize_camera()
                camera_props = gymapi.CameraProperties()
                camera_props.width = capture_img_width
                camera_props.height = capture_img_height
                camera_props.horizontal_fov = horizontal_fov

                # Attach camera to terrasentia
                camera_handle = gym.create_camera_sensor(envs[i], camera_props)
                camera_offset = None
                camera_orient = None
                if cam == 0:
                    # Central camera pose (position and orientation relative to the camera node body frame)
                    cam_x = 0.1 + self.randomizer.cam_x_error
                    cam_y = 0.0 + self.randomizer.cam_y_error
                    cam_z = 0.0 + self.randomizer.cam_z_error
                    cam_yaw = 0.0 + camera_yaw
                    camera_offset = gymapi.Vec3(cam_x, cam_y, cam_z)
                    camera_orient = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(cam_yaw))
                    self.recorder.record_camera_params(cam, original_img_width, original_img_height, camera_intrinsic, cam_x, cam_y, cam_z, cam_yaw)
                elif cam == 1:
                    # Left camera pose
                    cam_x = 0.1 + self.randomizer.cam_x_error
                    cam_y = 0.3 + self.randomizer.cam_y_error
                    cam_z = 0.0 + self.randomizer.cam_z_error
                    cam_yaw = 90.0 + camera_yaw
                    camera_offset = gymapi.Vec3(0.1 + self.randomizer.cam_x_error, 0.3 + self.randomizer.cam_y_error,
                                                0.0 + self.randomizer.cam_z_error)
                    camera_orient = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(90.0 + camera_yaw))
                    self.recorder.record_camera_params(cam, original_img_width, original_img_height, camera_intrinsic, cam_x, cam_y, cam_z, cam_yaw)
                else:
                    # Right camera pose
                    cam_x = 0.1 + self.randomizer.cam_x_error
                    cam_y = -0.3 + self.randomizer.cam_y_error
                    cam_z = 0.0 + self.randomizer.cam_z_error
                    cam_yaw = -90.0 + camera_yaw
                    camera_offset = gymapi.Vec3(0.1 + self.randomizer.cam_x_error, -0.3 + self.randomizer.cam_y_error,
                                                0.0 + self.randomizer.cam_z_error)
                    camera_orient = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(-90.0 + camera_yaw))
                    self.recorder.record_camera_params(cam, original_img_width, original_img_height, camera_intrinsic, cam_x, cam_y, cam_z, cam_yaw)
                robot_handle = actor_handles[i][0]
                body_handle = gym.get_actor_rigid_body_handle(envs[i], robot_handle,
                                                              10)  # Attach to 10th rigid body (camera node)

                gym.attach_camera_to_body(camera_handle, envs[i], body_handle,
                                          gymapi.Transform(camera_offset, camera_orient),
                                          gymapi.FOLLOW_TRANSFORM)
                camera_handles_env.append([camera_handle, camera_intrinsic, original_img_width, original_img_height,
                                           capture_img_width, capture_img_height, x_offset_diff, y_offset_diff])
            camera_handles.append(camera_handles_env)
        
        # Save environment params
        self.recorder.save_env_params(env_idx)

        # Save a copy of the initial state of the environment
        initial_state = np.copy(gym.get_env_rigid_body_states(envs[0], gymapi.STATE_ALL))

        # Run Simulation
        sim_handle = SimHandle(self.randomizer.trajectory_list)
        trajectory = sim_handle.get_cur_trajectory()
        controller = PIDController(trajectory)  # Initialize controller

        frame_count = 0
        while True:
            # Step the physics
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Retrieve robot current pose
            body_states = gym.get_actor_rigid_body_states(envs[0], actor_handles[0][-2], gymapi.STATE_ALL)
            x_pos = body_states["pose"]["p"][1][0]
            y_pos = body_states["pose"]["p"][1][1]
            x_quat = body_states["pose"]["r"][1][0]
            y_quat = body_states["pose"]["r"][1][1]
            z_quat = body_states["pose"]["r"][1][2]
            w_quat = body_states["pose"]["r"][1][3]
            r = R.from_quat([x_quat, y_quat, z_quat, w_quat])
            euler_angles = r.as_euler('zxy', degrees=False)
            yaw = euler_angles[0]

            if sim_handle.is_transition():
                if sim_handle.get_frame_from_transition_start(frame_count) < 2000:
                    # Buffer to allow the robot to stabilize after reset
                    right_angular = 0.0
                    left_angular = 0.0
                    right_linear = 0.0
                    left_linear = 0.0
                    transition_complete = False
                else:
                    right_angular, left_angular, right_linear, left_linear, transition_complete = controller.transition_control(yaw,
                                                                                                                         sim_handle.get_new_yaw())
                if transition_complete:
                    sim_handle.set_transition(False)
            else:
                # Control output
                right_angular, left_angular, right_linear, left_linear = controller.compute_control(x_pos, y_pos, yaw)
            
    
            route_frame_count = sim_handle.get_frame_count()
            if route_frame_count > 2000:
                # Invalid trajectory
                trajectory_idx = sim_handle.get_cur_trajectory()
                shutil.rmtree(f'data/env{env_idx}/route_{trajectory_idx}')

                # Update robot position and controller
                gym.set_env_rigid_body_states(envs[0], initial_state, gymapi.STATE_ALL)
                sim_handle.update_trajectory_idx()

                # End the simulation if run out of trajectories
                if sim_handle.is_end_simulation():
                    break

                sim_handle.record_transition_start(frame_count)
                sim_handle.reset_frame_count()
                new_trajectory = sim_handle.get_cur_trajectory()
                controller.update_trajectory(new_trajectory)

                
            elif controller.is_complete():
                # Current trajectory completes, prepare to start next trajectory
                # Specify new robot position
                x_offset, y_offset = sim_handle.calc_pos_offset()
                for i in range(-16, -1, 1):
                    original_pos = initial_state[i][0][0]
                    new_pos = (original_pos[0] + x_offset, original_pos[1] + y_offset, original_pos[2])
                    initial_state[i][0][0] = new_pos

                # Update robot position and controller
                gym.set_env_rigid_body_states(envs[0], initial_state, gymapi.STATE_ALL)
                sim_handle.update_trajectory_idx()

                # End the simulation if run out of trajectories
                if sim_handle.is_end_simulation():
                    break

                sim_handle.record_transition_start(frame_count)
                sim_handle.reset_frame_count()
                new_trajectory = sim_handle.get_cur_trajectory()
                controller.update_trajectory(new_trajectory)

            # Drive the robot with control output
            vel_targets = np.zeros(8).astype('f')
            vel_targets[1] = np.float32(left_angular * 1.5)  
            vel_targets[3] = np.float32(right_angular * 1.5)
            vel_targets[5] = np.float32(left_angular * 1.5)
            vel_targets[7] = np.float32(right_angular * 1.5)
            gym.set_actor_dof_velocity_targets(envs[0], actor_handles[0][-2], vel_targets)

            # Update the viewer
            gym.step_graphics(sim)

            # Render cameras
            gym.render_all_camera_sensors(sim)
            # Save images
            for i in range(num_envs):
                for j in range(3):
                    # Get image processing parameters
                    camera_handle = camera_handles[i][j][0]
                    camera_intrinsic = camera_handles[i][j][1]  # (3, 3) ndarray
                    capture_img_width = camera_handles[i][j][4]
                    capture_img_height = camera_handles[i][j][5]
                    original_img_width = camera_handles[i][j][2]
                    original_img_height = camera_handles[i][j][3]
                    x_offset_diff = camera_handles[i][j][6]
                    y_offset_diff = camera_handles[i][j][7]
                    if sim_handle.get_frame_count() % 10 == 0:
                        # Retrieve RGB image
                        rgb_image = gym.get_camera_image(sim, envs[i], camera_handle, gymapi.IMAGE_COLOR)
                        rgb_image = np.reshape(rgb_image, (capture_img_height, capture_img_width, 4))
                        # Crop the image for the randomized principal point
                        new_image = rgb_image[
                                    abs(y_offset_diff) + y_offset_diff: abs(y_offset_diff) + original_img_height +
                                    y_offset_diff,
                                    abs(x_offset_diff) + x_offset_diff: abs(x_offset_diff) + original_img_width +
                                    x_offset_diff]
                        image = Image.fromarray(new_image)

                        # Retrieve depth image
                        depth_image = gym.get_camera_image(sim, envs[i], camera_handle, gymapi.IMAGE_DEPTH)
                        depth_image[depth_image == -np.inf] = 0  # -inf to 0 (no depth)
                        depth_image[depth_image < -10] = -10  # Clamp depth image to 10 meters
                        normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))  # Flip the direction so near-objects are light and far objects are dark
                        # Crop the image for the randomized principal point
                        normalized_depth = np.array(normalized_depth)
                        new_depth_image = normalized_depth[
                                          abs(y_offset_diff) + y_offset_diff: abs(y_offset_diff) + original_img_height +
                                          y_offset_diff,
                                          abs(x_offset_diff) + x_offset_diff: abs(x_offset_diff) + original_img_width +
                                          x_offset_diff]
                        # Convert to a pillow image and write it to disk
                        normalized_depth_image = Image.fromarray(new_depth_image.astype(np.uint8), mode="L")

                        # Save images
                        trajectory_idx = sim_handle.get_trajectory_idx()
                        trajectory_frame_count = sim_handle.get_frame_count()
                        image.save(f"data/env{env_idx}/route_{trajectory_idx}/rgb/camera_{j}/frame{trajectory_frame_count}.png")
                        normalized_depth_image.save(f"data/env{env_idx}/route_{trajectory_idx}/depth/camera_{j}/frame{trajectory_frame_count}.png")

                if sim_handle.get_frame_count() % 10 == 0:
                    # Save robot pose
                    trajectory_idx = sim_handle.get_trajectory_idx()
                    robot_pose = [x_pos, y_pos, x_quat, y_quat, z_quat, w_quat]
                    self.recorder.record_robot_pose(trajectory_idx, robot_pose)

            frame_count += 1
            sim_handle.update_frame_count()
        
        self.recorder.save_robot_pose(env_idx)  # Save robot pose data as csv

        print(f"Shutting down env{env_idx}...")
        gym.destroy_sim(sim)  # Shut down simulator
    
    def run(self):
        if not os.path.exists("data"):
            os.mkdir("data")

        env_num = 1
        for i in range(env_num):
            self.randomizer = Randomizer()
            self.recorder = DataRecorder()
            self.run_simulator(i)


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()


