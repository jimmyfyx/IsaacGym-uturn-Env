import numpy as np
from simple_pid import PID
import math

# PID further info
# https://github.com/m-lundberg/simple-pid


def pi_clip(angle):
    """Function to map angle error values between [-pi, pi)
    """
    if angle > 0:
        if angle > math.pi:
            return angle - 2*math.pi
    else:
        if angle < -math.pi:
            return angle + 2*math.pi
    return angle


class PIDController:
    pid_v = PID(-0.7, -0.0, -0.0, setpoint=0.0, output_limits=(0, 3))
    pid_w = PID(-10, -0.0, -0.2, setpoint=0.0, output_limits=(-7, 7))
    pid_w.error_map = pi_clip  # Function to map angle error values between -pi and pi.

    def __init__(self, trajectory):
        self.xd = trajectory[0][0]  # (xd, yd) Target position
        self.yd = trajectory[0][1]
        self.dt = 1 / 60  # seconds

        self.base_length = 0.27  # meters
        self.wheel_radius = 0.09  # meters

        self.trajectory = trajectory
        self.num_waypoints = len(trajectory)
        self.cur_waypoint = 0
        self.complete = False

    def is_complete(self):
        return self.complete

    def update_trajectory(self, new_trajectory):
        if new_trajectory is not None:
            self.xd = new_trajectory[0][0]
            self.yd = new_trajectory[0][1]
            self.trajectory = new_trajectory
            self.num_waypoints = len(new_trajectory)
            self.cur_waypoint = 0
            self.complete = False

    def compute_control(self, robot_x, robot_y, robot_yaw):
        if not self.complete:
            distance_error = math.sqrt((self.xd - robot_x) ** 2 + (self.yd - robot_y) ** 2)
            angle_error = math.atan2(self.yd - robot_y, self.xd - robot_x) - robot_yaw

            if distance_error <= 0.1:
                control_w = 2.0
                control_v = 2.0

                # Update waypoint
                self.cur_waypoint += 1
                if self.cur_waypoint >= self.num_waypoints:
                    self.complete = True
                else:
                    self.xd = self.trajectory[self.cur_waypoint][0]
                    self.yd = self.trajectory[self.cur_waypoint][1]
            else:
                control_w = self.pid_w(angle_error, dt=self.dt)
                control_v = self.pid_v(distance_error, dt=self.dt)
        else:
            control_w = 0.0
            control_v = 0.0

        return self.body_to_wheel_vel(control_v, control_w)

    def transition_control(self, robot_yaw, target_yaw):
        angle_error = np.radians(target_yaw) - robot_yaw
        transition_complete = False
        if angle_error > 0.05:
            control_w = self.pid_w(angle_error, dt=self.dt)
            control_v = 0.0
        else:
            control_w = 0.0
            control_v = 0.0
            transition_complete = True
        return self.body_to_wheel_vel(control_v, control_w), transition_complete

    def body_to_wheel_vel(self, body_x_lin, body_z_ang):
        """Use differential drive model to convert body velocities to wheel velocities

        :param body_x_lin: Robot body linear velocity
        :type body_x_lin: float
        :param body_z_ang: Robot angular velocity
        :type body_z_ang: float

        :return right_angular: Angular velocity for right wheel
        :type right_angular: float
        :return left_angular: Angular velocity for left wheel
        :type left_angular: float
        """
        right_linear = (body_z_ang * self.base_length + 2 * body_x_lin) / 2
        left_linear = (2 * body_x_lin - body_z_ang * self.base_length) / 2
        right_angular = right_linear / self.wheel_radius
        left_angular = left_linear / self.wheel_radius
        return right_angular, left_angular, right_linear, left_linear

