import numpy as np
import math


class SimHandle:
    def __init__(self, trajectory_list):
        self.trajectory_list = trajectory_list
        self.trajectory_num = len(self.trajectory_list)
        self.cur_trajectory = 0
        self.transition = False
        self.transition_start_frame = None
        self.new_yaw = None
        self.frame_count = 0

    def is_transition(self):
        return self.transition

    def set_transition(self, transition_state):
        self.transition = transition_state

    def record_transition_start(self, frame_num):
        self.transition_start_frame = frame_num

    def get_frame_from_transition_start(self, frame_num):
        return frame_num - self.transition_start_frame

    def reset_frame_count(self):
        self.frame_count = 0

    def update_frame_count(self):
        self.frame_count += 1

    def get_frame_count(self):
        return self.frame_count

    def update_trajectory_idx(self):
        self.cur_trajectory += 1

    def get_trajectory_idx(self):
        return self.cur_trajectory

    def get_cur_trajectory(self):
        if self.cur_trajectory < self.trajectory_num:
            return self.trajectory_list[self.cur_trajectory][3]
        else:
            return None

    def get_new_yaw(self):
        return self.new_yaw

    def is_end_simulation(self):
        return self.cur_trajectory >= self.trajectory_num

    def calc_pos_offset(self):
        if self.cur_trajectory + 1 < self.trajectory_num:
            x_offset = self.trajectory_list[self.cur_trajectory + 1][1][0] - self.trajectory_list[self.cur_trajectory][1][0]
            y_offset = self.trajectory_list[self.cur_trajectory + 1][1][1] - self.trajectory_list[self.cur_trajectory][1][1]
            self.new_yaw = self.trajectory_list[self.cur_trajectory + 1][1][2]
        else:
            x_offset = 0.0
            y_offset = 0.0
            self.new_yaw = 0.0
        return x_offset, y_offset



