# IsaacGym Environment for Agricultural Robots Row-turning Policy Demonstration 

## Overview
This repository contains an IsaacGym environment designed for demonstrating row-turning policies for under-canopy agricultural field robots. The simulator provides a platform to facilitate collections of policies for navigating agricultural fields, specifically efficient row-turning maneuvers.

## Environment Description
The simulator will be launched with multiple parallel environments, where each environment is initiated with a grid of crops rows (specified by the plant type) and a robot agent (Terrasentia). When the simulator is running, the robot in each environment will follow its own randomly generated trajectory to complete round of demonstrations. 

## Assets and Credits
All the assets used in the simulator are owned by the **DASLAB at UIUC**. They are used here with permission for educational and research purposes. Assets can be obtained by contacting the **DASLAB at UIUC**.

## Installation and Setup
### Prerequisites
- Python 3.x
- NVIDIA GPU with CUDA support
- IsaacGym (Follow the included `isaac_gym_installation.pdf` for setup)

### Create Workspace
Clone the repo to anywhere you like in your workspace. Activate the `conda` environment.
<br/>
```commandLine
$ conda activate rlgpu
$ (rlgpu) cd IsaacGym-uturn-Env
```
Download the assets and place them in the folder `IsaacGym-uturn-Env/resources`.

## Usage
### Running the Simulator
To run the simulator in headless mode:
```commandline
$ (rlgpu) python simulator.py --headless
```
To run the simulator in graphical mode:
```commandline
$ (rlgpu) python simulator.py
```
The other two arguments `num_envs` and `num_traj` are available in both mode.

### Modify Plant Assets
In `randomization.py`, under function`randomize_plant()`, modify `self.plant_name`.

## Scripts Functionality
- **`assets_loader.py`**: Manages loading of assets like robots, plants, and environment elements in the IsaacGym simulator
- **`controller.py`**: Implements control logic for the robot's movement, handling actuators for row-turning and navigation
- **`data_recorder.py`**: Records data during the simulation, such as robot trajectory, turning accuracy, and collisions
- **`randomization.py`**: Handles environment randomization, allowing variations in plant assets and row configurations. Modify the function `randomize_plant()` to change `self.plant_name` for different assets
- **`simulation_handle.py`**: Provides utility functions for interfacing with the IsaacGym environment, managing simulation states and interactions
- **`simulator.py`**: Main script to launch the simulation. Supports both headless and graphical modes. This script should be used for running the environment

