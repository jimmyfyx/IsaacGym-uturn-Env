# isaacgym_uturn_git

## Create Isaac Gym Environment
Follow `isaac_gym_installation.pdf` to create the environment.

## Create Workspace
Clone the repo to anywhere you like in your workspace. Activate the `conda` environment.
<br/>
```commandLine
$ conda activate rlgpu
$ (rlgpu) cd isaac_gym_uturn_git
```
Download the assets and place them in the folder `isaac_gym_uturn_git/resources`.

## Run Simulator
To run the simulator in headless mode:
```commandline
$ (rlgpu) python simulator_cpu.py --headless
```
To run the simulator in graphical mode:
```commandline
$ (rlgpu) python simulator_cpu.py
```
The other two arguments `num_envs` and `num_traj` are available in both mode.

## Change Plant Assets
In `randomization.py`, under function`randomize_plant()`, modify `self.plant_name`.
