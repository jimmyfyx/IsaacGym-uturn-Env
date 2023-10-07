from isaacgym import gymapi

'''
Loading assets helper functions
'''


def load_terrasentia(gym, sim):
    """Load Terrasentia asset from local directory.

    :param gym: The gym object
    :type gym: gym class
    :param sim: The simulator object
    :type gym: simulator class

    :return asset: The robot asset object in gym API
    :type asset: asset class
    """
    asset_root = "resources/isaac_gym_urdf_files/terra_description/urdf"
    asset_file = "terrasentia2022.urdf"

    # Terrasentia asset options
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return asset


def load_plant(gym, sim, plant_name):
    """Load plants asset from local directory.

    :param gym: The gym object
    :type gym: gym class
    :param sim: The simulator object
    :type gym: simulator class
    :param plant_name: Name of plant asset
    :type plant_name: str

    :return asset: The plant asset object in gym API
    :type asset: asset class
    """
    asset_root = f"resources/isaac_gym_urdf_files/terra_worlds/models/{plant_name}"
    asset_file = "model.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return asset


def load_ground(gym, sim):
    """Load the ground skin from local directory as a URDF file.

    :param gym: The gym object
    :type gym: gym class
    :param sim: The simulator object
    :type gym: simulator class

    :return asset: The ground asset object in gym API
    :type asset: asset class
    """
    asset_root = "resources"
    asset_file = "carpet.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return asset
