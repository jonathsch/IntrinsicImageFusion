import os


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', -1))

