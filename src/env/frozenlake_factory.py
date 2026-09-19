import gymnasium as gym
from .frozenlake_view_wrapper import FrozenLakeViewSystem

def make_env(render_mode=None, map_name="8x8", is_slippery=True, desc=None, success_rate=None, reward_schedule=None, size=None):
    """Factory FrozenLake slippery, analogo a factory.make_env per DoorKey."""
    # size override map_name: 4->4x4, 8->8x8
    if size is not None:
        map_name = f"{size}x{size}"
    kwargs: dict = {"render_mode": render_mode, "is_slippery": is_slippery}
    if desc is not None:
        kwargs["desc"] = desc
        kwargs["map_name"] = None
    else:
        kwargs["map_name"] = map_name
    if success_rate is not None:
        kwargs["success_rate"] = success_rate
    if reward_schedule is not None:
        kwargs["reward_schedule"] = reward_schedule
    env = gym.make("FrozenLake-v1", **kwargs)
    env = FrozenLakeViewSystem(env)
    return env
