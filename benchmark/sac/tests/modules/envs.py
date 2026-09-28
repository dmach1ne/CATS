import gymnasium as gym

def make_ant_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make("Ant-v5", render_mode=render_mode)
        return env
    return _init

def make_lunar_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            render_mode=render_mode
        )
        return env
    return _init

def make_hopper_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make('Hopper-v5', render_mode=render_mode)
        return env
    return _init

def make_pendulum_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        return env
    return _init

def make_walker_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make('Walker2d-v5', render_mode=render_mode)
        return env
    return _init
