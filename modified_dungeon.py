import os
import sys

from gym import spaces

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")

from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    def __init__(self,
                 width: int = 64,
                 height: int = 64,
                 max_rooms: int = 25,
                 min_room_xy: int = 10,
                 max_room_xy: int = 25,
                 observation_size: int = 11,
                 vision_radius: int = 5,
                 max_steps: int = 2000,
                 seed: int = 10):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            vision_radius=vision_radius,
            max_steps=max_steps
        )
        self.seed(seed)
        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3])
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        reward = self.calculate_reward(info)
        return observation[:, :, :-1], reward, done, info

    def reset(self):
        observation = super().reset()
        return observation[:, :, :-1]

    def calculate_reward(self, info):
        return info['new_explored'] * (1 / info['step'])
