import contextlib
import math
import random
import time

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

# for rendering
import matplotlib
with contextlib.redirect_stdout(None):
    import pygame

import farmworld.const as const
from farmworld.geojson import get_geojson, poly_and_scaling_from_geojson, util
from farmworld.renderer import Renderer
from farmworld.types import Field

class FarmEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, geojson=None, num_fields=1, screen_size=(500, 500), max_fps=50):
        super(FarmEnv, self).__init__()

        self.geojson = geojson
        self.max_fps = max_fps

        # plant, or harvest (0, 1) + 1 (do nothing on all fields)
        # only work on 1 field per day -if you want multiple actions over multiple fields per day, use box.
        #self.action_space = spaces.MultiDiscrete((num_fields, 3))
        self.num_actions = 2
        self.action_space = spaces.Discrete(1 + (num_fields * self.num_actions))
        self.length_of_year = 100

        # 1 row that describes whether planted, whether harvested, day of year, crop height

        self.observation_space = spaces.Dict(
            {
                "farm_center": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "crop_height": spaces.Box(low=-1, high=100, shape=(num_fields,), dtype=np.float32),
                "day_of_year": spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32),
                "current_yield": spaces.Box(low=0, high=1000, shape=(1, ), dtype=np.float32),
            }
        )

        self.day_of_year = 0  # jan 1st
        self.total_rewards = 0.0
        self.num_fields = num_fields

        self.farm_center_x = (random.random() - 0.5) * 2
        self.farm_center_y = (random.random() - 0.5) * 2

        geojson = get_geojson(self.geojson, num_fields=num_fields)
        polys, render_info, self.grid_size, self.cell_size = poly_and_scaling_from_geojson(geojson["features"], screen_size)
        self.fields = [Field(render_coords=c, feature=c_orig, render_info=r) for c, c_orig, r in zip(polys, geojson["features"], render_info)]
        for i, field in enumerate(self.fields):
            field.idx = i

        check_env(self)
        self.reset()

        self.renderer = Renderer(screen_size)
        for field in self.fields:
            bg = pygame.Surface(screen_size, pygame.SRCALPHA)
            bg.fill(const.SKY)
            pygame.draw.polygon(bg, const.SOIL, field.render_coords)
            field.covered_area = util.get_covered_area(bg, screen_size, self.grid_size)
    
    @property
    def current_observation(self):
        crop_heights = [[f.crop_height] for f in self.fields]
        crop_heights = np.array(crop_heights, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        observation = {
                "farm_center": farm_center,
                "crop_height": crop_heights.flatten(),
                "day_of_year": np.array([self.day_of_year], dtype=np.float32),
                "current_yield": np.array([float(self.total_rewards)], dtype=np.float32),
               }
        return observation

    def gameover(self, illegal_move=True):
        rewards = 0.0
        info = {}
        if not illegal_move:
            rewards += self.total_rewards
        else:
            rewards += self.total_rewards / 10
        if self.day_of_year >= 100:
            rewards *= 2
            info = self.print_success(rewards)
            info["TimeLimit.truncated"] = True
        return self.current_observation, rewards, True, info
        
    def print_success(self, reward):
        return {"success": True, "log_data": {
                    "message": f"HARVESTED HERE ********************************{reward}"
                    }
                }
    
    def deconstruct_action(self, action_):
        # NEXT: I can cut this down - each field has only 2 actions in reality, plus
        # a general "zero"/do nothing action. so totally 9 instead of 12.
        if action_ == 0:
            return (None, None)
        action_ -= 1
        field = int(math.floor(action_ / self.num_actions))
        action = action_ % self.num_actions
        field = self.fields[field]
        # for ppo/with multidiscrete action space, easier: field, action = self.fields[action[0]], action[1]
        return field, action

    def step(self, action_):
        # General actions per timestep
        for field_ in self.fields:
            field_.grow()
        self.day_of_year += 1

        # Return values
        info = {}

        # Policy initiated actions
        field, action = self.deconstruct_action(action_)

        if action == 1:  # harvest
            if not field.is_planted:
                return self.gameover(illegal_move=True)

            reward_ = field.harvest(self.day_of_year)
            self.total_rewards += reward_

        if action == 0:
            if field.is_planted:
                return self.gameover(illegal_move=True)
            if hasattr(self, "renderer") and self.renderer and not field.is_planted:
                poly = matplotlib.path.Path(field.render_coords)
                while len(field.plants) < (math.prod(self.renderer.screen_size) * field.covered_area * 0.1) * 0.1:
                    x, y = random.randint(field.render_info["start_x"], field.render_info["end_x"]), random.randint(field.render_info["start_y"], field.render_info["end_y"])
                    if poly.contains_point((x, y)):
                        field.plants.append((x, y))
            field.plant(self.day_of_year)

        if self.day_of_year >= 100:
            return self.gameover(illegal_move=False)

        return self.current_observation, 0.0, False, info

    def reset(self):
        if self.total_rewards > 0:
            self.print_success(self.total_rewards)
        for field in self.fields:
            field.reset()
        self.day_of_year = 0
        self.total_rewards = 0.0

        self.farm_center_x = (random.random() - 0.5) * 2
        self.farm_center_y = (random.random() - 0.5) * 2

        return self.current_observation

    def render(self, mode="human"):
        now = time.time()
        bg = self.renderer.get_surface(const.SKY)
        fg = self.renderer.get_surface(const.TRANSPARENT_WHITE)

        for field in self.fields:
            pygame.draw.polygon(bg, const.SOIL, field.render_coords)
            for x, y in field.plants:
                color = (20, 220, 50) if field.crop_height >= 0 else (220, 50, 20)
                pygame.draw.line(fg, color, start_pos=(x, y - field.crop_height), end_pos=(x, y), width=2)
        
        # draw grid lines
        for cell in range(1, self.grid_size):
            pygame.draw.line(fg, const.TRANSLUCENT_GREY, (cell * self.cell_size[0], 0), (cell * self.cell_size[0], self.renderer.screen_size[1]))
            pygame.draw.line(fg, const.TRANSLUCENT_GREY, (0, cell * self.cell_size[1]), (self.renderer.screen_size[0], cell * self.cell_size[1]))
        
        bg.blit(fg, (0, 0), None)

        info = [f"Day: {self.day_of_year}"]
        info += [f"Total Yield: {self.total_rewards}"]
        self.renderer.add_global_info(bg, info)

        for field in self.fields:
            info = [f"{field.idx}. Height: {round(field.crop_height, 3)}"]
            #info += [f"Plant Day: {field.plant_date}"]
            #info += [f"Harvest Day: {field.harvest_date} Yield: {round(field.crop_height_at_harvest, 3)}"]
            self.renderer.add_field_info(bg, field, info)
        

        self.renderer.show(bg)

        time_taken = time.time() - now
        if time_taken < (1 / self.max_fps):
            time.sleep((1 / self.max_fps) - time_taken)

    def close(self):
        self.renderer.quit()
