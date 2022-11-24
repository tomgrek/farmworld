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

    def __init__(self, geojson=None, num_fields=1, screen_size=(500, 500)):
        super(FarmEnv, self).__init__()

        self.geojson = geojson

        # do nothing, plant, or harvest (0, 1, 2)
        # only work on 1 field per day -if you want multiple actions over multiple fields per day, use box.
        self.action_space = spaces.MultiDiscrete((num_fields, 3))

        # 1 row that describes whether planted, whether harvested, day of year, crop height

        self.observation_space = spaces.Dict(
            {
                "planted": spaces.Box(low=0, high=1, shape=(num_fields, 1), dtype=np.int8),
                "harvested": spaces.Box(low=0, high=1, shape=(num_fields, 1), dtype=np.int8),
                "farm_center": spaces.Box(low=-50, high=50, shape=(2,), dtype=np.float32),
                "crop_height": spaces.Box(low=0, high=100, shape=(num_fields, 1), dtype=np.float32),
                "day_of_year": spaces.Box(low=0, high=100, shape=(1, ), dtype=np.float32)
            }
        )

        self.day_of_year = 0  # jan 1st

        self.farm_center_x = (random.random() - 0.5) * 100
        self.farm_center_y = (random.random() - 0.5) * 100

        geojson = get_geojson(self.geojson, num_fields=num_fields)
        polys, render_info, self.grid_size, self.cell_size = poly_and_scaling_from_geojson(geojson["features"], screen_size)
        self.fields = [Field(render_coords=c, feature=c_orig, render_info=r) for c, c_orig, r in zip(polys, geojson["features"], render_info)]

        check_env(self)

        self.renderer = Renderer(screen_size)
        for field in self.fields:
            bg = pygame.Surface(screen_size, pygame.SRCALPHA)
            bg.fill(const.SKY)
            pygame.draw.polygon(bg, const.SOIL, field.render_coords)
            field.covered_area = util.get_covered_area(bg, screen_size, self.grid_size)
        
        self.total_rewards = 0.0

    def gameover(self):
        crop_heights = [[f.crop_height] for f in self.fields]
        crop_heights = np.array(crop_heights, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        observation = {"planted": np.array([[int(field.planted)] for field in self.fields], dtype=np.int8), 
                "harvested": np.array([[int(field.harvested)] for field in self.fields], dtype=np.int8),
                "farm_center": farm_center,
                "crop_height": crop_heights,
                "day_of_year": np.array([float(self.day_of_year)], dtype=np.float32)
               }
        return observation, self.total_rewards + (self.day_of_year / 1000), True, {}
        
    def print_success(self, reward):
        return {"success": True, "log_data": {
                    "message": f"HARVESTED HERE ********************************{reward}"
                    }
                }

    def step(self, action):
        info = {}
        field, action = self.fields[action[0]], action[1]

        if action == 1 and field.planted:
            return self.gameover()

        if field.planted:
            if 25 < self.day_of_year < 50:
                field.crop_height += 1
                if field.plant_date < 25:
                    field.crop_height -= (25 - field.plant_date) * 0.01
            if self.day_of_year > 50:
                field.crop_height = max(field.crop_height - 1, 0)

        self.day_of_year += 1

        reward = self.day_of_year / 1000
        done = False

        if action == 2:  # harvest
            if field.harvested:
                return self.gameover()
            if not field.planted:
                return self.gameover()
            # if self.day_of_year < 25:
            #     return self.gameover()
            field.harvested = True
            field.harvest_date = self.day_of_year
            field.crop_height_at_harvest = field.crop_height
            field.crop_height = 0

            crop_heights = [[f.crop_height] for f in self.fields]
            crop_heights = np.array(crop_heights, dtype=np.float32)
            farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
            farm_center = np.array(farm_center, dtype=np.float32)
            observation = {"planted": np.array([[int(field.planted)] for field in self.fields], dtype=np.int8), 
                    "harvested": np.array([[int(field.harvested)] for field in self.fields], dtype=np.int8),
                    "farm_center": farm_center,
                    "crop_height": crop_heights,
                    "day_of_year": np.array([float(self.day_of_year)], dtype=np.float32)
                }
            self.total_rewards += field.crop_height_at_harvest
            if field.crop_height_at_harvest > 1:
                info = self.print_success(reward)
            return observation, float(reward), False, info

        if action == 1:
            if field.harvested:
                return self.gameover()
            if hasattr(self, "renderer") and self.renderer and not field.planted:
                poly = matplotlib.path.Path(field.render_coords)
                while len(field.plants) < (math.prod(self.renderer.screen_size) * field.covered_area * 0.1) * 0.1:
                    x, y = random.randint(field.render_info["start_x"], field.render_info["end_x"]), random.randint(field.render_info["start_y"], field.render_info["end_y"])
                    if poly.contains_point((x, y)):
                        field.plants.append((x, y))
            field.planted = True
            field.plant_date = self.day_of_year

        if action == 0:
            reward = 0.0  # 0.01
        if self.day_of_year >= 100:
            reward = self.total_rewards
            if self.total_rewards > 0:
                info = self.print_success(reward)
            done = True

        crop_heights = [[f.crop_height] for f in self.fields]
        crop_heights = np.array(crop_heights, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        observation = {"planted": np.array([[int(field.planted)] for field in self.fields], dtype=np.int8), 
                "harvested": np.array([[int(field.harvested)] for field in self.fields], dtype=np.int8),
                "farm_center": farm_center,
                "crop_height": crop_heights,
                "day_of_year": np.array([float(self.day_of_year)], dtype=np.float32)
               }

        return observation, float(reward), done, info

    def reset(self):
        for field in self.fields:
            field.crop_height = 0.0
            field.crop_height_at_harvest = 0.0
            field.planted = False
            field.plant_date = 0
            field.plants = []
            field.harvested = False
            field.harvest_date = 0
        self.day_of_year = 0
        self.total_rewards = 0.0

        self.farm_center_x = (random.random() - 0.5) * 100
        self.farm_center_y = (random.random() - 0.5) * 100

        crop_heights = [[f.crop_height] for f in self.fields]
        crop_heights = np.array(crop_heights, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        
        return {"planted": np.array([[int(field.planted)] for field in self.fields], dtype=np.int8), 
                "harvested": np.array([[int(field.harvested)] for field in self.fields], dtype=np.int8),
                "farm_center": farm_center,
                "crop_height": crop_heights,
                "day_of_year": np.array([float(self.day_of_year)], dtype=np.float32)
               }

    def render(self, max_fps=None, mode="human"):
        if max_fps is not None:
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
        for field in self.fields:
            info += [f"Height: {round(field.crop_height, 3)}"]
            info += [f"Planted: {field.planted} Plant Day: {field.plant_date}"]
            info += [f"Harvested: {field.harvested} Harvest Day: {field.harvest_date} Yield: {round(field.crop_height_at_harvest, 3)}"]
        info += [f"Total Yield: {self.total_rewards}"]
        self.renderer.add_info(bg, info)

        self.renderer.show(bg)

        if max_fps is not None:
            time_taken = time.time() - now
            if time_taken < (1 / max_fps):
                time.sleep((1 / max_fps) - time_taken)

    def close(self):
        self.renderer.quit()
