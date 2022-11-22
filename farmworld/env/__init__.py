import math
import random

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

# for rendering
import matplotlib
import pygame

import farmworld.const as const
from farmworld.geojson import get_geojson, poly_and_scaling_from_geojson, util
from farmworld.renderer import Renderer

class FarmEnv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, geojson=None, num_fields=1, screen_size=(500, 500)):
        super(FarmEnv, self).__init__()

        self.geojson = geojson
        self.screen_size = screen_size

        # do nothing, plant, or harvest (0, 1, 2)
        self.action_space = spaces.Discrete(3)

        # 1 row that describes whether planted, whether harvested, day of year, crop height

        self.observation_space = spaces.Dict(
            {
                "planted": spaces.Discrete(2),
                "harvested": spaces.Discrete(2),
                "farm_center": spaces.Box(low=-50, high=50, shape=(2,), dtype=np.float32),
                "continuous": spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),
            }
        )

        self.planted = False
        self.plant_date = 0
        self.harvested = False
        self.harvest_date = 0
        self.crop_height = 0.0
        self.crop_height_at_harvest = 0.0
        self.day_of_year = 0  # jan 1st

        self.farm_center_x = (random.random() - 0.5) * 100
        self.farm_center_y = (random.random() - 0.5) * 100

        check_env(self)

        geojson = get_geojson(self.geojson, num_fields=num_fields)
        polys, render_info, self.grid_size, self.cell_size = poly_and_scaling_from_geojson(geojson["features"], self.screen_size)
        self.fields = [{"render_coords": c, "feature": c_orig, "render_info": r} for c, c_orig, r in zip(polys, geojson["features"], render_info)]
        
        self.renderer = Renderer(self.screen_size)

    def gameover(self):
        observation = [float(self.day_of_year), float(self.crop_height)]
        observation = np.array(observation, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        return {"planted": int(self.planted), "harvested": int(self.harvested), "farm_center": farm_center, "continuous": observation}, 0.0, True, {}

    def print_success(self, reward):
        return {"success": True, "log_data": {
                    "message": f"HARVESTED HERE ********************************{reward}",
                    "plant_and_harvest": f"Planted on {self.plant_date} and harvested on {self.harvest_date}",
                    "other": [[float(self.planted), float(self.harvested), float(self.day_of_year), float(self.crop_height_at_harvest)]]
                    }
                }

    def step(self, action):
        info = {}

        if action == 1 and self.planted:
            return self.gameover()

        if self.planted:
            if 25 < self.day_of_year < 50:
                self.crop_height += 1
                if self.plant_date < 25:
                    self.crop_height -= (25 - self.plant_date) * 0.01
            if self.day_of_year > 50:
                self.crop_height = max(self.crop_height - 1, 0)

        self.day_of_year += 1

        reward = 0.0
        done = False

        if action == 2:  # harvest
            if self.harvested:
                return self.gameover()
            if not self.planted:
                return self.gameover()
            # if self.day_of_year < 25:
            #     return self.gameover()
            self.harvested = True
            self.harvest_date = self.day_of_year
            self.crop_height_at_harvest = self.crop_height
            self.crop_height = 0
            # self.planted = False
            observation = [float(self.day_of_year), float(self.crop_height)]
            observation = np.array(observation, dtype=np.float32)
            farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
            farm_center = np.array(farm_center, dtype=np.float32)
            observation = {"planted": int(self.planted), "harvested": int(self.harvested), "farm_center": farm_center, "continuous": observation}
            reward = self.crop_height_at_harvest
            done = True
            if reward > 1:
                info = self.print_success(reward)
            return observation, float(reward), done, info

        if action == 1:
            if self.harvested:
                return self.gameover()
            self.planted = True
            self.plant_date = self.day_of_year

        if self.day_of_year == 100:
            reward = -1  # self.crop_height_at_harvest
            done = True
        if action == 0:
            reward = 0.0  # 0.01

        observation = [float(self.day_of_year), float(self.crop_height)]
        observation = np.array(observation, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        observation = {"planted": int(self.planted), "harvested": int(self.harvested), "farm_center": farm_center, "continuous": observation}

        return observation, float(reward), done, info

    def reset(self):
        self.crop_height = 0.0
        self.crop_height_at_harvest = 0.0
        self.planted = False
        self.plant_date = 0
        self.harvested = False
        self.harvest_date = 0
        self.day_of_year = 0

        self.farm_center_x = (random.random() - 0.5) * 100
        self.farm_center_y = (random.random() - 0.5) * 100

        observation = [float(self.day_of_year), float(self.crop_height)]
        observation = np.array(observation, dtype=np.float32)
        farm_center = [float(self.farm_center_x), float(self.farm_center_y)]
        farm_center = np.array(farm_center, dtype=np.float32)
        observation = {"planted": int(self.planted), "harvested": int(self.harvested), "farm_center": farm_center, "continuous": observation}
        return observation

    def render(self, mode="human"):

        bg = pygame.Surface(self.screen_size, pygame.SRCALPHA)
        bg.fill(const.SKY)

        fg = pygame.Surface(self.screen_size, pygame.SRCALPHA).convert_alpha()
        fg.fill((255, 255, 255, 0))

        for field in self.fields:
            pygame.draw.polygon(bg, const.SOIL, field["render_coords"])
            if not field.get("covered_area", False):
                field["covered_area"] = util.get_covered_area(bg, self.screen_size)
            if not field.get("plants", False):
                poly = matplotlib.path.Path(field["render_coords"])
                field["plants"] = []
                while len(field["plants"]) < (math.prod(self.screen_size) * field["covered_area"] * 0.1) * 0.1:
                    x, y = random.randint(field["render_info"]["start_x"], field["render_info"]["end_x"]), random.randint(field["render_info"]["start_y"], field["render_info"]["end_y"])
                    if poly.contains_point((x, y)):
                        field["plants"].append((x, y))

            for x, y in field["plants"]:
                color = (20, 220, 50) if self.crop_height >= 0 else (220, 50, 20)
                pygame.draw.line(fg, color, start_pos=(x, y - self.crop_height), end_pos=(x, y), width=2)
        
        # draw grid lines
        for cell in range(1, self.grid_size):
            pygame.draw.line(fg, (0, 0, 0, 128), (cell * self.cell_size[0], 0), (cell * self.cell_size[0], self.screen_size[1]))
            pygame.draw.line(fg, (0, 0, 0, 128), (0, cell * self.cell_size[1]), (self.screen_size[0], cell * self.cell_size[1]))
        
        bg.blit(fg, (0, 0), None)

        info = [f"Day: {self.day_of_year} Height: {round(self.crop_height, 3)}"]
        if self.planted:
            info += [f"Plant Day: {self.plant_date}"]
        if self.harvested:
            info += [f"Harvest Day: {self.harvest_date} Yield: {round(self.crop_height_at_harvest, 3)}"]
        
        self.renderer.add_info(bg, info)

        self.renderer.show(bg)

    def close(self):
        self.renderer.quit()
