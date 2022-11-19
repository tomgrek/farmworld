import random

import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

# for rendering
import matplotlib
import pygame

import farmworld


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}
    screen = None
    plants = None
    covered_area = None
    coords = None
    plants = None
    covered_area = None
    font = None

    def __init__(self):
        super(CustomEnv, self).__init__()

        # do nothing, plant, or harvest (0, 1, 2)
        self.action_space = spaces.Discrete(3)

        # 1 row that describes whether planted, whether harvested, day of year, crop height

        self.observation_space = spaces.Dict(
            {
                "planted": spaces.Discrete(2),
                "harvested": spaces.Discrete(2),
                "field_center": spaces.Box(low=-50, high=50, shape=(2,), dtype=np.float32),
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

        self.field_center_x = (random.random() - 0.5) * 100
        self.field_center_y = (random.random() - 0.5) * 100

        check_env(self)

    def gameover(self):
        observation = [float(self.day_of_year), float(self.crop_height)]
        observation = np.array(observation, dtype=np.float32)
        field_center = [float(self.field_center_x), float(self.field_center_y)]
        field_center = np.array(field_center, dtype=np.float32)
        return {"planted": int(self.planted), "harvested": int(self.harvested), "field_center": field_center, "continuous": observation}, 0.0, True, {}

    def step(self, action):

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
            field_center = [float(self.field_center_x), float(self.field_center_y)]
            field_center = np.array(field_center, dtype=np.float32)
            observation = {"planted": int(self.planted), "harvested": int(self.harvested), "field_center": field_center, "continuous": observation}
            reward = self.crop_height_at_harvest
            done = True
            info = {}
            if reward > 1:
                print(f"HARVESTED HERE ********************************{reward}")
                print(f"Planted on {self.plant_date} and harvested on {self.harvest_date}")
                print([[float(self.planted), float(self.harvested), float(self.day_of_year), float(self.crop_height_at_harvest)]])
                info = {"successfully_harvested": True}
            return observation, float(reward), done, info

        if action == 1:
            if self.harvested:
                return self.gameover()
            self.planted = True
            self.plant_date = self.day_of_year

        if self.day_of_year == 100:
            reward = -1#self.crop_height_at_harvest
            done = True
            if reward > 1:
                print(f"DIDIT********************************{reward}")
                print(f"Planted on {self.plant_date} and harvested on {self.harvest_date}")
                print([[float(self.planted), float(self.harvested), float(self.day_of_year), float(self.crop_height_at_harvest)]])
        if action == 0:
            reward = 0.0  # 0.01

        observation = [float(self.day_of_year), float(self.crop_height)]
        observation = np.array(observation, dtype=np.float32)
        field_center = [float(self.field_center_x), float(self.field_center_y)]
        field_center = np.array(field_center, dtype=np.float32)
        observation = {"planted": int(self.planted), "harvested": int(self.harvested), "field_center": field_center, "continuous": observation}

        return observation, float(reward), done, {}

    def reset(self):
        self.crop_height = 0.0
        self.crop_height_at_harvest = 0.0
        self.planted = False
        self.plant_date = 0
        self.harvested = False
        self.harvest_date = 0
        self.day_of_year = 0

        self.field_center_x = (random.random() - 0.5) * 100
        self.field_center_y = (random.random() - 0.5) * 100

        observation = [float(self.day_of_year), float(self.crop_height)]
        observation = np.array(observation, dtype=np.float32)
        field_center = [float(self.field_center_x), float(self.field_center_y)]
        field_center = np.array(field_center, dtype=np.float32)
        observation = {"planted": int(self.planted), "harvested": int(self.harvested), "field_center": field_center, "continuous": observation}
        return observation

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            if self.font is None:
                self.font = pygame.font.SysFont("dejavusans", 18)
            self.screen = pygame.display.set_mode([500, 500])
            geojson = farmworld.geojson.get_geojson("example2.json")
            self.coords = farmworld.geojson.poly_from_geojson(geojson)

        bg = pygame.Surface((500, 500), pygame.SRCALPHA)
        bg.fill((190, 200, 255))

        fg = pygame.Surface((500, 500), pygame.SRCALPHA).convert_alpha()
        fg.fill((0, 0, 0, 0))

        pygame.draw.polygon(bg, (100, 90, 70), self.coords)

        if self.covered_area is None:
            buf = bg.get_buffer().raw
            bgcol = 0
            for px in range(0, len(buf), 4):
                b, g, r, a = buf[px : px + 4]
                if (r, g, b) == (190, 200, 255):
                    bgcol += 1
            self.covered_area = ((500 * 500) - bgcol) / (500 * 500)

        if self.plants is None:
            poly = matplotlib.path.Path(self.coords)
            self.plants = []
            while len(self.plants) < (500 * 500 * self.covered_area * 0.1) * 0.1:
                x, y = random.randint(0, 500), random.randint(0, 500)
                if poly.contains_point((x, y)):
                    self.plants.append((x, y))

        fg.fill((255, 255, 255, 0))
        for x, y in self.plants:
            color = (20, 220, 50) if self.crop_height >= 0 else (220, 50, 20)
            pygame.draw.line(fg, color, start_pos=(x, y - self.crop_height), end_pos=(x, y), width=2)
        bg.blit(fg, (0, 0), None)

        info = [f"Day: {self.day_of_year} Height: {round(self.crop_height, 3)}"]
        if self.planted:
            info += [f"Plant Day: {self.plant_date}"]
        if self.harvested:
            info += [f"Harvest Day: {self.harvest_date} Yield: {round(self.crop_height_at_harvest, 3)}"]
        for i, text in enumerate(info):
            bitmap = self.font.render(text, True, (0, 0, 0))
            bg.blit(bitmap, (0, i*18))

        self.screen.blit(bg, (0, 0))
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
