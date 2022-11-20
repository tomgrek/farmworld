import argparse

import torch

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.utils import set_random_seed

import farmworld

parser = argparse.ArgumentParser()
parser.add_argument("--no-resume", action="store_true", help="Continue training from modelpolicy.bin")
parser.add_argument("--no-save", action="store_true", default=True, help="Save policy to modelpolicy.bin")
parser.add_argument("--steps", type=int, default=50000, help="Number of timesteps to train")
parser.add_argument("--filename", type=str, default="modelpolicy.bin", help="Name of file to save policy to.")
parser.set_defaults(no_resume=False, no_save=False)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    env = farmworld.env.CustomEnv(geojson="example2.json", screen_size=(700, 700))

    set_random_seed(42)

    # model = PPO("MultiInputPolicy", env, verbose=1)
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        exploration_initial_eps=0.15,
        exploration_fraction=0.5,
        gradient_steps=-1,
        train_freq=4,
        learning_starts=100,
        target_update_interval=2000,
        gamma=0.995,
    )

    if not args.no_resume:
        print(f"Loading policy from {args.filename}")
        model.policy = torch.load(args.filename)
    
    def logger(accum, other):
        for info in accum.get("infos", []):
            if info.get("success", False):
                print(info)
                for k, v in info.get("log_data", {}).items():
                    print(f"{k}: {v}")
                print("-------")

    model.learn(total_timesteps=args.steps, callback=logger)
    if not args.no_save:
        print(f"Saved policy to {args.filename}")
        torch.save(model.policy, args.filename)

    # Demo the policy, with rendering
    import time

    obs = env.reset()
    max_its = 10000
    its = 0
    info = {}
    while not info.get("successfully_harvested", False) and its < max_its:
        its += 1
        now = time.time()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        time_taken = time.time() - now
        if time_taken < (1 / 20):  # 20fps max
            time.sleep((1 / 20) - time_taken)
        if done:
            time.sleep(1)
            obs = env.reset()

    # it should plant at 25, harvest day 50 to get the max gain

    env.close()
