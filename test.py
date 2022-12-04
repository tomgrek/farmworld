import argparse

from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO, TRPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.utils import set_random_seed

import farmworld

parser = argparse.ArgumentParser()
parser.add_argument("--no-resume", action="store_true", help="Continue training from modelpolicy.bin")
parser.add_argument("--no-save", action="store_true", default=True, help="Save policy to modelpolicy.bin")
parser.add_argument("--steps", type=int, default=50000, help="Number of timesteps to train")
parser.add_argument("--eps", type=float, default=0.4, help="Initial exploration")
parser.add_argument("--filename", type=str, default="modelpolicy.bin", help="Name of file to save policy to.")
parser.set_defaults(no_resume=False, no_save=False)

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    GAMMA = 0.995

    # env = farmworld.env.FarmEnv(screen_size=(700, 700), num_fields=4, max_fps=40) # geojson="example2.json", 
    # env = VecNormalize(DummyVecEnv([lambda: env]), gamma=GAMMA, clip_reward=1000, clip_obs=1000)
    set_random_seed(42)

    #model = PPO("MultiInputPolicy", env, verbose=1, gamma=0.999, ent_coef=0.1)#0.001)
    #model = TRPO("MultiInputPolicy", env, verbose=1)
    #model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1, gamma=0.95, ent_coef=0.0, learning_rate=0.003)#0.1)    

    if not args.no_resume:
        env = farmworld.env.FarmEnv(screen_size=(700, 700), num_fields=4, max_fps=40)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(args.filename + ".env", env)
        print(f"Loaded vec env from {args.filename + '.env'}")
    else:
        env = farmworld.env.FarmEnv(screen_size=(700, 700), num_fields=4, max_fps=40) # geojson="example2.json", 
        env = VecNormalize(DummyVecEnv([lambda: env]), gamma=GAMMA, clip_reward=1000, clip_obs=1000)
    
    model = DQN("MultiInputPolicy", env, learning_starts=10000, gamma=GAMMA, exploration_fraction=0.5, exploration_initial_eps=args.eps, verbose=1)
    if not args.no_resume:
        print(f"Loaded policy from {args.filename}")
        model.set_parameters(args.filename)
        model.set_env(env)
    
    def logger(accum, other):
        for info in accum.get("infos", []):
            if info.get("success", False):
                print(info)
                for k, v in info.get("log_data", {}).items():
                    print(f"{k}: {v}")
                print("-------")

    if args.steps > 0:
        model.learn(total_timesteps=args.steps, callback=logger)
    if not args.no_save:
        model.save(args.filename)
        model.get_env().save(args.filename + '.env')
        print(f"Saved policy to {args.filename} and vec env to {args.filename + '.env'}")
    
    # Demo the policy, with rendering
    env.training = False
    env.norm_reward = False
    obs = env.reset()
    max_its = 20000000
    its = 0
    info = {}
    done = 1; _states = None; reward = -10
    while its < max_its: # not info.get("success", False) and reward < 2 and 
        its += 1
        action, _states = model.predict(obs, state=_states, episode_start=done, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            print(f"{reward}(normd):{env.get_original_reward()}(orig)")
            import time; time.sleep(0.2)
            obs = env.reset()

    env.close()
