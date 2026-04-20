
import gymnasium as gym
from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login # To log to our Hugging Face account to be able to upload models to the Hub.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    # Create the environment
# Train env
    env = make_vec_env("LunarLander-v3", n_envs=8)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        ent_coef=0.005,
        n_steps=1024,
        gae_lambda=0.98,
        verbose=1,
    )

    model.learn(total_timesteps=1_500_000)

    model.save("ppo-lunarlander-v3")

    # Eval env
    eval_env = Monitor(gym.make("LunarLander-v3"))

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=50,
        deterministic=True
)
