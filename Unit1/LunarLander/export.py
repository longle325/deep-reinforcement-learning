import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from huggingface_sb3 import package_to_hub
if __name__ == "__main__":
    env_id = "LunarLander-v3"
    model_name = "ppo-lunarlander-v2"
    model_architecture = "PPO"
    model = PPO.load(model_name) 
    repo_id = "longle325/ppo-LunarLander-v3" # Change with your repo id, you can't push with mine 😄
    commit_message = "Upload PPO LunarLander-v3 trained agent"

    # Create the evaluation env and set the render_mode="rgb_array"
    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

    # PLACE the package_to_hub function you've just filled here
    package_to_hub(model=model, # Our trained model
                model_name=model_name, # The name of our trained model
                model_architecture=model_architecture, # The model architecture we used: in our case PPO
                env_id=env_id, # Name of the environment
                eval_env=eval_env, # Evaluation Environment
                repo_id=repo_id, # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
                commit_message=commit_message)
