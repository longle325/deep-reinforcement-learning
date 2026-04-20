import gymnasium as gym

env = gym.make("LunarLander-v3")

obs, info = env.reset(seed=2811)

for _ in range(20):
    action = env.action_space.sample()
    print(f"Action taken: {action}")

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Environment is reset")
        obs, info = env.reset()

    env.close()