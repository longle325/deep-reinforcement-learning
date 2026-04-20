import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    env.reset(seed=2811)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample()) # Get a random observation
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample()) # Take a random action

