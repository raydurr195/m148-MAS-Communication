import gymnasium as gym
from stable_baselines3 import PPO
from werewolf_ev import werewolf

# Create the environment
env = werewolf()

# Define the model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("werewolf_model")


