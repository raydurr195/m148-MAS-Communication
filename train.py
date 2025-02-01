from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from werewolf_ev import Werewolf

# Creates the Werewolf environment
def env_creator(config):
    return Werewolf(num_agents=config["num_agents"], comm_rounds=config["comm_rounds"], num_wolf=config["num_wolf"], max_days=config["max_days"])

env_name = "Werewolf-v1"
register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

# Creates an instants
env = Werewolf(num_agents=7, comm_rounds=4, num_wolf=1, max_days=15)
obs_spaces = [env.observation_space(agent) for agent in env.possible_agents]
act_spaces = [env.action_space(agent) for agent in env.possible_agents]

# Settings used
config = (
    PPOConfig()
    .environment(env=env_name, env_config={"num_agents": 7, "comm_rounds": 4, "num_wolf": 1, "max_days": 15})
    .framework("torch")  # Pytorch
    .env_runners(num_env_runners=2)  # Number of workers
    .training( # Training settings
        train_batch_size=1024,
        minibatch_size=128,  
        num_epochs=10  
    )
    .multi_agent( # Types of policies
        policies={
            "villager_policy": (None, obs_spaces[0], act_spaces[0], {}),
            "werewolf_policy": (None, obs_spaces[0], act_spaces[0], {})
        },
        
        policy_mapping_fn=lambda agent_id, **kwargs: "villager_policy" if "villager" in agent_id else "werewolf_policy"
    )
)

# Training
analysis = tune.run(
    "PPO",
    name = "werewolf_training",
    stop = {"training_iteration": 100},
    config=config.to_dict(),
    checkpoint_at_end = True,
    storage_path = "/Users/------/Desktop/Code/werewolf agent/ray_results" # Add storage path of trained data
)