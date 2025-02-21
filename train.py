from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.train import SyncConfig
from werewolf_ev import werewolf
import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import os
from static_wolf_policy import StaticWerewolfPolicy

env_name = "werewolf-v1"

# Creates the Werewolf environment
def env_creator(config):
    env = werewolf(num_players=config["num_players"], comm_rounds=config["comm_rounds"], num_wolf=config["num_wolf"], max_days=config["max_days"])
    env.reset()  # Ensure agents are properly initialized
    para_env = ParallelPettingZooEnv(env)
    para_env.agents = para_env.par_env.agents
    para_env.possible_agents = para_env.par_env.possible_agents
    return para_env

register_env(env_name, lambda config: env_creator(config))

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "villager_policy" if "villager" in agent_id else "werewolf_policy"

# Initialize environment
env = werewolf(num_players=7, comm_rounds=4, num_wolf=1, max_days=15)
obs_spaces = {agent: env.observation_space(agent) for agent in env.possible_agents}
act_spaces = {agent: env.action_space(agent) for agent in env.possible_agents}

rl_module_spec = RLModuleSpec(
    observation_space=obs_spaces["player_0"],
    action_space=act_spaces["player_0"],
    #catalog_class=ModelCatalog,
    model_config={
        "fcnet_hiddens": [256, 256], 
        "fcnet_activation": "relu"
    }
)

# Update policy specification
config = (
    PPOConfig()
    .environment(
        env=env_name, 
        env_config={
            "num_players": 7, 
            "comm_rounds": 4, 
            "num_wolf": 1, 
            "max_days": 15
        }
    )
    .framework("torch")
    .training(
        train_batch_size=4096,
        minibatch_size=256,
        num_epochs=20,
        lr=1e-4,
        lambda_=0.95,
        vf_loss_coeff=1.0,
        entropy_coeff=0.01,
        clip_param=0.2
    )
    .resources(
        num_gpus=0,
        num_cpus=4,
    )
    .multi_agent(
        policies={
            "villager_policy": (None, obs_spaces["player_0"], act_spaces["player_0"], {}),
            "werewolf_policy": (StaticWerewolfPolicy, obs_spaces["player_0"], act_spaces["player_0"], {})
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["villager_policy"]  # Only train the villager policy
    )
)

# Training with more iterations
tune.run(
    "PPO",
    name="werewolf_training",
    stop={"training_iteration": 10000},
    config=config.to_dict(),
    num_samples=1,  # Number of times to sample from the hyperparameter space.
    storage_path="/workspaces/m148-MAS-Communication/training_results",  # Local path in workspace
    checkpoint_freq=100,
    checkpoint_at_end=True,
    # Remove the resources_per_trial parameter since PPO handles this automatically
)
