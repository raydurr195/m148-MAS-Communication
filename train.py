from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
import ray
from ray.train import SyncConfig
from werewolf_ev import werewolf
import torch
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import os
from static_wolf_policy import StaticWerewolfPolicy
from ray.rllib.algorithms.callbacks import DefaultCallbacks

ray.init()
#Callbacks

class WerewolfCustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
       
        wrapped_env = base_env.get_sub_environments()[0]  # This is ParallelPettingZooEnv
        real_env = wrapped_env.par_env                    # We access its underlying "par_env", which should be a `werewolf` instance.

        episode.custom_metrics["v_acc_v"] = real_env.v_acc_v
        episode.custom_metrics["v_acc_w"] = real_env.v_acc_w
        episode.custom_metrics["v_def_v"] = real_env.v_def_v
        episode.custom_metrics["v_def_w"] = real_env.v_def_w
        episode.custom_metrics["suicide"] = real_env.suicide
        episode.custom_metrics["win"] = real_env.win
        episode.custom_metrics["vill_reward"] = real_env.vill_reward
        episode.custom_metrics["wolf_reward"] = real_env.wolf_reward

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
    wolves = worker.env.par_env.wolves
    ag_id = int(agent_id.split('_')[1])
    return "werewolf_policy" if ag_id in wolves else "villager_policy"

# Initialize environment
env = werewolf(num_players=6, comm_rounds=4, num_wolf=1, max_days=15)
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
            "num_players": 6, 
            "comm_rounds": 4, 
            "num_wolf": 1, 
            "max_days": 15
        }
    )
    .callbacks(WerewolfCustomMetricsCallback)
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
         #num_gpus_per_learner=1,
         #num_cpus_per_worker = 1
     )
    .multi_agent(
        policies={
            "villager_policy": (None, obs_spaces["player_0"], act_spaces["player_0"], {}),
            "werewolf_policy": (StaticWerewolfPolicy, obs_spaces["player_0"], act_spaces["player_0"], {})
            #"werewolf_policy": (None, obs_spaces["player_0"], act_spaces["player_0"], {})
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["villager_policy"]  # Only train the villager policy
        #policies_to_train["villager_policy", "werewolf_policy"]
    )
    .api_stack(
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
    .env_runners(
        num_env_runners = 6,
        num_cpus_per_env_runner = 1
    )
)

cwd = os.getcwd()
# Training with more iterations
tune.run(
    "PPO",
    name="werewolf_training",
    stop={"training_iteration": 1600},
    config=config.to_dict(),
    storage_path= cwd + "/static",  # Local path in workspace
    checkpoint_freq=50,
    checkpoint_at_end=True,
    #restore = cwd + "/no_static/werewolf_training/PPO_werewolf-v1_d284d_00000_0_2025-02-28_06-39-32/checkpoint_000007"
)

ray.shutdown()