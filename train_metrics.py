from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from werewolf_ev import werewolf
import os

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.algorithms.callbacks import DefaultCallbacks

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
    return "villager_policy" if "villager" in agent_id else "werewolf_policy"

# Initialize environment
env = werewolf(num_players=7, comm_rounds=4, num_wolf=1, max_days=15)
obs_spaces = {agent: env.observation_space(agent) for agent in env.possible_agents}
act_spaces = {agent: env.action_space(agent) for agent in env.possible_agents}

# Directory to save training results
save_path = "/Users/sia/Library/CloudStorage/OneDrive-UCLAITServices/math m148/werewolf_training"
os.makedirs(save_path, exist_ok=True)




# Update policy specification
config = (
    PPOConfig()
    .environment(env=env_name, env_config={"num_players": 7, "comm_rounds": 4, "num_wolf": 1, "max_days": 15})
    .callbacks(WerewolfCustomMetricsCallback)  # Include the custom callback
    .framework("torch")  # Use PyTorch backend
    .env_runners(num_env_runners=4)  # Increase workers for faster training
    .training( 
        train_batch_size=4096,  # Increased batch size for efficiency
        minibatch_size=256,  
        num_epochs=20  # More training epochs
    )
    .multi_agent( 
        policies={
            "villager_policy": (None, obs_spaces["player_0"], act_spaces["player_0"], {}),
            "werewolf_policy": (None, obs_spaces["player_0"], act_spaces["player_0"], {})
        },
        policy_mapping_fn=policy_mapping_fn
    )
    .api_stack(
        enable_rl_module_and_learner=False, 
        enable_env_runner_and_connector_v2=False
    )
)



# Training
tune.run(
    "PPO",
    name="werewolf_training",
    stop={"training_iteration": 100}, 
    config=config.to_dict(),
    checkpoint_at_end=True,
    storage_path=save_path  
)
