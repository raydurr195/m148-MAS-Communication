from ray.rllib.policy.policy import Policy
import numpy as np
from gymnasium import spaces

class StaticWerewolfPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        
    def compute_actions(self,
                       obs_batch,
                       state_batches=None,
                       prev_action_batch=None,
                       prev_reward_batch=None,
                       info_batch=None,
                       episodes=None,
                       **kwargs):
        num_players = 7
        # Define observation space structure
        obs_space = spaces.Dict({
            'role': spaces.MultiDiscrete([3 for i in range(num_players)]),
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(num_players, num_players), dtype=np.float64),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(num_players, num_players), dtype=np.float64),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(num_players, num_players), dtype=np.float64),
            'trust': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.float64),
            'life_status': spaces.MultiBinary(num_players),
            'phase': spaces.Box(low=0, high=2, shape=(1,1)),
            'comm_round': spaces.Box(low=0, high=3, shape=(1,1)),
            'day': spaces.Box(low=0, high=14, shape=(1,1))
        })

        actions = []
        for flat_obs in obs_batch:
            # Unflatten observation
            obs = spaces.unflatten(obs_space, flat_obs)
            
            # Find wolf index (self)
            wolf_idx = np.where(obs['role'] == 1)[0][0]
            
            # Get living players excluding self
            living_players = [i for i in range(num_players) if obs['life_status'][i] == 1 and i != wolf_idx]
            
            if not living_players:
                actions.append([0, 0])  # Fallback if no valid targets
                continue
                
            phase = int(obs['phase'])
            if phase == 0:  # Night phase
                # Target players with high trust or those who defended others
                defense_scores = np.sum(obs['public_defense'], axis=1)
                target_scores = obs['trust'] + defense_scores * 0.5
                target_scores[wolf_idx] = -np.inf
                target_scores[obs['life_status'] == 0] = -np.inf
                target = np.argmax(target_scores)
                actions.append([0, target])
                
            elif phase == 1:  # Communication phase
                # Calculate how much each player accused wolves
                wolf_accusations = obs['public_accusation'][:, wolf_idx]
                
                target_scores = wolf_accusations + obs['trust'] * 0.5
                target_scores[wolf_idx] = -np.inf
                target_scores[obs['life_status'] == 0] = -np.inf
                
                action_type = np.random.choice([1, 3], p=[0.7, 0.3])
                target = np.argmax(target_scores)
                actions.append([action_type, target])
                
            else:  # Voting phase
                accusation_counts = np.sum(obs['public_accusation'], axis=0)
                accusation_counts[wolf_idx] = -np.inf
                accusation_counts[obs['life_status'] == 0] = -np.inf
                target = np.argmax(accusation_counts)
                actions.append([0, target])

        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
