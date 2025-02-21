from ray.rllib.policy.policy import Policy
import numpy as np

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
        num_obs = len(obs_batch)
        actions = []
        
        for obs in obs_batch:
            # Extract relevant information from flattened observation
            num_players = 7
            role_start = 0
            accusation_start = 7
            defense_start = 56
            trust_start = 105
            life_status_start = 167
            phase_index = 187
            
            # Reshape matrices from flattened form
            roles = obs[role_start:accusation_start]
            accusations = obs[accusation_start:defense_start].reshape(num_players, num_players)
            defenses = obs[defense_start:trust_start].reshape(num_players, num_players)
            trust = obs[trust_start:trust_start + num_players]
            life_status = obs[life_status_start:life_status_start + num_players]
            phase = int(obs[phase_index])
            
            # Find wolf index (self)
            wolf_idx = np.where(roles == 1)[0][0]
            
            # Get living players excluding self
            living_players = [i for i in range(num_players) if life_status[i] == 1 and i != wolf_idx]
            
            if not living_players:
                action = [0, 0]  # Fallback if no valid targets
                actions.append(action)
                continue
                
            if phase == 0:  # Night phase
                # Target players with high trust or those who defended others
                defense_scores = np.sum(defenses, axis=1)  # How much each player defends others
                target_scores = trust + defense_scores * 0.5
                target_scores[wolf_idx] = -np.inf  # Cannot target self
                target_scores[life_status == 0] = -np.inf  # Cannot target dead players
                target = np.argmax(target_scores)
                action = [0, target]
                
            elif phase == 1:  # Communication phase
                # Calculate how much each player accused wolves
                wolf_accusations = accusations[:, wolf_idx]
                
                # Prioritize accusing players who:
                # 1. Have accused the wolf
                # 2. Have high trust
                target_scores = wolf_accusations + trust * 0.5
                target_scores[wolf_idx] = -np.inf  # Cannot target self
                target_scores[life_status == 0] = -np.inf  # Cannot target dead players
                
                # Randomly choose between accusing (1) and defending (3)
                action_type = np.random.choice([1, 3], p=[0.7, 0.3])  # 70% chance to accuse
                target = np.argmax(target_scores)
                action = [action_type, target]
                
            else:  # Voting phase
                # Vote for player with most accusations, excluding self
                accusation_counts = np.sum(accusations, axis=0)
                accusation_counts[wolf_idx] = -np.inf  # Cannot vote for self
                accusation_counts[life_status == 0] = -np.inf  # Cannot vote for dead players
                target = np.argmax(accusation_counts)
                action = [0, target]
                
            actions.append(action)

        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
