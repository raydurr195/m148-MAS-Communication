import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from pettingzoo import ParallelEnv

class werewolf(gymnasium.Env):  # Inherit from gymnasium.Env
    metadata = {'name' : 'werewolf_v1'}

    def __init__(self, num_agents=7, comm_rounds=4, num_wolf=1, max_days=15):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render_mode = None
        self.num_agents = num_agents
        self.num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']

        self.max_days = max_days  # maximum number of days
        self.comm_max = comm_rounds  # maximum number of communication rounds per day

        self.possible_agents = [f'player_{i}' for i in range(self.num_agents)]

        # Define action space
        self.action_space = spaces.MultiDiscrete([
            4,  # 0: lie, 1: accuse, 2: tell truth, 3: defend
            self.num_agents  # target agent
        ])

        # Modify observation space to be a single Box space
        single_obs_size = (
            self.num_agents +  # role
            self.num_agents * self.num_agents +  # public_accusation
            self.num_agents * self.num_agents +  # public_vote
            self.num_agents * self.num_agents +  # public_defense
            self.num_agents +  # trust
            self.num_agents +  # life_status
            1 +  # phase
            1 +  # comm_round
            1   # day
        )
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(single_obs_size,), 
            dtype=np.float32
        )

        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        wolves = np.random.choice(self.num_agents, size=self.num_wolf, replace=False)
        self.wolves = wolves
    
        self.phase = 0
        self.comm_round = 0
        self.day = 0
    
        # Get initial state
        self.state = self.get_obs_res()
        # Return just the observation for the first agent and empty info dict
        # This assumes single-agent training where we train from player_0's perspective
        return self._dict_to_flat_obs(self.state['player_0']), {}

    def _dict_to_flat_obs(self, obs_dict):
        # Convert all observations to tensors first
        tensors = [
            torch.as_tensor(obs_dict['role'], device=self.device).flatten(),
            torch.as_tensor(obs_dict['public_accusation'], device=self.device).flatten(),
            torch.as_tensor(obs_dict['public_vote'], device=self.device).flatten(),
            torch.as_tensor(obs_dict['public_defense'], device=self.device).flatten(),
            torch.as_tensor(obs_dict['trust'], device=self.device).flatten(),
            torch.as_tensor(obs_dict['life_status'], device=self.device).flatten(),
            torch.tensor([obs_dict['phase']], device=self.device),
            torch.tensor([obs_dict['comm_round']], device=self.device),
            torch.tensor([obs_dict['day']], device=self.device)
        ]
        
        # Concatenate on GPU if available
        flat_obs = torch.cat(tensors).cpu().numpy()
        return flat_obs.astype(np.float32)

    def get_obs_res(self):
        observations = {}
        for agent in self.agents:
            role = torch.zeros((self.num_agents,), device=self.device)
            if int(agent.split('_')[1]) in self.wolves:
                role[int(agent.split('_')[1])] = 1

            obs_n = {
                'role': role,
                'public_accusation': torch.zeros((self.num_agents, self.num_agents), device=self.device),
                'public_vote': torch.zeros((self.num_agents, self.num_agents), device=self.device),
                'public_defense': torch.zeros((self.num_agents, self.num_agents), device=self.device),
                'trust': torch.full((self.num_agents,), 0.5, device=self.device),
                'life_status': torch.ones((self.num_agents,), device=self.device),
                'phase': torch.tensor([0], device=self.device),
                'comm_round': torch.tensor([0], device=self.device),
                'day': torch.tensor([0], device=self.device)
            }
            observations.update({agent: obs_n})
        return observations
        
    
    # helper function to kill agents
    def update_life_status(self, target, status):
        self.state[self.agents[0]]['life_status'][target] = status

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Pre-allocate tensors on GPU
        if not hasattr(self, 'cached_tensors'):
            self.cached_tensors = {
                'zeros': torch.zeros((self.num_agents, self.num_agents), device=self.device),
                'ones': torch.ones(self.num_agents, device=self.device)
            }
        
        """
        Update the game state based on a single action

        Parameters:
            action : action for the current agent (player_0)

        Output:
            observations : updated game state
            reward : reward for the agent
            terminated : whether the game has ended by game rules
            truncated : whether the game has ended by hitting max days
            info : additional information
        """
        # initialize rewards and termination/truncation flags
        observations = self.state
        reward = 0
        terminated = False
        truncated = False
        phase = self.phase

        # actions based on current phase
        # NIGHT
        if phase == 0:
            target = action[1]
            if target < self.num_agents:  # Ensure valid target
                self.phase = 1
                self.update_life_status(target, 0)
                for key in observations:
                    observations[key]['life_status'][target] = 0
                    observations[key]['phase'] = self.phase
                if target == 0:  # if player_0 is killed
                    reward -= 10

        # DAY: communication phase
        elif phase == 1:
            agent_id = 0  # player_0
            if self.state['player_0']['life_status'][agent_id] == 1:
                if action[0] == 1:  # accuse
                    target = action[1]
                    if target < self.num_agents:
                        for key in observations:
                            observations[key]['public_accusation'][agent_id, target] += 1
                        reward += 1
                elif action[0] == 3:  # defend
                    target = action[1]
                    if target < self.num_agents:
                        for key in observations:
                            observations[key]['public_defense'][agent_id, target] += 1
                        reward += 1

            self.comm_round += 1
            if self.comm_round >= self.comm_max:
                self.phase = 2

            for key in observations:
                observations[key]['comm_round'] = self.comm_round
                observations[key]['phase'] = self.phase

        # Voting phase
        elif phase == 2:
            self.phase = 0
            self.comm_round = 0
            self.day += 1
            target = action[1]
            if target < self.num_agents:
                self.update_life_status(target, 0)
                for key in observations:
                    observations[key]['life_status'][target] = 0
                    observations[key]['comm_round'] = self.comm_round
                    observations[key]['phase'] = self.phase
                    observations[key]['day'] = self.day
                if target == 0:  # if player_0 is voted out
                    reward -= 10

        # Reward for surviving
        if self.state['player_0']['life_status'][0] == 1:
            reward += 2

        # Check for end of game conditions
        if self.day >= self.max_days:
            terminated = True
            truncated = True

        num_werewolves = sum(1 for i in self.wolves if self.state['player_0']['life_status'][i] == 1)
        num_villagers = sum(1 for i in range(self.num_agents) if i not in self.wolves and self.state['player_0']['life_status'][i] == 1)

        if num_werewolves >= num_villagers:
            terminated = True
            reward = 100 if 0 in self.wolves else -100
        elif num_werewolves == 0:
            terminated = True
            reward = 100 if 0 not in self.wolves else -100

        return self._dict_to_flat_obs(observations['player_0']), reward, terminated, truncated, {}