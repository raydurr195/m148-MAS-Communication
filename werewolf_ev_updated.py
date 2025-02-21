import gymnasium
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

class Werewolf(): # Fixing naming convention (CamelCase)
    metadata = {'name': 'werewolf_v1'}

    def __init__(self, num_agents=7, comm_rounds=4, num_wolf=1, max_days=15):
        self.render_mode = None
        self._num_agents = num_agents
        self._num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']

        self.max_days = max_days    # Maximum number of days
        self.comm_max = comm_rounds # Maximum communication rounds per day
        # There is a maximum number of communication rounds equal to comm_rounds for phase 0
        # We have multiple comm_rounds per day to simulate agents being able to reply to each other

        self.possible_agents = [f'player_{i}' for i in range(self._num_agents)]
    
    def action_space(self, agent):
        # Two discrete spaces: 
        return spaces.MultiDiscrete([4, # Goes from 0-3 and is used all stages
                                     # 0 is a lie, 1 is accuse, 2 is telling the truth, 3 is defneding 
                                     self._num_agents # the target and is used on all stages
                                     ])
    
    def observation_space(self, agent):
        return spaces.Dict({
            'role': spaces.MultiDiscrete([2 for _ in range(self._num_agents)]),  # 0 = villager, 1 = werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self._num_agents, self._num_agents), dtype=np.float32),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self._num_agents, self._num_agents), dtype=np.float32),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self._num_agents, self._num_agents), dtype=np.float32),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses
            #agent i(row) has levied at agent j(column)
            'trust': spaces.Box(low=0, high=1, shape=(self._num_agents,), dtype=np.float32),
            'life_status': spaces.MultiBinary(self._num_agents),  # 1 = alive, 0 = dead
            'phase': spaces.Discrete(3), # 0: night, 1: communication, 2: voting (the phase)
            'comm_round': spaces.Discrete(self.comm_max + 1), # shows the current round
            'day': spaces.Discrete(self.max_days + 1)
        })
    
    def reset(self, seed=None, options=None):
        # Initialize a new environment
        self.agents = self.possible_agents[:]  # selects agents from possible agents
        self.wolves = np.random.choice(range(self._num_agents), size=self._num_wolf, replace=False)
        # randomly choose num_wolf amount of wolves from agents (these are index numbers)
        # stores index number of wolves, remember to delete the entry if wolf is eliminated

        # Just naming the agents
        self.agent_roles = {}
        for agent in self.agents:
            agent_id = int(agent.split('_')[1])
            if agent_id in self.wolves:
                self.agent_roles[agent] = 'werewolf'
            else:
                self.agent_roles[agent] = 'villager'

        self.phase = 0  # stage 0 is werewolf killing, 1 is communication, 2 is voting
        self.comm_round = 0 # start with a communication round of 0
        self.day = 0 # goes from 0 to self.max_days - 1
        
        self.state = self.get_obs_res()

        infos = {agent: {} for agent in self.agents} # weird thing from petting zoo//not sure why its needed or what it does but documentation shows an empty dict works
        return self.state, infos # added a return
    
    def get_obs_res(self):
        observations = {}
        for agent in self.agents:
            # agent_id = int(agent.split('_')[1])
            # Werewolves know the identities of all wolves; villagers see all zeros.
            if self.agent_roles[agent] == 'werewolf':
                role_obs = np.zeros((self._num_agents,), dtype=int)
                role_obs[self.wolves] = 1
            else:
                role_obs = np.zeros((self._num_agents,), dtype=int)
            obs_n = {
                'role': role_obs,
                'public_accusation': np.zeros((self._num_agents, self._num_agents), dtype=np.float32),
                'public_vote': np.zeros((self._num_agents, self._num_agents), dtype=np.float32),
                'public_defense': np.zeros((self._num_agents, self._num_agents), dtype=np.float32),
                'trust': np.full((self._num_agents,), 0.5, dtype=np.float32),
                'life_status': np.ones((self._num_agents,), dtype=int),  
                'phase': np.array(self.phase),
                'comm_round': np.array(self.comm_round),
                'day': np.array(self.day)
            }
            observations[agent] = obs_n
        return observations
    

    # helper function to kill agents
    def update_life_status(self, target, status):
        for agent in self.agents:
            self.state[agent]['life_status'][target] = status

    def step(self, actions):
        """
        Update the game state based on actions

        Parameters:
            actions (dict) : dict mapping each agent to their chosen action

        Output:
            observations (dict) : updated game state for each agent
            rewards (dict) : rewards for each agent
            terminations (dict) : whether the game has ended by game rules
            truncations (dict) : whether the game has ended by hitting max days

        TODO: add verbose option??? to print state of game after each day
        TODO: filter so that only living agents can participate
        TODO: use the update_life_status helper function
        """
        observations = self.state
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        phase = self.phase

        # actions based on current phase
        # NIGHT
        if phase == 0:
            # Night phase
            # seer sees 
            # TODO: are we allowing multiple seers???

            #for agent,action in actions.items():
            #    agent_id = int(agent.split('_')[1])
            #    if self.state[agent]['role'][agent_id] == 2:  # seer role
            #        target = action[1]  # the agent the seer investigates
            #        observations[agent]['role'][target]= 1 if self.state[agent]['role'][target] == 1 else 0


            # werewolves choose a target to kill
            werewolf_targets = []
            for agent, action in actions.items():
                agent_id = int(agent.split('_')[1])
                if self.agent_roles[agent] == 'werewolf' and self.state[agent]['life_status'][agent_id] == 1:
                    werewolf_targets.append(action[1])
            if werewolf_targets:
                target = np.random.choice(werewolf_targets)
                self.update_life_status(target, 0)

            # ^^ TODO: change this to be a choice by the agent
                # also add a voting thing if there are multiple werewolves

            #should loop through and update life status for every agent
            self.phase = 1
            for agent in self.agents:
                self.state[agent]['phase'] = self.phase

             #return observations, rewards, terminations, truncations, {}
            # move to day//I think we should return here, update each agents observations to have a phase equal to 1 and allow them to take more actions


        # DAY: communication phase
        # TODO: what else......
        elif phase == 1:
            accusations = np.zeros((self._num_agents, self._num_agents), dtype=np.float32)
            defenses = np.zeros((self._num_agents, self._num_agents), dtype=np.float32)
            for agent, action in actions.items():
                agent_id = int(agent.split('_')[1])
                # Only alive agents can communicate
                if self.state[agent]['life_status'][agent_id] == 1:
                    # maybe punish for voting for a dead person/voting when you are dead?
                    act_type = action[0]  # if the agent decides to accuse another agent
                    target = action[1] # agent they accused
                    # Action type 1: accuse; Action type 3: defend.
                    if act_type == 1:
                        accusations[agent_id, target] += 1
                    elif act_type == 3:  #if the agent decides to defend another agent
                        defenses[agent_id, target] += 1
            self.comm_round += 1
            if self.comm_round >= self.comm_max:
                self.phase = 2  # Move to voting phase
            for agent in self.agents:
                self.state[agent]['public_accusation'] += accusations
                self.state[agent]['public_defense'] += defenses
                self.state[agent]['comm_round'] = self.comm_round
                self.state[agent]['phase'] = self.phase

        # return observations, rewards, terminations, truncations, {}

        # Voting
        elif phase == 2:
            votes = np.zeros(self._num_agents, dtype=int)
            for agent, action in actions.items():
                agent_id = int(agent.split('_')[1])
                # only LIVING agents can vote
                # punish for voting for a dead player/a dead player voting?
                if self.state[agent]['life_status'][agent_id] == 1:
                    target = action[1]
                    votes[target] += 1
            if np.sum(votes) > 0:
                vote_target = np.argmax(votes)
                self.update_life_status(vote_target, 0)
            # Reset phase and comm_round, and move to the next day
            self.phase = 0
            self.comm_round = 0
            self.day += 1
            for agent in self.agents:
                self.state[agent]['phase'] = self.phase
                self.state[agent]['comm_round'] = self.comm_round
                self.state[agent]['day'] = self.day

        # Rewards
        # Alive agents earn +2; dead agents lose 2.
        for agent in self.agents:
            agent_id = int(agent.split('_')[1])
            if self.state[agent]['life_status'][agent_id] == 1:
                rewards[agent] += 2
            else:
                rewards[agent] -= 2

        # Termination
        if self.day >= self.max_days:
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
        
        # Checking win condition
        living_werewolves = 0
        living_villagers = 0
        for agent in self.agents:
            agent_id = int(agent.split('_')[1])
            if self.state[agent]['life_status'][agent_id] == 1:
                if self.agent_roles[agent] == 'werewolf':
                    living_werewolves += 1
                else:
                    living_villagers += 1
        if living_werewolves >= living_villagers:
            terminations = {agent: True for agent in self.agents}
            for agent in self.agents:
                agent_id = int(agent.split('_')[1])
                if self.state[agent]['life_status'][agent_id] == 1:
                    if self.agent_roles[agent] == 'werewolf':
                        rewards[agent] = 100   # Werewolves win
                    else:
                        rewards[agent] = -100  # Villagers lose
        elif living_werewolves == 0:
            terminations = {agent: True for agent in self.agents}
            for agent in self.agents:
                agent_id = int(agent.split('_')[1])
                if self.state[agent]['life_status'][agent_id] == 1:
                    if self.agent_roles[agent] == 'villager':
                        rewards[agent] = 100   # Villagers win
                    else:
                        rewards[agent] = -100  # Werewolves lose

        return self.state, rewards, terminations, truncations, infos



