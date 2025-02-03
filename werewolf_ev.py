import gymnasium
from gymnasium import spaces
import numpy as np

from pettingzoo import ParallelEnv

class werewolf():
    metadata = {'name' : 'werewolf_v1'}

    def __init__(self, num_agents = 7, comm_rounds = 4, num_wolf = 1, max_days = 15):
        self.render_mode = None
        self.num_agents = num_agents
        self.num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']

        self.max_days = max_days #maximum number of days
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 0
        #we have multiple comm_rounds per day to simulate agents being able to reply to each other

        self.possible_agents = [f'player_{i}' for i in range(self.num_agents)]
    
    def action_space(self, agent):
        return spaces.MultiDiscrete([#this action space is divided into two different discrete spaces 
                4, #this space goes from 0-3 and is used in all stages
                #in the communication stage: 0 is lie, 1 is accuse, 2 is tell truth, 3 is defend//all other phases any entry will represent a vote action or for seer the watch option
                self.num_agents  #the second is the target and is used in all stages
                ]) 
    
    def observation_space(self, agent):
        return spaces.Dict( {
            'role': spaces.MultiDiscrete([2 for i in range(self.num_agents)]), #0 is villager, 1 is werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self.num_agents, self.num_agents), dtype=np.float32),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses 
            #agent i(row) has levied at agent j(column)
            'trust': spaces.Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32),
            'life_status': spaces.MultiBinary(self.num_agents), #1 means alive, 0 is dead

            'phase' : spaces.Discrete(3), #shows current phase
            'comm_round' : spaces.Discrete(self.comm_max), #shows current comm round
            'day' : spaces.Discrete(self.max_days)
        })
    
    def reset(self):
        #initializes a new enviornment
        self.agents = self.possible_agents[:] #selects agents from possible agents
        wolves = np.random.choice(self.agent_id, size = self.num_wolf, replace = False) #randomly choose num_wolf amount of wolves from the agents(these are index numbers)
        self.wolves = wolves #stores index number of wolves//should remember to delete the entry if wolf is eliminated

        self.phase = 0 #stage 0 is werewolf killing, 1 is communication, 2 is voting
        self.comm_round = 0 #start with a communication round of 0
        self.day = 0 #goes from 0 to self.max_days - 1
        
        infos = {agent: {} for agent in self.agents} #weird thing from petting zoo//not sure why its needed or what it does but documentation shows an empty dict works
        self.state = self.get_obs_res() #self.state should be a dictionary where each key is the name of the agent(from self.agents) and the value is the observation

    def get_obs_res(self):
        observations = {}
        for agent in self.agents:
            role = np.zeros((self.num_agents,))
            if agent in self.wolves:
                role[wolves] = 1

            obs_n = {
            'role': role,
            'public_accusation': np.zeros((self.num_agents,self.num_agents), dtype=np.float32),
            'public_vote': np.zeros((self.num_agents,self.num_agents), dtype=np.float32),
            'public_defense': np.zeros((self.num_agents,self.num_agents), dtype=np.float32),
            'trust': np.full((self.num_agents,), 0.5, dtype=np.float32),
            'life_status': np.ones((self.num_agents,)),

            'phase' : np.array(0),
            'comm_round' : np.array(0),
            'day' : np.array(0)
        }
            observations.update({agent : obs_n})
        return observations
    
    # helper function to kill agents
    def update_life_status(self, target, status):
        self.state[self.agents[0]]['life_status'][target] = status


    """
    def update_matrices(self, actions):
        if self.phase == 1: #if communication phase then only update public accusations and defense
            #get old accusation and defense matrices for efficient modifications
            new_acc = self.state[self.agents[0]]['public_accusation']
            new_defense = self.state[self.agents[0]]['public_defense']
            #loop through each agent and action
            for agent, action in actions.items():
                comm_type = action[0] #recall 0 is lie, 1 is accuse, 2 is tell truth, and 3 is defend
                target = action[1] #this should be who an action targets

                agent_id = agent.split('_')[1] #doing this so as agents are terminated and removed from self.agents we keep a good track of agent_ids
                self.agents.index(agent)
                if comm_type == 1: #update accusation matrix 
                    new_acc[agent_id, target] = new_acc[agent_id, target] + 1
                elif comm_type == 3:
                    new_defense[agent_id, target] = new_acc[agent_id, target] + 1
            for agent in self.agents:
                self.state[agent]['public_accusation'] = new_acc
                self.state[agent]['public_defense'] = new_defense
    """
    
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

        # initialize rewards and termination/truncation flags
        # move the below to reset function
        observations = self.get_obs_res()
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents} 
        truncations = {agent: False for agent in self.agents}
        
        # actions based on current phase
        # NIGHT
        if self.phase == 0:
            # Night phase
            # seer sees 
            # TODO: are we allowing multiple seers???
            for agent,action in actions.items():
                agent_id = int(agent.split('_')[1])
                if self.state[agent]['role'][agent_id] == 2:  # seer role
                    target = action[1]  # the agent the seer investigates
                    self.seer_knowledge[agent] = (target, self.state[self.agents[0]]['role'][target])  # (target, werewolf status)

            # werewolves choose a target to kill
            werewolf_actions = [action[1] for agent, action in actions.items() if self.state[agent]['role'][agent] == 1]
            if werewolf_actions:
                target = np.random.choice(werewolf_actions) 
                # ^^ TODO: change this to be a choice by the agent
                # also add a voting thing if there are multiple werewolves
                self.state[self.agents[0]]['life_status'][target] = 0

            self.phase = 1 # move to day


        # DAY: communication phase
        # TODO: what else......
        elif self.phase == 1 : 
            self.accusations = {} #store accusations
            for agent,action in actions.items():
                # only alive agents communicate
                if self.state[agent]['life_status'][int(agent.split('_')[1])] == 1:  
                    target = action[1]  # agent they accuse
                    self.accusations[agent] = target
            self.comm_round += 1
            if self.comm_round >= self.comm_max:
                self.phase = 2 # move onto voting phase

        elif self.phase == 2 :
            # Voting phase
            votes = np.zeros(self.num_agents)
            for agent, action in action.items():
                if self.state[agent]['life_status'][int(agent.split('_')[1])] == 1:
                    # only LIVING agents can vote
                    target = action[1]
                    votes[target] += 1

                # eliminate agent that gets the most votes
                target = np.argmax(votes)
                self.state[self.agents[0]]['life_status'][target] = 0

                # reset phase and move to next day
                self.phase = 0
                self.comm_round = 0
                self.day += 1

        # Reward agents for surviving another day
        for agent in self.agents:
            if self.state[agent]['life_status'][int(agent.split('_')[1])] == 1:
                rewards[agent] += 2
            else:
                rewards[agent] -= 2

    # Check for end of game conditions (terminations)
    if self.current_day >= self.max_days:
        terminations = {agent: True for agent in self.agents}

        # Check for end of game conditions (terminations)
        if self.current_day >= self.max_days:
            terminations = {agent: True for agent in self.agents}

        # check for terminations
        num_werewolves = sum(self.state[agent]['role'][int(agent.split('_')[1])] for agent in self.agents)
        num_villagers = sum(1 - self.state[agent]['role'][int(agent.split('_')[1])] for agent in self.agents)

        if num_werewolves >= num_villagers:
            terminations = {agent: True for agent in self.agents}
            # Assign rewards for winning or losing
            for agent in self.agents:
                if self.state[agent]['life_status'][int(agent.split('_')[1])] == 1:  # Check if the agent is alive
                    if self.state[agent]['role'] == 'werewolf':
                        rewards[agent] = 100  # Werewolves win
                else:
                    rewards[agent] = -100  # Villagers lose
        elif num_werewolves == 0:
            terminations = {agent: True for agent in self.agents}
            # Assign rewards for winning or losing
            for agent in self.agents:
                if self.state[agent]['life_status'][int(agent.split('_')[1])] == 1:  # Check if the agent is alive
                    if self.state[agent]['role'] == 'villager':
                        rewards[agent] = 100  # Villagers win
                else:
                    rewards[agent] = -100  # Werewolves lose
    
        return observations, rewards, terminations, truncations, {}