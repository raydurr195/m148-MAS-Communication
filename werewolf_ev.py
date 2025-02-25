import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pettingzoo import ParallelEnv

class werewolf(ParallelEnv):
    metadata = {'name' : 'werewolf_v1'}

    def __init__(self, num_players = 7, comm_rounds = 4, num_wolf = 1, max_days = 15):
        super().__init__()
        self.render_mode = None
        self.num_players = num_players
        self.num_wolf = num_wolf
        self.roles = ['werewolf', 'villager']

        self.max_days = max_days #maximum number of days
        self.comm_max = comm_rounds #there is a maximum number of communication rounds equal to comm_rounds for phase 0
        #we have multiple comm_rounds per day to simulate agents being able to reply to each other
        self.act_space = spaces.MultiDiscrete([
                3, 
                self.num_players  
        ])
        #this action space is divided into two different discrete spaces 
        #this space goes from 0-2 and is used in all stages
        #in the communication stage: 0 is defend, 1 is accuse, 2 is to do nothing
        # all other phases any entry will represent a vote action or for seer the watch option
        #the second is the target and is used in all stages
        self.obs_space = spaces.Dict({
            'role': spaces.MultiDiscrete([3 for i in range(self.num_players)]), #0 is villager, 1 is werewolf
            'public_accusation': spaces.Box(low=0, high=np.inf, shape=(self.num_players, self.num_players), dtype=np.float64),
            'public_vote': spaces.Box(low=0, high=np.inf, shape=(self.num_players, self.num_players), dtype=np.float64),
            'public_defense': spaces.Box(low=0, high=np.inf, shape=(self.num_players, self.num_players), dtype=np.float64),
            #public_x is an nxn matrix representing the number of accusations/votes/defenses 
            #agent i(row) has levied at agent j(column)
            'trust': spaces.Box(low=0, high=1, shape=(self.num_players,), dtype=np.float64),
            'life_status': spaces.MultiBinary(self.num_players), #1 means alive, 0 is dead

            'phase' : spaces.Box(low = 0, high = 2, shape=(1,1)), #shows current phase
            'comm_round' : spaces.Box(low = 0, high = self.comm_max-1, shape=(1,1)), #shows current comm round
            'day' : spaces.Box(low = 0, high = self.max_days-1, shape=(1,1))
        })

        self.observation_spaces = self.obs_space
        self.action_spaces = self.act_space
        self.possible_agents = [f'player_{i}' for i in range(self.num_players)]
    
    def action_space(self, agent):
        return self.act_space
                
    
                
    def observation_space(self, agent):
        return spaces.flatten_space(self.obs_space)
    
    def reset(self, *, seed = None, options = None):
        #initializes a new enviornment
        self.agents = self.possible_agents[:] #selects agents from possible agents

        # assign wolves
        wolves = np.random.choice(len(self.agents), size = self.num_wolf, replace = False) #randomly choose num_wolf amount of wolves from the agents(these are index numbers)
        self.wolves = wolves #stores index number of wolves
        #//should remember to delete the entry if wolf is eliminated

        # assign villagers and seer
        villager_indices = [i for i in range(self.num_players) if i not in self.wolves]
        self.seer = np.random.choice(villager_indices)
        self.villagers = villager_indices
        self.phase = 0 #stage 0 is werewolf killing, 1 is communication, 2 is voting
        self.comm_round = 0 #start with a communication round of 0
        self.day = 0 #goes from 0 to self.max_days - 1
        self.state = self.get_obs_res() #self.state should be a dictionary where each key is the name of the agent(from self.agents) and the value is the observation
        
        obs = self.state
        info = {agent: {'role' : spaces.unflatten(self.obs_space,self.state[agent])} for agent in self.agents} #log the role of each agent within their infos
        self.infos = info

        #store accussation and defense matrices
        self.acc = np.zeros((self.num_players,self.num_players), dtype=np.float64)
        self.defense = np.zeros((self.num_players,self.num_players), dtype=np.float64)
        #custom metrics to be used within callback feature on episode end
        self.v_acc_v = 0 #number of times villagers accuse other villagers
        self.v_acc_w = 0 #number of times accuse wolves
        self.v_def_v = 0 #number of times villagers defend other villagers
        self.v_def_w = 0 #number of times villagers defend wolves
        self.suicide = 0 #number of times that agents vote to kill themselves
        self.win = 2 #who wins: 0 represents villagers, 1 represents wolves, and 2 represents a draw
        self.vill_reward = 0 #the total rewards for the villager team
        self.wolf_reward = 0 #the total rewards for the wolf team
        return obs, info

    def get_obs_res(self):
        observations = {}
        for agent in self.agents:
            agent_id = int(agent.split('_')[1])
            role = np.zeros((self.num_players,), dtype = np.int32)
            if agent_id in self.wolves:
                role[agent_id] = 1  # werewolves
            elif agent_id == self.seer:
                role[agent_id] = 2  # seer
    

            obs_n = {
            'role': role,
            'public_accusation': np.zeros((self.num_players,self.num_players), dtype=np.float64),
            'public_vote': np.zeros((self.num_players,self.num_players), dtype=np.float64),
            'public_defense': np.zeros((self.num_players,self.num_players), dtype=np.float64),
            'trust': np.full((self.num_players,), 0.5, dtype=np.float64),
            'life_status': np.ones((self.num_players,)),

            'phase' : np.array(0).reshape((1,1)),
            'comm_round' : np.array(0).reshape((1,1)),
            'day' : np.array(0).reshape((1,1))
        }
            observations.update({agent : spaces.flatten(self.obs_space, obs_n)})
        return observations
    

    
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
        observations = self.state
        observations = {agent : spaces.unflatten(self.obs_space, obs) for agent, obs in observations.items()}
        #actions = {agent : spaces.unflatten(self.act_space, action) for agent,action in actions.items()}
        rewards = {agent: 0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents} 
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        phase = self.phase
        # actions based on current phase
        # NIGHT
        if phase == 0:
            # Night phase
            # seer sees and wolves choose who to kill
            werewolf_actions = list()
            for agent,action in actions.items():
                agent_id = int(agent.split('_')[1])
                target = action[1]
                #seer logic
                if observations[agent]['role'][agent_id] == 2:  # seer role
                    observations[agent]['role'][target]= 1 if observations[self.agents[target]]['role'][target] == 1 else 0 
                # werewolf logic
                elif observations[agent]['role'][agent_id] == 1:
                    werewolf_actions.append(target)
            #suicide tracker
            wolf_vote_wolf = np.bincount(werewolf_actions, minlength = self.num_players)[self.wolves] #displays number of times a werewolf voted for a wolf
            self.suicide += np.sum(wolf_vote_wolf)
            #choose target
            target = np.random.choice(werewolf_actions) 
            self.phase = 1
            #update life status of target(kill target)
            for agent in self.agents:
                observations[agent]['life_status'][target] = 0
                observations[agent]['phase'] = np.array(self.phase, dtype = np.int32).reshape((1,1))

        # DAY: communication phase
        elif phase == 1 : 
            accusations = np.zeros((self.num_players,self.num_players), dtype=np.float64) #store new accusations
            defenses = np.zeros((self.num_players,self.num_players), dtype=np.float64) #stores new defenses
            for agent,action in actions.items():
                agent_id = int(agent.split('_')[1])
                # only alive agents communicate
                if observations[agent]['life_status'][agent_id] == 1:  
                    #maybe punish for voting for a dead person/voting when you are dead?
                    if action[0] == 1:#if the agent decides to accuse another agent
                        target = action[1]  # agent they accuse
                        accusations[agent_id,target] +=1
                        
                    elif action[0] == 0: #if the agent decides to defend another agent
                        target = action[1]
                        defenses[agent_id,target] += 1
            self.acc += accusations
            self.defense += defenses
            self.comm_round += 1
            if self.comm_round >= (self.comm_max-1): # move onto voting phase
                #self.comm_round = 0
                self.phase = 2
            for agent in self.agents:
                #update all agents accusations and defenses matrix
                observations[agent]['public_accusation'] = self.acc
                observations[agent]['public_defense'] = self.defense
                observations[agent]['comm_round'] = np.array(self.comm_round,dtype = np.int32).reshape((1,1))
                observations[agent]['phase'] = np.array(self.phase,dtype = np.int32).reshape((1,1))
        
        # DAY: Voting Phase
        elif phase == 2 :
            # first, reset phase and move to next day
            self.phase = 0
            self.comm_round = 0
            self.day += 1
            # Voting phase
            votes = np.zeros((1,self.num_players))
            for agent, action in actions.items():
                if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:
                    # only LIVING agents can vote
                    target = action[1]
                    votes[0,target] += 1
                    if int(agent.split('_')[1]) == target: #if an agent votes for themselves update the suicide attribute
                        self.suicide += 1

            # eliminate agent that gets the most votes
            max_votes = np.max(votes)
            targets = np.where(votes== max_votes)[1]
            target = np.random.choice(targets)
            # update life status of target in all agent's observations
            for agent in self.agents:
                observations[agent]['life_status'][target] = 0
                observations[agent]['comm_round'][0] = np.array(self.comm_round,dtype=np.int32).reshape((1,1))
                observations[agent]['phase'][0] = np.array(self.phase,dtype=np.int32).reshape((1,1))
                observations[agent]['day'][0] = np.array(self.day,dtype=np.int32).reshape((1,1))


        # Reward agents for surviving another day
        for agent in self.agents:
            if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:
                rewards[agent] += 2
            else:
                rewards[agent] -= 2
            if observations[agent]['role'][int(agent.split('_')[1])] == 1: #if agent wolf
                self.wolf_reward += rewards[agent]
            else:
                self.vill_reward += rewards[agent]

        # check for terminations
        num_villagers = 0
        num_werewolves = 0
        for agent in self.agents:
            agent_id = int(agent.split('_')[1])
            if observations[agent]['life_status'][agent_id] == 1:
                num_werewolves += observations[agent]['role'][agent_id]
                num_villagers += 1 - observations[agent]['role'][agent_id]

        if num_werewolves >= num_villagers: #werewolves win condition
            self.win = 1 #werewolves win
            terminations = {agent: True for agent in self.agents}
            # Assign rewards for winning or losing
            for agent in self.agents:
                if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:  # Check if the agent is alive
                    if observations[agent]['role'][int(agent.split('_')[1])] == 1:
                        rewards[agent] = 100  # Werewolves win
                        self.wolf_reward += rewards[agent]
                else:
                    rewards[agent] = -100  # Villagers lose
                    self.vill_reward += rewards[agent]
            self.get_final_metric()


        elif num_werewolves == 0: #villager win conditions
            terminations = {agent: True for agent in self.agents}
            self.win = 0 #villagers win
            # Assign rewards for winning or losing
            for agent in self.agents:
                if observations[agent]['life_status'][int(agent.split('_')[1])] == 1:  # Check if the agent is alive
                    if observations[agent]['role'][int(agent.split('_')[1])] == 0:
                        rewards[agent] = 100  # Villagers win
                        self.vill_reward += rewards[agent]
                else:
                    rewards[agent] = -100  # Werewolves lose
                    self.wolf_reward += rewards[agent]
            self.get_final_metric()


        # check for truncations
        if self.day >= (self.max_days-1):
            truncations = {agent: True for agent in self.agents}
            self.get_final_metric()


        
        # update observations (???)
        #note that we should be updating observations after each phase
        observations = {agent : spaces.flatten(self.obs_space, obs) for agent, obs in observations.items()}
        self.state = observations
        return observations, rewards, terminations, truncations, infos
    def get_final_metric(self):
            acc = self.acc
            defense = self.defense
            self.v_acc_v = np.sum(
                acc[np.ix_(self.villagers,self.villagers)]  #this filters for only those entries where the row is within
                                                            #the villager index and the column is within the villager index
                                                            #ie how many times a villager accused a villager
            )
            self.v_acc_w = np.sum(
                acc[np.ix_(self.villagers,self.wolves)]  
            )
            self.v_def_w = np.sum(
                defense[np.ix_(self.villagers, self.wolves)] #this filters for only those entries where the row is within
                                                             #the villager index and the column is within the wolf index
                                                             #ie how many times a villager defended a wolf
            )
            self.v_def_v = np.sum(
                defense[np.ix_(self.villagers, self.villagers)] 
            )