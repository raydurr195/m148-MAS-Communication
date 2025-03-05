import numpy as np

from werewolf_ev_updated import Werewolf


# Testing it
if __name__ == "__main__":
    # Creating the environment
    env = Werewolf(num_agents=4, comm_rounds=7, num_wolf=1, max_days=3)
    obs, infos = env.reset() # new env
    
    print("--- OBSERVATION ---")
    for agent, ob in obs.items():
        print(agent, ob)
    
    # Night Phase
    actions_night = {}
    for agent in env.agents:
        if env.agent_roles[agent] == 'werewolf':
            target = np.random.randint(0, env._num_agents)
            actions_night[agent] = [0, target]  # action type 0 on the target
        else:
            actions_night[agent] = [0, 0]  
    obs, rewards, terminations, truncations, infos = env.step(actions_night) # the results
    print("\n--- After Night ---")
    for agent, ob in obs.items():
        print(agent, ob)
    print("Rewards:", rewards)
    
    # Communication Phase
    actions_comm = {}
    for agent in env.agents:
        agent_id = int(agent.split('_')[1])
        if obs[agent]['life_status'][agent_id] == 1:
            actions_comm[agent] = [1, 0]  # 1 = accuse, targeting agent 0
        else:
            actions_comm[agent] = [0, 0]
    obs, rewards, terminations, truncations, infos = env.step(actions_comm) # the results
    print("\n--- After Comm Phase ---")
    for agent, ob in obs.items():
        print(agent, ob)
    print("Rewards:", rewards)
    
    # Voting Phase
    if env.phase == 2:
        actions_vote = {}
        for agent in env.agents:
            agent_id = int(agent.split('_')[1])
            if obs[agent]['life_status'][agent_id] == 1:
                actions_vote[agent] = [0, 0]  # Voting for agent 0 (test)
            else:
                actions_vote[agent] = [0, 0]
        obs, rewards, terminations, truncations, infos = env.step(actions_vote) # the results
        print("\n---After Voting ---")
        for agent, ob in obs.items():
            print(agent, ob)
        print("Rewards:", rewards)
