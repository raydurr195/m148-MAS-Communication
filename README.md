# Werewolf Multi-Agent RL Project

**Werewolf** is a social deduction game where there are villagers and werewolf. The goal is for werewolf to eliminate all the villagers. While the villagers try to find the werewolf and vote them out. The goal of our project is to create a Multi-Agent Reinforcement Learning model to simulate **Werewolf**, a popular social deduction game and identify AI decision-making.

## Description

**AI Agents:** Used to simulate players action.
**Roles:** There are two different roles: Villagers and Werewolf. (More can be added such as a seer)
**Reinforcement Learning:** Uses [**Ray**]([url](https://www.ray.io/)) library to implement PPO training algorithm onto the agents
**Customizable Rules**: Modify rounds, amount of villagers / werewolf to simulate different scenarios

## Rules
Agents are randomly assigned roles. Every night a werewolf chooses a villager to be eliminated. After that, there is a disccussion phase, where everyone can choose to lie or tell the truth. Everyone then votes on a player to kick out. This is repeated until all werewolf are eliminated (villagers win) or werewolves outnumber the villagers (werewolf wins).

## Getting Started

### Dependencies

* [Python]([url](https://www.python.org/downloads/)) 3.13.2 (This is the version we used - other version may be usable but not guarenteed)

## Settings
There is a static and non-static training model. The static training model is used to compare results with the model trained using PPO algorithm. The model is set to static on default. This can be changed by commenting out the following code:... (WIP) 

The amount of agents, rounds, werewolf, days can be changed in the werewolf_ev.py file. (maybe add an easy way to change the settings at the very top) ------ (WIP) 

### Installing

* (Insert a txt file for all libraries used)
* Clone the repository
```
git clone ...
```
* Navigate folder
```
cd werewolf...
```
* Install dependencies
```
pip install -r requirements.txt
```
* Run 
```
python ...py
```

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Authors

Arnav Chittibabu, Kenneth Chow, Raymond Durr, Leigh Anne Lemoine, Wenhan Lu, Sia Phulambrikar, Jason Vo


## Acknowledgments

Thank prof and TA

Inspiration, code snippets, etc.
