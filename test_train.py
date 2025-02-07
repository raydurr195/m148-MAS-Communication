import unittest
import torch
import time
from stable_baselines3 import PPO
from werewolf_ev import werewolf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def make_env(rank, seed=0):
    def _init():
        env = werewolf()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Increased parallel environments for GPU
        if self.device.type == "cuda":
            self.num_envs = 16  # Increased from 8 to 16 for better GPU utilization
        else:
            self.num_envs = 4
            
        env_fns = [make_env(i) for i in range(self.num_envs)]
        self.env = SubprocVecEnv(env_fns) if self.device.type == "cuda" else DummyVecEnv(env_fns)

    def test_model_instantiation(self):
        model = PPO("MlpPolicy", self.env, verbose=1, device=self.device,
                   batch_size=256 if self.device.type == "cuda" else 64,
                   n_steps=2048,
                   n_epochs=10,
                   learning_rate=3e-4)
        self.assertIsNotNone(model)
    
    

    def test_model_learning(self):
        model = PPO("MlpPolicy", self.env, verbose=1, device=self.device)
        try:
            model.learn(total_timesteps=1000)  # Reduced timesteps for testing
        except Exception as e:
            self.fail(f"Model learning failed with exception: {e}")

    def test_model_save_load(self):
        model = PPO("MlpPolicy", self.env, verbose=1, device=self.device)
        model.learn(total_timesteps=1000)  # Reduced timesteps for testing
        model.save("test_werewolf_model")
        loaded_model = PPO.load("test_werewolf_model", device=self.device)
        self.assertIsNotNone(loaded_model)
        self.assertEqual(loaded_model.policy_class.__name__, "ActorCriticPolicy")

    def test_performance(self):
        model = PPO("MlpPolicy", 
                    self.env, 
                    verbose=1, 
                    device=self.device,
                    batch_size=2048 if self.device.type == "cuda" else 128,  # Increased for more parallel envs
                    n_steps=8192,  # Increased to match parallel envs
                    n_epochs=5,
                    learning_rate=3e-4,
                    policy_kwargs=dict(
                        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Wider network for more data
                        optimizer_class=torch.optim.AdamW  # Better optimizer for GPU
                    ))

if __name__ == '__main__':
    unittest.main()