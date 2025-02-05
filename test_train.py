import unittest
from stable_baselines3 import PPO
from werewolf_ev import werewolf

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.env = werewolf()

    def test_model_instantiation(self):
        model = PPO("MultiInputPolicy", self.env, verbose=1)
        self.assertIsNotNone(model)

    def test_model_learning(self):
        model = PPO("MultiInputPolicy", self.env, verbose=1)
        try:
            model.learn(total_timesteps=1000)  # Reduced timesteps for testing
        except Exception as e:
            self.fail(f"Model learning failed with exception: {e}")

    def test_model_save_load(self):
        model = PPO("MultiInputPolicy", self.env, verbose=1)
        model.learn(total_timesteps=1000)  # Reduced timesteps for testing
        model.save("test_werewolf_model")
        loaded_model = PPO.load("test_werewolf_model")
        self.assertIsNotNone(loaded_model)
        self.assertEqual(loaded_model.policy_class.__name__, "MultiInputPolicy")

if __name__ == '__main__':
    unittest.main()