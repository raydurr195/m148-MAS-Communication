import torch
from stable_baselines3 import PPO
from werewolf_ev import werewolf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import multiprocessing as mp

def make_env(rank, seed=0):
    def _init():
        env = werewolf()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create vectorized environment
    num_envs = 8 if torch.cuda.is_available() else 4
    env_fns = [make_env(i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns) if device.type == "cuda" else DummyVecEnv(env_fns)

    # Initialize model with optimized parameters
    model = PPO("MlpPolicy", 
               env, 
               verbose=1,
               device=device,
               n_steps=2048,
               batch_size=256,
               n_epochs=10,
               learning_rate=3e-4,
               ent_coef=0.01,
               vf_coef=0.5,
               clip_range=0.2,
               policy_kwargs=dict(
                   net_arch=dict(
                       pi=[256, 256],
                       vf=[256, 256]
                   ),
                   optimizer_class=torch.optim.AdamW
               ))

    # Create callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./checkpoints/',
        name_prefix='werewolf_model'
    )

    # Train the model
    total_timesteps = 2_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False  # Disable progress bar if dependencies are missing
    )

    # Save the final model
    model.save("werewolf_model_final")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    # Required for proper multiprocessing
    mp.freeze_support()
    main()


