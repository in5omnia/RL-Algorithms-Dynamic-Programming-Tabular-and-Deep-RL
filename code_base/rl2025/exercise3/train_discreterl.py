import copy
import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict

from rl2025.constants import EX3_DISCRETERL_CARTPOLE_CONSTANTS as CARTPOLE_CONSTANTS
from rl2025.constants import EX3_DQN_MOUNTAINCAR_CONSTANTS as MOUNTAINCAR_CONSTANTS
from rl2025.exercise3.agents import DiscreteRL
from rl2025.util.hparam_sweeping import generate_hparam_configs
from rl2025.util.result_processing import Run

RENDER = False # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGHTS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "MOUNTAINCAR" # "CARTPOLE" is also possible if you uncomment the corresponding code, but is not assessed here.

### ASSIGNMENT: CHANGE epsilon_decay_strategy: "constant" TO "linear" OR "exponential" TO ANSWER QUESTIONS 3.2 TO 3.6 IN answer_sheet.py ###
MOUNTAINCAR_CONFIG = {
    "eval_freq": 10000, # HOW OFTEN WE EVALUATE (AND RENDER IF RENDER=TRUE)
    "eval_episodes": 100, # DECREASING THIS MIGHT REDUCE EVALUATION ACCURACY; BUT MAKES IT EASIER TO SEE HOW THE POLICY EVOLVES OVER TIME (BY ENABLING RENDER ABOVE)
    "learning_rate": 3e-4,
    "hidden_size": (64,64),
    "target_update_freq": 2000,
    "batch_size": 64,   # not used here
    "epsilon_decay_strategy": "constant", # "constant" or "linear" or "exponential"
    "epsilon_start": 0.5,
    "epsilon_min": 0.05, # only used in linear and exponential decay strategies
    "epsilon_decay": None, # For exponential epsilon decay
    "exploration_fraction": None, # For linear epsilon decay, fraction of training time at which epsilon=epsilon_min
    "buffer_capacity": int(1e6),
    "plot_loss": False, # SET TRUE FOR 3.3 (Understanding the Loss)
}

MOUNTAINCAR_CONFIG.update(MOUNTAINCAR_CONSTANTS)

MOUNTAINCAR_HPARAMS_LINEAR_DECAY = {
    "epsilon_start": [1.0,],
    "exploration_fraction": [0.99, 0.75, 0.01]
    }

MOUNTAINCAR_HPARAMS_EXP_DECAY = {
    "epsilon_start": [1.0, ],
    "epsilon_decay": [1.0, 0.5, 1e-5]
    }

if MOUNTAINCAR_CONFIG['epsilon_decay_strategy'] == "linear":
    MOUNTAINCAR_HPARAMS = MOUNTAINCAR_HPARAMS_LINEAR_DECAY
elif MOUNTAINCAR_CONFIG['epsilon_decay_strategy'] == "exponential":
    MOUNTAINCAR_HPARAMS = MOUNTAINCAR_HPARAMS_EXP_DECAY
else:
    MOUNTAINCAR_HPARAMS = None

SWEEP_RESULTS_FILE_MOUNTAINCAR = f"DiscreteRL-MountainCar-sweep-decay-{MOUNTAINCAR_CONFIG['epsilon_decay_strategy']}-results.pkl"



CARTPOLE_CONFIG = {
    "eval_freq": 10000,
    "eval_episodes": 100,
    "hidden_size": (64,),
    "learning_rate": 1e-3,
}

CARTPOLE_CONFIG.update(CARTPOLE_CONSTANTS)

CARTPOLE_HPARAMS = {
    "learning_rate": [2e-2, 2e-3, 2e-4],
    }

SWEEP_RESULTS_FILE_CARTPOLE = "DiscreteRL-CartPole-sweep-results.pkl"


def play_episode(
    env: gym.Env,
    agent: DiscreteRL,
    train: bool = True,
    explore=True,
    render=False,
    max_steps=200,
) -> Tuple[int, float, Dict]:
    """
    Play one episode and train discrete RL algorithm

    :param env (gym.Env): gym environment
    :param agent (DiscreteRL): DiscreteRL agent
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether exploration is used
    :param render (bool): flag whether environment should be visualised
    :param max_steps (int): max number of timesteps for the episode
    :return (Tuple[int, float]): total number of executed steps and received reward
    """

    ep_data = defaultdict(list)
    obs, _ = env.reset()
    """
    The observation space for position and velocity is (-1.2,0.6)x(-0.07,0.07).
    You can discretise the space in 8x8 cells.
    """

    if render:
        env = gym.make(CONFIG["env"], render_mode="human")

    done = False
    num_steps = 0
    episode_return = 0

    while not done and num_steps < max_steps:
        action = agent.act(np.array(obs), explore=explore)
        nobs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if train:
            # Update agent after each step with (s, a, r, s', done) tuple
            q_value = agent.update(obs, action, rew, nobs, done)
            ep_data['q_values'].append(q_value)

        num_steps += 1
        episode_return += rew

        obs = nobs

    return num_steps, episode_return, ep_data


def train(env: gym.Env, config, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of DISCRETE_RL on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[np.ndarray, np.ndarray, np.ndarray]): average eval returns during training, evaluation
            timesteps and compute times at evaluation
    """
    timesteps_elapsed = 0

    agent = DiscreteRL(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )

    total_steps = config["max_timesteps"]
    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()
    with tqdm(total=total_steps) as pbar:
        while timesteps_elapsed < total_steps:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, total_steps)
            num_steps, ep_return, ep_data = play_episode(
                env,
                agent,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
            )
            timesteps_elapsed += num_steps
            pbar.update(num_steps)
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            if timesteps_elapsed % config["eval_freq"] < num_steps:
                eval_return = 0
                if config["env"] == "CartPole-v1" or config["env"] == "MountainCar-v0":
                    max_steps = config["episode_length"]
                else:
                    raise ValueError(f"Unknown environment {config['env']}")

                for _ in range(config["eval_episodes"]):
                    _, total_reward, _ = play_episode(
                        env,
                        agent,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=max_steps,
                    )
                    eval_return += total_reward / (config["eval_episodes"])
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                    )
                eval_returns_all.append(eval_return)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

    if config["save_filename"]:
        print("\nSaving to: ", agent.save(config["save_filename"]))

    # you may add logging of additional metrics here
    run_data["train_episodes"] = np.arange(1, len(run_data["train_ep_returns"]) + 1).tolist()

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":
    print(ENV)
    if ENV == "MOUNTAINCAR":
        CONFIG = MOUNTAINCAR_CONFIG
        HPARAMS_SWEEP = MOUNTAINCAR_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_MOUNTAINCAR
    elif ENV == "CARTPOLE":
        CONFIG = CARTPOLE_CONFIG
        HPARAMS_SWEEP = CARTPOLE_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_CARTPOLE
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])

    if SWEEP:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run...")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i+1}/{NUM_SEEDS_SWEEP}")
                run_save_filename = '--'.join([run.config["algo"], run.config["env"], hparams_values, str(i)])
                if SWEEP_SAVE_ALL_WEIGHTS:
                    run.set_save_filename(run_save_filename)
                eval_returns, eval_timesteps, times, run_data = train(env, run.config, output=False)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")

        if SWEEP_SAVE_RESULTS:
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)
    else:
        _ = train(env, CONFIG)
    env.close()
