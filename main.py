from src.utils import set_random_seed, Logger, get_device
from src.agents.agent import BaseAgent
from src.training.trainer import train
from src.template import SimpleGridEnv, pre_process

def main():
    # configuration
    config = {
        "algo": "TestRun",
        "env_name": "SimpleGrid",
        "seed": 42,
        "episodes": 3,
        "obs_shape": (3, 8, 8), 
    }

    # setup infra:
    set_random_seed(seed=config["seed"])
    device = get_device()
    logger = Logger(config=config)
    print(f"Running on: {device}")

    # init environment (from template)
    env = SimpleGridEnv(preprocess=pre_process) 
    
    # init agent
    agent = BaseAgent(
        config=config, 
        obs_shape=config["obs_shape"], 
        num_actions=env.action_space.n, # todo: is this available?
        device=device
    )

    # run training
    train(env=env, agent=agent, logger=logger, config=config)

if __name__ == "__main__":
    main()