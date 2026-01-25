import time
from src.agents.agent import BaseAgent


# todo: docstrings
# todo: type hints

def train(env, agent: BaseAgent, logger, config):
    print(f"Starting training: {config['algo']} agent on environment {config['env_name']}")
    
    num_episodes = config['episodes']
    for episode in range(1, num_episodes + 1):
        # episode resets:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # pick action
            action = agent.choose_action(obs)
            
            # env step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # agent step
            agent.step(
                obs=obs, 
                action=action, 
                reward=reward, 
                next_obs=next_obs, 
                done=done
            )
            
            # updates:
            total_reward += reward
            steps += 1
            obs = next_obs
            
        # log episode metrics + parameters
        logger.log(episode, total_reward, steps, epsilon=0.1)
        
        # printout:
        if episode % 1 == 0:# todo change printout frequency
            print(f"Ep {episode} | Reward: {total_reward:.2f} | Steps: {steps}")    # todo: add to printout