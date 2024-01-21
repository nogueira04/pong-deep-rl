import numpy as np
from agent import Agent
from utils import plot_learning, make_env

if __name__ == '__main__':
    env = make_env("PongNoFrameskip-v4")
    num_games = 250
    load_checkpoint = False
    best_score = -21 # -21 é o pior score possível

    # epsilon deve ser igual epsilon_min caso carregue um modelo treinado
    agent = Agent(gamma=0.99, epsilon=0.99, alpha=0.0001, input_dims=(4, 80, 80),
                  n_actions=6, mem_size=25000, eps_min=0.02, batch_size=32,
                  replace=1000, eps_dec=1e-5)
    print("preload")
    if load_checkpoint:
        agent.load_models()
    print("aifjiaskjfasjk")
    filename = "pong.png"

    scores, eps_history = [], []
    n_steps = 0

    for i in range(num_games):
        score = 0
        observation, _ = env.reset()
        done = False

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
            else:
                env.render()
            env.render()
            observation = observation_
        
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print(f"Episode {i}, Score: {score}, Average Score: {avg_score}, Best Score: {best_score}, Epsilon: {agent.epsilon}, Steps: {n_steps}")

        if avg_score > best_score:
            agent.save_models()
            best_score = avg_score
        
        eps_history.append(agent.epsilon)
    
    x = [i+1 for i in range(num_games)]
    plot_learning(x, scores, eps_history, filename)