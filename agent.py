from replay_buffer import ReplayBuffer
from network import build_dqn
from keras.models import load_model
import numpy as np

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, replace,
                 input_dims, eps_dec=1e-5, eps_min=0.01, mem_size=1000000,
                 q_eval_fname="q_eval.keras", q_target_fname="q_target.keras"):
        
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname
        self.learn_step = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 512)
        self.q_next = build_dqn(alpha, n_actions, input_dims, 512)

    def replace_target_network(self):
        if self.replace != 0 and self.learn_step % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action
    
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()

            q_eval = self.q_eval.predict(state)
            q_next = self.q_next.predict(new_state)

            q_next[done] = 0.0

            indices = np.arange(self.batch_size)
            q_target = np.copy(q_eval)

            q_target[indices, action] = reward + self.gamma * np.max(q_next, axis=1)

            self.q_eval.train_on_batch(state, q_target)
            
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

            self.learn_step += 1

    def save_models(self):
        print(">>> Saving models <<<")
        self.q_eval.save(self.q_eval_model_file)
        self.q_next.save(self.q_target_model_file)
        print(">>> Models saved <<<")

    def load_models(self):
        print(">>> Loading models <<<")
        self.q_eval = load_model(self.q_eval_model_file)
        self.q_next = load_model(self.q_target_model_file)
        print(">>> Models loaded <<<")