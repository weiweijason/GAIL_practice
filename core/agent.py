import multiprocessing
import numpy as np
import torch
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'mask', 'log_prob'])


class Agent:
    def __init__(self, env, policy_net, device, custom_reward=None,
                 running_state=None, num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads
        
    def collect_samples(self, min_batch_size, render=False, mean_action=False):
        t_start = torch.utils.data.get_worker_info() is not None
        thread_batch_size = int(min_batch_size / self.num_threads)
        
        queue = multiprocessing.Queue()
        workers = []
        
        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy_net, self.custom_reward,
                          self.running_state, thread_batch_size, mean_action)
            workers.append(multiprocessing.Process(target=self._collect_samples, args=worker_args))
        for worker in workers:
            worker.start()
            
        memory, log = self._collect_samples(0, None, self.env, self.policy_net, self.custom_reward,
                                          self.running_state, thread_batch_size, mean_action, render)
        
        worker_logs = []
        for _ in range(self.num_threads - 1):
            worker_memory, worker_log = queue.get()
            memory = memory + worker_memory
            worker_logs.append(worker_log)
            
        for worker in workers:
            worker.join()
            
        if self.num_threads > 1:
            log_avg = {k: sum(l[k] for l in worker_logs) / len(worker_logs) for k in worker_logs[0]}
            log_avg.update({k: log[k] for k in log if k not in log_avg})
            log = log_avg
            
        return memory, log
        
    def _collect_samples(self, thread_id, queue, env, policy, custom_reward, running_state,
                        min_batch_size, mean_action, render=False):
        memory = []
        log = {'num_steps': 0, 'num_episodes': 0, 'total_reward': 0, 'avg_reward': 0, 
              'total_c_reward': 0, 'avg_c_reward': 0, 'sample_time': 0}
        
        num_steps = 0
        total_reward = 0
        total_c_reward = 0
        num_episodes = 0
        
        while num_steps < min_batch_size:
            state = env.reset()
            if running_state is not None:
                state = running_state(state)
                
            reward_episode = 0
            c_reward_episode = 0
            
            for t in range(10000):  # Don't run too long
                state_var = torch.tensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if mean_action:
                        action = policy(state_var)[0][0].numpy()
                    else:
                        action = policy.select_action(state_var)[0].numpy()
                
                next_state, reward, done, _ = env.step(action)
                reward_episode += reward
                
                if running_state is not None:
                    next_state = running_state(next_state)
                    
                if custom_reward is not None:
                    c_reward = custom_reward(state, action)
                    c_reward_episode += c_reward
                else:
                    c_reward = reward
                    
                mask = 0 if done else 1
                
                # Get log probability of action
                with torch.no_grad():
                    log_prob = policy.get_log_prob(state_var, torch.tensor(action).unsqueeze(0).to(self.device))[0].item()
                
                memory.append(Transition(state, action, c_reward, next_state, mask, log_prob))
                
                if render:
                    env.render()
                    
                if done:
                    break
                    
                state = next_state
                
            # Log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            total_c_reward += c_reward_episode
            
        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_episodes
        
        if queue is not None:
            queue.put((memory, log))
        else:
            return memory, log