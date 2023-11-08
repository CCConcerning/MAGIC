import argparse
import os

'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
'''
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from magic import MAGICAgentTrainer
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="expand_simple_spread", help="name of the scenario script")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=2019)
    parser.add_argument("--max-episode-len", type=int, default=30, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--buffer-size", type=int, default=1e6, help="number of episodes to optimize at the same time")
    parser.add_argument("--min-buffer-size", type=int, default=1024 * 30, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="MAGIC_s2020", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/{}/4A_3L/{}/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="", help="directory where plot data is saved")

    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from envs.environment_prey import MultiAgentEnv
    import envs.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.done)
    return env


def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    trainer = MAGICAgentTrainer
    for i in range(env.n):
        trainers.append(trainer(
            "agent_%d" % i, obs_shape_n, env.action_space, i, arglist))
    return trainers

def train_new(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        arglist.num_goal = len(env.world.landmarks)

        np.random.seed(arglist.random_seed)
        tf.set_random_seed(arglist.random_seed)
        env.seed(arglist.random_seed)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        trainers = get_trainers(env, obs_shape_n, arglist)

        # Initialize
        U.initialize()

        agent_params = []
        saver = []

        for i in range(env.n):
            agent_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_%d" % i))
            variables_to_resotre = [v for v in agent_params[i] if v.name.split('/')[0] == "agent_%d" % i]
            saver.append(tf.train.Saver(variables_to_resotre))
        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        sess = U.get_session()
        # 恢复模型
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            for i in range(env.n):
                saver[i].restore(sess, arglist.load_dir + "/model_%i/" % (i) + 'model')

        save_dir = arglist.save_dir.format(arglist.scenario, arglist.exp_name)
        print("save_dir:", save_dir)
        log_dir = save_dir + "/logs/"
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(str(log_dir))

        episode_rewards = [0.0]
        last_reward = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []

        agent_info = [[[]]]  # placeholder for benchmarking info

        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        t0_start = time.time()
        occupied = []

        print(env.n, 'agents  Starting iterations...')
        while True:
            action_n = []
            goal_n = []

            for agent, obs in zip(trainers, obs_n):
                action, goal = agent.action(obs)
                action_n.append(action[0])
                goal_n.append(goal[0])

            # environment step 环境更新
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], goal_n[i], done_n[i], terminal)

            obs_n = new_obs_n
            last_reward[-1] += rew_n[0]  # 记录每一个episode的奖励，只算一个agent（智能体的奖励是一样的）
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i].append(rew)

            if done or terminal:  # 一局结束,重置环境
                if arglist.scenario == "expand_simple_spread":
                    occupied.append(env.occupied_multi_goal())
                elif arglist.scenario == "simple_push_ball":
                    occupied.append(env.push_ball_occupied_multi_goal())
                agent_rewards = [[0.0] for _ in range(env.n)]

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)  # 记录着每一轮的累积奖赏
                last_reward.append(0)
                agent_info.append([[]])

            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)  # 100ms
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            if train_step % 50 == 0:
                loss = None

                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step, len(episode_rewards), logger)

            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                for i in range(env.n):
                    U.save_state(arglist.save_dir + "/model_%i/" % i + 'model', saver=saver[i])

                print("steps: {},episodes: {}, occupied: {:.2}, sum_of_reward: {:.3},time: {}".format(
                    train_step, len(episode_rewards),
                    np.mean(occupied[-arglist.save_rate:]),
                    np.mean(episode_rewards[-arglist.save_rate:]),
                    round(time.time() - t_start, 3)))

            if len(episode_rewards) > arglist.num_episodes:
                print('...Finished total of {} episodes. Take {} seconds'.format(len(episode_rewards),
                                                                                 round(time.time() - t0_start, 3)))
                print("final_ep_rewards: {}".format(final_ep_rewards))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train_new(arglist)
