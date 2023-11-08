
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

import maddpg.common.tf_util as U
from magic import MAGICAgentTrainer
from alg.DDPG import DDPGAgentTrainer
import tensorflow.contrib.layers as layers
from tensorboardX import SummaryWriter
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="rescue", help="name of the scenario script")
    parser.add_argument('--random-seed', help='random seed for repeatability', default=2020)
    parser.add_argument("--max-episode-len", type=int, default=60, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=2, help="number of adversaries")
    parser.add_argument("--good-alg", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-alg", type=str, default="MAGIC", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type= int, default=128, help="number of units in the mlp")
    parser.add_argument("--buffer-size", type=int, default=1e6, help="300000number of episodes to optimize at the same time")
    parser.add_argument("--min-buffer-size", type=int, default=1024*60, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="MAGIC", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy/{}/{}/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-predator-dir", type=str, default="", help="directory in which training predator model are loaded")
    parser.add_argument("--load-dir", type=str, default="./policy/ddpg_15step/", help="directory in which pretrained prey model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True, help="set True for loading the pretrained prey models")
    parser.add_argument("--display", action="store_true", default=False, help="set True for loading the predator models and rendering")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=1000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./policy/{}/{}/learning_curves/", help="directory where plot data is saved")

    return parser.parse_args()

def ddpg_network(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        #out = layers.batch_norm(input)

        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        #out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)  tf.truncated_normal_initializer(stddev=0.1)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, goal=None, benchmark=False):
    from envs.environment_prey import MultiAgentEnv
    import envs.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    scenario.max_step_before_punishment = 3
    print('==============================================================')
    print('max_step_before_punishment: ', scenario.max_step_before_punishment)
    print('==============================================================')
    world = scenario.make_world(goal)
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.collision_number,
                            done_callback=scenario.done,
                            other_callbacks=[scenario.set_arrested_pressed_watched])
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    '''adv_trainer'''
    if arglist.adv_alg == 'ddpg' or arglist.adv_alg == 'maddpg':
        adv_trainer = DDPGAgentTrainer
        local_q_func = False if arglist.adv_alg == 'maddpg' else True
        for i in range(num_adversaries):
            trainers.append(adv_trainer("adv_agent_%d" % 0, ddpg_network, ddpg_network, obs_shape_n[:num_adversaries],
                env.action_space[:num_adversaries], i, arglist, local_q_func=local_q_func))
    elif arglist.adv_alg == 'MAGIC':
        adv_trainer = MAGICAgentTrainer
        for i in range(num_adversaries):
            trainers.append(adv_trainer("adv_agent_%d" % 0, obs_shape_n[:num_adversaries],
                env.action_space[:num_adversaries], i, arglist))

    '''good_trainer'''
    if arglist.good_alg == 'ddpg' or arglist.good_alg == 'maddpg':
        trainer = DDPGAgentTrainer
        local_q_func = False if arglist.good_alg == 'maddpg' else True
        for i in range(env.n - num_adversaries):
            trainers.append(trainer(
                "agent_%d" % i, ddpg_network, ddpg_network, obs_shape_n[num_adversaries:], env.action_space[num_adversaries:],
                i, arglist, local_q_func=local_q_func))
    elif arglist.good_alg == 'MAGIC':
        trainer = MAGICAgentTrainer
        for i in range(env.n - num_adversaries):
            trainers.append(trainer(
                "agent_%d" % i, obs_shape_n[num_adversaries:], env.action_space[num_adversaries:], i, arglist))
    return trainers

def reload_previous_models(session, env):
    import gc
    # 加载提前训练好的 prey 策略
    prey_vars = []
    for idx in range(arglist.num_adversaries, env.n):
        var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent_{}'.format(idx))
        prey_vars += var

    saver_prey = tf.train.Saver(var_list=prey_vars)
    saver_prey.restore(session, arglist.load_dir)

    print('[prey] successfully reload previously saved ddpg model({})...'.format(arglist.load_dir))
    del saver_prey
    gc.collect()

def train_new(arglist):
    with U.single_threaded_session():
        print('--scenario:', arglist.scenario,  '  --seed=',arglist.random_seed, ' --num-units=', arglist.num_units)
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        arglist.num_goal = env.n - arglist.num_adversaries

        np.random.seed(arglist.random_seed)
        tf.set_random_seed(arglist.random_seed)
        env.seed(arglist.random_seed)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        print('obs_shape_n: ', obs_shape_n)
        num_adversaries = min(env.n, arglist.num_adversaries)

        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_alg, arglist.adv_alg))

        # Initialize
        U.initialize()

        agent_params = []
        saver = []

        for i in range(env.n):
            if i < arglist.num_adversaries:
                agent_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adv_agent_%d" % 0))
                variables_to_resotre = [v for v in agent_params[i] if v.name.split('/')[0] == "adv_agent_%d" % 0]
                saver.append(tf.train.Saver(variables_to_resotre))
            else:
                agent_params.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_%d" % (i-arglist.num_adversaries)))
                variables_to_resotre = [v for v in agent_params[i] if v.name.split('/')[0] == "agent_%d" % (i-arglist.num_adversaries)]
                saver.append(tf.train.Saver(variables_to_resotre))
        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        sess = U.get_session()
        # 恢复模型

        if arglist.display or arglist.restore or arglist.benchmark:
            if arglist.display:
                print('Loading previous predator policy...')
                for i in range(arglist.num_adversaries):
                    saver[i].restore(sess, arglist.load_predator_dir + "/model_%i/" % (i) + 'model')
                print('[predator] successfully reload previously saved predator model({})...'.format(arglist.load_predator_dir))

            print('Loading previous prey policy...')
            for i in range(arglist.num_adversaries, env.n):  # 默认加载训练过的 DDPG prey
                saver[i].restore(sess, arglist.load_dir + "/model_%i/" % (i) + 'model')
            print('[prey] successfully reload previously saved ddpg model({})...'.format(arglist.load_dir))

        plots_dir = arglist.plots_dir.format(arglist.scenario, arglist.exp_name)
        save_dir = arglist.save_dir.format(arglist.scenario, arglist.exp_name)
        print("Save_dir:", save_dir)
        log_dir = save_dir + "/logs/"

        os.makedirs(log_dir,exist_ok=True)
        logger = SummaryWriter(str(log_dir))

        episode_rewards_0 = [0.0]
        last_reward_0 = [0.0]
        last_reward_1 = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards
        agent_info = [[[]]]  # placeholder for benchmarking info
        coordination_reach_times = [[0] for _ in range(env.n-arglist.num_adversaries)]

        obs_n = env.reset()

        episode_step = 0
        train_step = 0
        t_start = time.time()
        t0_start = time.time()
        occupied = []

        print(env.n,'agents  Starting iterations...')
        while True:
            action_n = []
            if arglist.adv_alg == 'MAGIC':
                goal_n_adv = []
                for adv_agent, adv_obs in zip(trainers[:arglist.num_adversaries], obs_n[:arglist.num_adversaries]):
                    action, goal_adv = adv_agent.action(adv_obs)
                    action_n.append(action[0])
                    goal_n_adv.append(goal_adv[0])
            else:
                for i in range(arglist.num_adversaries):
                    action_n.append(trainers[i].action(obs_n[i]))

            if arglist.good_alg == 'MAGIC':
                goal_n = []
                for agent, obs in zip(trainers[arglist.num_adversaries:], obs_n[arglist.num_adversaries:]):
                    action, goal = agent.action(obs)
                    action_n.append(action[0])
                    goal_n.append(goal[0])
            else:
                for i in range(arglist.num_adversaries, env.n):
                    action_n.append(trainers[i].action(obs_n[i]))

            new_obs_n, rew_n, done_n, info_n = env.step(action_n, restrict_move=True)
            info_n = info_n['n']
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            for i, agent in enumerate(trainers):
                if i < arglist.num_adversaries:  # adv agent
                    if arglist.adv_alg == 'MAGIC':
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i],
                                         goal_n_adv[i], done_n[i], terminal)
                    else:
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                else:  # good agent 不训练
                    if arglist.good_alg == 'ddpg' or arglist.good_alg == 'maddpg':
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                    elif arglist.good_alg == 'MAGIC':
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], goal_n[i-arglist.num_adversaries], done_n[i], terminal)

            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards_0[-1] += rew
                if i >= arglist.num_adversaries:
                    last_reward_0[-1] += rew
                else:
                    last_reward_1[-1] += rew
                agent_rewards[i].append(rew)
            for discrete_action in range(env.n-arglist.num_adversaries):
                coordination_reach_times[discrete_action][-1] += info_n[0][discrete_action]

            if terminal or done:
                occupied.append(env.prey_occupied())

                logger.add_scalar('ep_occupied', occupied[-1], len(occupied))
                logger.add_scalar('final_ep_rewards_0', episode_rewards_0[-1], len(episode_rewards_0))
                logger.add_scalar('last_rewards_0', last_reward_0[-1], len(last_reward_0))
                logger.add_scalar('last_rewards_1', last_reward_1[-1], len(last_reward_1))

                agent_rewards = [[0.0] for _ in range(env.n)]
                obs_n = env.reset()

                episode_rewards_0.append(0)  # 记录着每一轮的累积奖赏
                last_reward_0.append(0)
                last_reward_1.append(0)
                episode_step = 0

                agent_info.append([[]])
                for coord_count in coordination_reach_times:  # reset coordination times
                    coord_count.append(0)

            train_step += 1

            # for displaying learned policies
            # if arglist.display: #显示,不再训练
            #     time.sleep(0.1) #100ms
            #     env.render_prey()
            #     continue

            # update all trainers, if not in display or benchmark mode
            if train_step % 50 ==0:
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for i, agent in enumerate(trainers):
                    if i < arglist.num_adversaries:
                        loss = agent.update(trainers[:num_adversaries], train_step, len(episode_rewards_0), logger)
                    else:  # 不训练prey
                        continue

            if (terminal or done)and (len(episode_rewards_0) % arglist.save_rate == 0):
                for i in range(env.n):
                    U.save_state(save_dir+"/model_%i/"%i+'model', saver=saver[i])
                print("steps: {}, episodes: {}, occupied:{:.2}, coordination:{},  reward_0: {:.2}, reward_1: {:.2}, time: {}".format(
                    train_step, len(episode_rewards_0),
                    np.mean(occupied[-arglist.save_rate:]),
                    [[np.mean(c[-arglist.save_rate:])] for c in coordination_reach_times],
                    np.mean(last_reward_0[-arglist.save_rate:]),
                    np.mean(last_reward_1[-arglist.save_rate:]),
                    round(time.time()-t_start, 3)))
                t_start = time.time()
            # saves final episode reward for plotting training curve later
            if len(episode_rewards_0) > arglist.num_episodes:
                print('...Finished total of {} episodes. Take {} seconds'.format(len(episode_rewards_0),
                                                                                 round(time.time() - t0_start, 3)))
                print("final_ep_rewards: {}".format(final_ep_rewards))
                break


if __name__ == '__main__':
    arglist = parse_args()
    train_new(arglist)
