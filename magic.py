import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U

from maddpg.common.distributions import make_pdtype
from maddpg.common.distributions2 import gumbel_softmax
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer_goal, ReplayBuffer
import tensorflow.contrib.layers as layers


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0 - polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def kl(self_logits, other_logits):
    a0 = self_logits - U.max(self_logits, axis=1, keepdims=True)
    a1 = other_logits - U.max(other_logits, axis=1, keepdims=True)  # q(x)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = U.sum(ea0, axis=1, keepdims=True)  # a0 - tf.log(z0) = log(p(x))
    z1 = U.sum(ea1, axis=1, keepdims=True)  # a1 - tf.log(z1) = log(q(x))
    p0 = ea0 / z0  # p(x)
    return U.sum(p0 * (a0 - tf.math.log(z0) - a1 + tf.math.log(z1)), axis=1)


def policy(spilit_size, p_input, goal_input, goal_output, num_outputs, scope, stage=3, reuse=tf.AUTO_REUSE,
           num_units=128):
    with tf.variable_scope(scope, reuse=reuse):
        o_task = goal_input
        o_agents = p_input[:, :spilit_size]
        feature_agents = layers.fully_connected(o_agents, num_outputs=num_units, activation_fn=tf.nn.relu)

        o_g = tf.split(p_input[:, spilit_size:-stage], goal_output, axis=1)
        feature_g = layers.fully_connected(o_g, num_outputs=num_units, activation_fn=tf.nn.relu)

        feature_agents_goal = tf.concat([feature_agents, o_task], 1)
        query = layers.fully_connected(feature_agents_goal, num_outputs=num_units, activation_fn=tf.nn.relu)
        keys = feature_g
        outputs = tf.matmul(tf.expand_dims(query, 1), tf.transpose(keys, [1, 2, 0]))
        logits = outputs / np.sqrt(num_units)
        logits = tf.reshape(logits, [-1, logits[0].shape[1]])
        y = tf.nn.softmax(logits)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=1, keepdims=True)), y.dtype)

        goal_value = tf.reduce_sum(tf.transpose(feature_g, [1, 2, 0]) * tf.expand_dims(y, 1), axis=2)

        out2 = tf.concat([feature_agents, goal_value], 1)
        out2 = layers.fully_connected(out2, num_outputs=num_units, activation_fn=tf.nn.relu)
        action = layers.fully_connected(out2, num_outputs=num_outputs, activation_fn=None)
        return action, y_hard, logits, y


def p_train(spilit_size, goal_shape, stage, make_obs_ph_n, act_space_n, p_index, p_func, q_func, optimizer, tcd_lamda, scd_beta,
            grad_norm_clipping=None, num_units=128, scope="trainer", reuse=tf.AUTO_REUSE, average=False):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        goal_logi_ph_n = [tf.placeholder(tf.float32, [None, None], name="goal_logi" + str(i)) for i in
                          range(len(act_space_n))]

        q_goal_logi = tf.placeholder(tf.float32, [None, stage], name="q_goal_logi")
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        p_input = obs_ph_n[p_index]
        o_task = p_input[:, -stage:]
        p, goal, goal_logist, prob = p_func(spilit_size, p_input, goal_input=o_task, goal_output=goal_shape,
                                            stage=stage, num_outputs=int(act_pdtype_n[p_index].param_shape()[0]),
                                            scope="p_func/generator", num_units=num_units)

        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))  # 网络参数

        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_pd.sample()

        other_obs = list(np.delete(obs_ph_n, p_index, axis=0))
        other_obs = [ob[:, :-stage] for ob in other_obs]
        other_act = list(np.delete(act_ph_n, p_index, axis=0))
        q_input_other = tf.concat(other_obs + other_act, 1)
        if average:
            mean_other_obs = tf.reduce_mean(other_obs, 0)
            mean_other_act = tf.reduce_mean(other_act, 0)
            q_input_other = tf.concat([mean_other_obs, mean_other_act], 1)
        q, target_goal_logist, target_goal = q_func(spilit_size, obs_ph_n[p_index], o_task, act_input_n[p_index],
                                                    q_input_other, 1,
                                                    goal_shape=goal_shape, stage=stage, scope="q_func", reuse=True,
                                                    num_units=num_units)
        q = q[:, 0]
        pg_loss = -tf.reduce_mean(q)

        # 计算KL
        KL = 0
        for i in range(len(act_space_n)):
            if i != p_index:
                KL += kl(goal_logi_ph_n[p_index], goal_logi_ph_n[i])
        partner_KL_loss = tf.reduce_mean(KL)
        scd_KL_loss = tf.reduce_mean(kl(q_goal_logi, goal_logi_ph_n[p_index]))
        # expand_simple_spread & rescue: tcd_lamda = 1, scd_beta=0.001 | push_ball: tcd_lamda = 0.1, scd_beta=0.01
        loss = pg_loss + p_reg * 1e-3 + partner_KL_loss * tcd_lamda + scd_KL_loss * scd_beta
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        train = U.function(inputs=obs_ph_n + act_ph_n + goal_logi_ph_n + [q_goal_logi],
                           outputs=[loss, scd_KL_loss, partner_KL_loss], updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=[act_sample, prob])
        logi = U.function(inputs=[obs_ph_n[p_index]], outputs=goal_logist)
        p_values = U.function([obs_ph_n[p_index]], p)

        target_p, target_goal, target_goal_logist, target_prob = p_func(spilit_size, p_input, goal_input=o_task,
                                                                        goal_output=goal_shape,
                                                                        stage=stage,
                                                                        num_outputs=int(
                                                                            act_pdtype_n[p_index].param_shape()[0]),
                                                                        scope="target_p_func/generator",
                                                                        num_units=num_units)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)

        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()

        target_act = U.function(inputs=[obs_ph_n[p_index]],
                                outputs=[target_act_sample, target_goal])

        return act, logi, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def critic(spilit_size, input_ob, goal_input, self_act, input_other, num_outputs, goal_shape, scope,
           reuse=tf.AUTO_REUSE, stage=3, num_units=128):
    with tf.variable_scope(scope, reuse=reuse):
        o_task = goal_input
        size = spilit_size
        o_agents = input_ob[:, :size]

        o_g = tf.split(input_ob[:, size:-stage], goal_shape, axis=1)

        o_all_agents = tf.concat([self_act, o_agents, input_other], 1)
        feature_all_agents = layers.fully_connected(o_all_agents, num_outputs=num_units, activation_fn=tf.nn.relu)
        feature_g = [layers.fully_connected(o, num_outputs=num_units, activation_fn=tf.nn.relu) for o in o_g]
        # 生成目标
        feature_agents_goal = tf.concat([feature_all_agents, o_task], 1)
        query = layers.fully_connected(feature_agents_goal, num_outputs=num_units, activation_fn=tf.nn.relu)
        keys = feature_g
        outputs = tf.matmul(tf.expand_dims(query, 1), tf.transpose(keys, [1, 2, 0]))
        logits = outputs / np.sqrt(num_units)
        logits = tf.reshape(logits, [-1, logits[0].shape[1]])
        y = tf.nn.softmax(logits)

        goal_value = tf.reduce_sum(tf.transpose(feature_g, [1, 2, 0]) * tf.expand_dims(y, 1), axis=2)

        out = tf.concat([feature_all_agents, goal_value], 1)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out, logits, y


def q_train(spilit_size, goal_shape, stage, make_obs_ph_n, act_space_n, q_index, q_func, optimizer, tcd_alpha,
            grad_norm_clipping=None, scope="trainer", reuse=tf.AUTO_REUSE, num_units=64, average=False):
    with tf.variable_scope(scope, reuse=reuse):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        goal_logi_ph_n = [tf.placeholder(tf.float32, [None, None], name="q_goal_logi" + str(i)) for i in
                          range(len(act_space_n))]
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        other_obs = list(np.delete(obs_ph_n, q_index, axis=0))
        other_obs = [ob[:, :-stage] for ob in other_obs]
        other_act = list(np.delete(act_ph_n, q_index, axis=0))
        q_input_other = tf.concat(other_obs + other_act, 1)
        if average:
            mean_other_obs = tf.reduce_mean(other_obs, 0)
            mean_other_act = tf.reduce_mean(other_act, 0)
            q_input_other = tf.concat([mean_other_obs, mean_other_act], 1)

        o_task = obs_ph_n[q_index][:, -stage:]
        q, target_goal_logist, target_goal = q_func(spilit_size, obs_ph_n[q_index], o_task, act_ph_n[q_index],
                                                    q_input_other, 1,
                                                    goal_shape=goal_shape, stage=stage, scope="q_func",
                                                    num_units=num_units)
        q = q[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # 计算KL
        KL = 0
        for i in range(len(act_space_n)):
            if i != q_index:
                KL += kl(goal_logi_ph_n[i], goal_logi_ph_n[q_index])
        tcd_loss = tf.reduce_mean(KL)
        # expand_simple_spread & rescue: tcd_alpha = 1 | push_ball: tcd_alpha = 0.1
        loss = q_loss + tcd_loss * tcd_alpha

        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph] + goal_logi_ph_n, outputs=[loss, tcd_loss],
                           updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)
        logi = U.function(inputs=obs_ph_n + act_ph_n, outputs=target_goal_logist)
        q_prob = U.function(inputs=obs_ph_n + act_ph_n, outputs=target_goal)
        # target network
        target_q, _, _, = q_func(spilit_size, obs_ph_n[q_index], o_task, act_ph_n[q_index], q_input_other, 1,
                                 goal_shape=goal_shape, stage=stage, scope="target_q_func", num_units=num_units)
        target_q = target_q[:, 0]
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        return train, logi, q_prob, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MAGICAgentTrainer(AgentTrainer):
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        self.goal_shape = args.num_goal  # 目标数量

        if args.scenario == "expand_simple_spread":
            self.spilit_size = 4   # 观察中非目标信息的维度：4 for expand_simple_spread and push_ball, 6 for rescue task
            tcd_alpha = 1  # for critic
            tcd_lamda = 1  # for policy
            scd_beta = 0.001  # for policy
        elif args.scenario == "simple_push_ball":
            self.spilit_size = 4
            tcd_alpha = 0.1
            tcd_lamda = 0.1
            scd_beta = 0.01
        elif args.scenario == "rescue":
            self.spilit_size = 6
            tcd_alpha = 1
            tcd_lamda = 1
            scd_beta = 0.001
        self.stage = args.num_goal  # 阶段数量 = 目标数量

        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())

        self.q_train, self.q_goal_logi, self.q_prob, self.q_update, self.q_debug = q_train(
            scope=self.name,
            spilit_size=self.spilit_size,
            stage=self.stage,
            goal_shape=self.goal_shape,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=critic,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            tcd_alpha=tcd_alpha,
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            average=False
        )
        self.act, self.logi, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,
            spilit_size=self.spilit_size,
            goal_shape=self.goal_shape,
            stage=self.stage,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=policy,
            q_func=critic,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            tcd_lamda=tcd_lamda,
            scd_beta=scd_beta,
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            average=False
        )

        # Create experience buffer
        self.replay_buffer = ReplayBuffer_goal(args.buffer_size, random_seed=args.random_seed)
        self.max_replay_buffer_len = args.min_buffer_size
        self.replay_sample_index = None
        self.matrix = [list(np.eye(self.n)[self.agent_index]) for i in range(args.batch_size)]

    def action(self, obs):
        return self.act(*(list([obs[None]])))

    def q_goal(self, obs_n, act_n):
        return self.q_prob(*(obs_n + act_n))

    def experience(self, obs, act, rew, new_obs, goal, done, terminal):
        self.replay_buffer.add(obs, act, rew, new_obs, goal, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, episode, logger):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        goal_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, goal, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            goal_n.append(goal)
        obs, act, rew, obs_next, goal, done = self.replay_buffer.sample_index(index)

        num_sample = 1
        target_q = 0.0
        goal_logi_n = []
        q_goal_logi_n = []
        for i in range(num_sample):
            target_act_next_n = []
            for i in range(self.n):
                goal_logi_n.append(agents[i].logi(*(list([obs_n[i]]))))
                q_goal_logi_n.append(agents[i].q_goal_logi(*(obs_n + act_n)))
                [target_act_next, target_goal] = agents[i].p_debug['target_act'](*(list([obs_next_n[i]])))
                target_act_next_n.append(target_act_next)
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - 0) * target_q_next
        target_q /= num_sample
        [q_loss, q_KL_loss] = self.q_train(*(obs_n + act_n + [target_q] + q_goal_logi_n))

        [p_loss, p_KL_loss, partner_KL] = self.p_train(
            *(obs_n + act_n + goal_logi_n + [q_goal_logi_n[self.agent_index]]))

        self.p_update()
        self.q_update()
        logger.add_scalars("agent%i/losses" % self.agent_index,
                           {"vf_loss": q_loss,
                            "pol_loss": p_loss,
                            "partner_KL": partner_KL,
                            "p_KL_loss": p_KL_loss,
                            "q_KL_loss": q_KL_loss,
                            },
                           t / 50)

        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
