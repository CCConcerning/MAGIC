import numpy as np
from envs.core_prey_1 import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    good_colors = [np.array([0x66, 0x00, 0x66, ]) / 255, np.array([0x00, 0x99, 0xff]) / 255,
                   np.array([0x66, 0xff, 0xff]) / 255]

    num_good_agents = 2
    # agent number
    num_adversaries = 2

    max_step_before_punishment = 3

    r_inner = [
        [100, 1, -10],
        [-10, 10, -5],
        [-10, -5, 0]
    ]
    r_outer = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]


    rewards_range = [
        r_inner, r_outer
    ]
    print('reward definition: ', rewards_range)

    prey_init_pos = np.random.uniform(-1, +1, 2)

    def make_world(self, goal=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = self.num_good_agents

        num_adversaries = self.num_adversaries
        # 0 ~ num_adversaries: are adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.spread_rewards = True if i < num_adversaries else False
            agent.size = 0.05 if agent.adversary else 0.075
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.goal = np.array([0, 1])
        # for reward calculating...
        self.adversary_episode_max_rewards = [0] * self.num_adversaries
        self.end_without_supports = [False] * self.num_adversaries
        self.arrested_time = [0] * self.num_good_agents

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = self.good_colors[i - self.num_adversaries] if not agent.adversary else np.array(
                [0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.alive = True
            if agent.adversary:
                agent.reset_predator()
            else:
                agent.reset_prey()

            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)
            # print('agent state: ', agent.state)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        return []

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, predator, prey, collision_level=Agent.distance_spread[1]):
        delta_pos = predator.state.p_pos - prey.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = prey.size + predator.size * collision_level
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    # define the reward (coordination)
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        # doesn't change (rewarded only when there is a real collision)
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    # print('collision...')
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                # print('agent is out...')
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        # TODO: if bounded, then hidden this code.
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def return_collision_good_agent_idx(self, predator, good_agents, distance_range):
        for idx, good in enumerate(good_agents):
            if self.is_collision(predator, good, distance_range):
                return idx
        return -1

    def set_arrested(self, prey):
        prey.arrested = True
        prey.color = np.array([0.89803922, 0., 0.])
        prey.movable = False

    def set_unarrested(self, prey):
        prey.arrested = False

    def set_watched(self, prey):
        prey.watched = True
        # prey.color = np.array([0.75294118, 0.98431373, 0.17647059])

    def set_unwatched(self, prey):
        prey.watched = False

    def set_pressed(self, prey):
        prey.pressed = True
        prey.movable = False
        # prey.max_speed = 0.65
        # prey.color =  np.array([0.34509804, 0.7372549 , 0.03137255])

    def set_unpressed(self, prey, world):
        prey.pressed = False
        prey.movable = True
        prey.state.p_vel = np.zeros(world.dim_p)
        prey.max_speed = 1.3

    def set_predator_pressed(self, predator, prey_idx):
        if predator.press_prey_idx == -1:  # 没抓过
            predator.press_prey_idx = prey_idx
            predator.press_down_step += 1
        elif predator.press_prey_idx == prey_idx:  # 抓了同一个
            predator.press_down_step += 1
        else:  # 没抓同一个
            predator.press_prey_idx = prey_idx
            predator.press_down_step = 1

    def release_predator_pressed(self, predator, prey_idx):
        if predator.press_prey_idx == prey_idx:
            predator.reset_predator()

    def set_prey_died(self, prey):
        prey.alive = False
        prey.movable = False
        prey.arrested = True

    def set_arrested_pressed_watched(self, world):
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        for dis_idx, distance_range in enumerate(Agent.distance_spread[1:]):
            for prey_idx, prey in enumerate(good_agents):
                collision_num = 0
                for predator in adversaries:
                    if self.is_collision(predator, prey, collision_level=distance_range):
                        collision_num += 1
                        # 处理 predator 状态
                        if dis_idx == 0:
                            self.set_predator_pressed(predator, prey_idx)
                        elif dis_idx == 1:
                            pass
                    else:  # 没抓当前这个
                        if dis_idx == 0:
                            self.release_predator_pressed(predator, prey_idx)

                if dis_idx == 0:
                    if collision_num == self.num_adversaries:
                        self.arrested_time[prey_idx] += 1
                        self.set_arrested(prey)
                        self.set_prey_died(prey)
                        # print("Setting arrested....")
                    elif collision_num >= 1:
                        self.set_pressed(prey)
                        # print("Setting pressed....")
                    elif collision_num == 0:
                        self.set_unarrested(prey)
                        self.set_unpressed(prey, world)
                    if prey.alive == False:
                        self.set_arrested(prey)
                        self.set_prey_died(prey)

                elif dis_idx == 1:
                    if collision_num >= 1:
                        self.set_watched(prey)
                    else:
                        self.set_unwatched(prey)

    def adversary_reward(self, agent, world):
        step_penalize = 0
        Finish = 0
        # Adversaries are rewarded for collisions with agents
        good_agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        for prey_idx, prey in enumerate(good_agents):
            if prey.alive == False:
                Finish += 1

        if Finish == self.num_good_agents:
            return 200

        if agent.collide:
            rew = 0
            distance_range = agent.distance_spread[1]
            for prey_idx, prey in enumerate(good_agents):
                collision_num = 0
                self_collision = 0
                for predator in adversaries:  # 计算与当前prey碰撞的predator的个数
                    if self.is_collision(predator, prey, collision_level=distance_range):
                        collision_num += 1
                        if predator == agent:  # 自己
                            self_collision += 1
                if collision_num == self.num_adversaries and self_collision == 1:
                    rew += self.rewards_range[0][prey_idx][prey_idx]
                    rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                    self.adversary_episode_max_rewards[agent.idx] = rew
                    return rt_rew

                elif collision_num == 1:
                    if self_collision == 1:
                        if Finish == 0:  # 还没有成功抓捕任何一个，虽然在追当前这个活着的，但是没有与队友合作，惩罚
                            # TODO: 所有队友(当前队友只有一个)
                            partners = [a for a in adversaries if a != agent]
                            partner_collision_action_idx = self.return_collision_good_agent_idx(partners[0],
                                                                                                good_agents,
                                                                                                distance_range)
                            if partner_collision_action_idx != -1:  # 在当前观察级别下，队友确实在抓另一个（没合作）
                                rew += self.rewards_range[0][prey_idx][partner_collision_action_idx]
                                if rew > 0:
                                    rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                    self.adversary_episode_max_rewards[agent.idx] = rew
                                    return rt_rew
                                else:
                                    return rew
                            else:  # 队友没在抓其他的，但是也没合作
                                if agent.press_down_step > self.max_step_before_punishment:
                                    self.end_without_supports[agent.idx - 0] = True
                                    # 惩罚 reward
                                    rew += self.rewards_range[0][prey_idx][2]
                                    if rew > 0:
                                        rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                        self.adversary_episode_max_rewards[agent.idx] = rew
                                        return rt_rew
                                    else:
                                        return rew
                                else:
                                    pass
                        else:
                            if prey.alive == False:  # 继续撞已经finish的
                                rew += 100
                                rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                return rt_rew
                            else:
                                partners = [a for a in adversaries if a != agent]
                                partner_collision_action_idx = self.return_collision_good_agent_idx(partners[0],
                                                                                                    good_agents,
                                                                                                    distance_range)
                                if partner_collision_action_idx != -1:  # 队友在撞原来的，我在撞目标
                                    rew += 100
                                    rew += 10
                                    if rew > 0:
                                        rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                        self.adversary_episode_max_rewards[agent.idx] = rew
                                        return rt_rew
                                    else:
                                        return rew
                                else:  # 队友没在抓其他的，但是也没合作
                                    if agent.press_down_step > self.max_step_before_punishment:
                                        self.end_without_supports[agent.idx - 0] = True
                                        # 惩罚 reward
                                        rew = self.rewards_range[0][prey_idx][2]
                                        return rew
                                    else:
                                        pass
                    elif self_collision == 0:  # 我没抓当前这个
                        if Finish == 0:  # 还没有成功抓捕任何一个，当前这个也没抓
                            self_collision_action_idx = self.return_collision_good_agent_idx(agent, good_agents,
                                                                                             distance_range)
                            if self_collision_action_idx != -1:  # 在当前观察级别下，我确实在抓另一个（没与队友合作），惩罚
                                # rew += 1
                                rew += self.rewards_range[0][self_collision_action_idx][prey_idx]
                                rt_rew = rew - self.adversary_episode_max_rewards[agent.idx]
                                self.adversary_episode_max_rewards[agent.idx] = rew
                                return rt_rew

                            else:  # 我没抓另一个，但也没与队友合作，可能在路上，有一定的步数缓冲
                                partners = [a for a in adversaries if a != agent]
                                if partners[0].press_down_step > self.max_step_before_punishment:
                                    self.end_without_supports[agent.idx - 0] = True
                                    rew = self.rewards_range[0][2][prey_idx]
                                    # TODO: 要加生命值的话，在这儿加
                                    return rew
                                else:
                                    pass
                        else:
                            if not prey.alive:
                                pass
                            else:
                                self_collision_action_idx = self.return_collision_good_agent_idx(agent, good_agents,
                                                                                                 distance_range)
                                if self_collision_action_idx != -1:  # 在当前观察级别下，我确实在抓另一个（没与队友合作），惩罚
                                    rew = -10
                                    return rew
                                else:  # 我没抓另一个，但也没与队友合作，可能在路上，有一定的步数缓冲
                                    partners = [a for a in adversaries if a != agent]
                                    if partners[0].press_down_step > self.max_step_before_punishment:
                                        self.end_without_supports[agent.idx - 0] = True
                                        # 惩罚 reward
                                        rew = -10
                                        # TODO: 要加生命值的话，在这儿加
                                        return rew
                                    else:  # TODO 这里考虑还要不要惩罚（按着的时候），我在路上，还没有超过时间限制
                                        pass
        return step_penalize

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        goal = [world.goal]
        if not agent.adversary:  # 被捕食者观察所有其他智能体的相对位置，以及队友的速度
            for other in world.agents:
                if other is agent: continue  # 跳过当前 agent
                comm.append(other.state.c)  # 2
                other_pos.append(other.state.p_pos - agent.state.p_pos)  # 2
                if not other.adversary:  # 我的观察里有被捕食者，添加其速度,我是被捕食者
                    other_vel.append(other.state.p_vel)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
        else:  # 捕食者观察
            goal_info = []
            for other in world.agents:
                if other is agent: continue  # 跳过当前 agent
                comm.append(other.state.c)  # 2
                if not other.adversary:  # 观察被捕食者的相对位置和速度
                    goal_info.append(other.color)
                    goal_info.append(other.state.p_vel)
                    goal_info.append(other.state.p_pos - agent.state.p_pos)  # 2
                else:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + goal_info + goal)

    def done(self, agent, world):
        agents = self.good_agents(world)
        finish = 0
        for ag in agents:
            if not ag.alive:
                finish += 1
        if finish == self.num_good_agents:
            return True
        elif finish == 1 and not agents[1].alive:
            return True
        else:
            return False

    def collision_number(self, agent, world):
        # 只要达到了coordination 就终止（不管最优次优）
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        result = {idx: 0 for idx in range(self.num_good_agents)}
        # TODO: For each action
        for idx, ag in enumerate(agents):  # for each good agent, test whether there is a collision
            collision_adv = 0
            for adv in adversaries:
                if self.is_collision(ag, adv):
                    collision_adv += 1
            # For coordination
            if collision_adv == self.num_adversaries:
                result[idx] = 1
        return result

    def info(self, agent, world):
        adj = []
        dis = [np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos))) for other in world.agents]
        zzz = sorted(range(len(dis)), key=lambda k: dis[k])
        for i in zzz[1:3]:
            if dis[i] <= 0.5:
                adj.append(i)
            else:
                adj.append(len(world.agents))
        return adj