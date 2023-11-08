import numpy as np
from envs.core_prey_1 import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        # self.arena_size = 1
        self.NObsRange = 2  # 观察范围，不包括自己
        self.NballsRange = 1 # 可观察的球和目标范围
        print('in env ')

    def make_world(self):
        world = World()
        # set any world properties first
        world.collaborative = False
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 1 # 球
        num_goal = 3  # 目标
        world.goal = None
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05  # agent.size = 0.15
            agent.initial_mass = 1
            agent.goal = 0
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True  # 可以冲突和移动
            landmark.movable = True
            landmark.size = 0.1  # agent.size = 0.15
            landmark.initial_mass = 20
            landmark.index = list(np.eye(len(world.landmarks))[i])
        world.obstacle = [Landmark() for _ in range(num_goal)]
        for i, obstacle in enumerate(world.obstacle):  # 目标
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = False
            obstacle.movable = False
            obstacle.size = 0.010

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.goal = [0, 1, 2]  #  np.random.randint(0, len(world.landmarks))
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.5, 0.5+i*0.3, 0.5])

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):  # 球
            landmark.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        p = []
        p.append(np.array([0, 0]))
        p.append(np.array([0.5, 0]))
        for i, obstacle in enumerate(world.obstacle):  # 目标
            obstacle.color = np.array([0.5, 0.5 + i * 0.3, 0.5])
            obstacle.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)
            obstacle.finish = False

        world.dis_between_goals = [
            np.sqrt(np.sum(np.square(world.obstacle[i].state.p_pos - world.obstacle[i + 1].state.p_pos))) for i in
            range(0, len(world.goal) - 1)]
        world.dis_between_goals.append(0)

        return []


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew2 = 0
        # 离最近目标的距离，如果有多个目标的话。
        rew1 = - np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos)))*2

        for i, g in enumerate(world.goal):
            if g is not world.goal[-1] and world.obstacle[g].finish == True:
                continue
            else:
                dis = np.sqrt(np.sum(np.square(world.landmarks[0].state.p_pos - world.obstacle[g].state.p_pos)))
                if dis <= 0.1:
                    world.obstacle[g].finish = True
                    world.obstacle[g].color = np.array([0.85, 0.55, 0.55])
                rew2 -= dis * 5
                rew2 -= np.sum(world.dis_between_goals[i:]) * 5
                break

        rew = rew1 + rew2
        return rew

    def observation(self, agent, world):
        #目标信息
        goal_info = []
        for i, entity in enumerate(world.obstacle):  # world.entities:
            goal_info.append(np.eye(len(world.obstacle))[i])
            goal_info.append(agent.state.p_pos-entity.state.p_pos)
            goal_info.append(entity.color)

        # 智能体到每个球的距离
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]

        # entity colors
        entity_color = []
        entity_color_smallest = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        ball_pos = []
        ball_pos_smallest = []
        observe_ball_index = []
        for entity in world.landmarks:
            ball_pos.append(agent.state.p_pos - entity.state.p_pos )  # 智能体与球的相对位置

        if len(dists) < self.NballsRange:  # 可观察范围大于智能体个数
            zzz = sorted(range(len(dists)), key=lambda k: dists[k])
            for i in range(len(dists)):
                ball_pos_smallest.append(ball_pos[zzz[i]])
                observe_ball_index.append(world.landmarks[zzz[i]].index)  # 观察目标索引
        else:
            zzz = sorted(range(len(dists)), key=lambda k: dists[k])  # 输出距离的排序索引（升序）。
            for i in range(self.NballsRange):  # 观察距离最近的几个球
                observe_ball_index.append(world.landmarks[zzz[i]].index)  # 观察目标索引
                ball_pos_smallest.append(ball_pos[zzz[i]])
                entity_color_smallest.append(entity_color[zzz[i]])

        '''观察与其他智能体的相对位置'''
        other_pos = []
        other_dis = []
        other_pos_smallest = []

        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 智能体的相对位置
            other_dis.append(np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos))))  # 距离
            #other_vel.append(other.state.p_vel - agent.state.p_vel)  # 相对速度

        if len(other_dis) < self.NObsRange:  # 可观察范围大于智能体个数
            zzz = sorted(range(len(other_dis)), key=lambda k: other_dis[k])
            for i in range(len(other_dis)):
                other_pos_smallest.append(other_pos[zzz[i]])
        else:
            zzz = sorted(range(len(other_dis)), key=lambda k: other_dis[k])  # 输出距离的排序索引（升序）。
            for i in range(self.NObsRange):  # 选择距离最近的智能体
                other_pos_smallest.append(other_pos[zzz[i]])

        goal = [np.array(world.goal)]

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + ball_pos + goal_info + goal)

    def done(self, agent, world):
        return world.obstacle[-1].finish

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