import numpy as np
from envs.core_prey_1 import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def __init__(self):
        self.Ngoalsrange = 2
        self.NObsRange = 2
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 3
        num_obstacle = 0
        world.collaborative = False
        world.goal = None
        # add agents
        world.agents = [Agent(i) for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            #landmark.boundary = True
        world.obstacle = [Landmark() for i in range(num_obstacle)]
        for i, obstacle in enumerate(world.obstacle):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = True
            obstacle.movable = False
            obstacle.size = 0.08
            obstacle.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.goal = np.random.choice(np.arange(len(world.landmarks)), [len(world.landmarks), ], False) #np.random.randint(0, len(world.landmarks))

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25 + 0.3 * i, 0.25, 0.25])
            landmark.index = np.array(np.eye(len(world.landmarks))[i])

        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        for i, obstacle in enumerate(world.obstacle):
            obstacle.color = np.array([0.55, 0.55, 0.55])
            obstacle.state.p_pos = np.array([0.0, 0.0])
            obstacle.state.p_vel = np.zeros(world.dim_p)
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.5, +0.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.finish = False

        world.dis_between_goals = [
            np.sqrt(np.sum(np.square(world.landmarks[i].state.p_pos - world.landmarks[i + 1].state.p_pos))) for i in
            range(0, len(world.goal) - 1)]
        world.dis_between_goals.append(0)
        return []

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0
        for i, g in enumerate(world.goal):
            if g is not world.goal[-1] and world.landmarks[g].finish == True:
                continue
            else:
                dists = [np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[g].state.p_pos))) for a in world.agents]
                if agent is world.agents[-1] and all(d <= 0.2 for d in dists):  #算到最后一个agent时才改变目标状态
                    world.landmarks[g].finish = True
                    world.landmarks[g].color = np.array([0.85, 0.55, 0.55])
                rew -= np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmarks[g].state.p_pos)))
                rew -= np.sum(world.dis_between_goals[i:])
                break
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_dis = []
        entity_pos_smallest = []
        entity_color = []
        entity_color_smallest = []
        entity_index_smallest = []
        landmark_info = []
        for i, entity in enumerate(world.landmarks):  # world.entities:
            entity_color.append(entity.color)
            landmark_info.append(np.eye(len(world.landmarks))[i])  # 2
            landmark_info.append(agent.state.p_pos-entity.state.p_pos)  # 2
            landmark_info.append(entity.color)  # 3
        landmark_pos = []
        # 观察地标
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            entity_dis.append(np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos))))
            landmark_pos.append(entity.state.p_pos)  # 目标的绝对位置

        if len(entity_dis) < self.Ngoalsrange:  # 可观察范围大于智能体个数
            zzz = sorted(range(len(entity_dis)), key=lambda k: entity_dis[k])
            for i in range(len(entity_dis)):
                entity_pos_smallest.append(entity_pos[zzz[i]])
        else:
            zzz = sorted(range(len(entity_dis)), key=lambda k: entity_dis[k])
            for i in range(self.Ngoalsrange):  # 选择距离最近的
                entity_pos_smallest.append(entity_pos[zzz[i]])
                entity_color_smallest.append(entity_color[zzz[i]])
                entity_index_smallest.append(world.landmarks[zzz[i]].index)

        # agents
        other_pos = []
        other_dis = []
        other_pos_smallest = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)  # 智能体的相对位置
            other_dis.append(np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos))))  # 距离

        if len(other_dis) < self.NObsRange:  # 可观察范围大于智能体个数
            zzz = sorted(range(len(other_dis)), key=lambda k: other_dis[k])
            for i in range(len(other_dis)):
                other_pos_smallest.append(other_pos[zzz[i]])
        else:
            zzz = sorted(range(len(other_dis)), key=lambda k: other_dis[k])  # 输出距离的排序索引（升序）。
            for i in range(self.NObsRange):  # 选择距离最近的智能体
                other_pos_smallest.append(other_pos[zzz[i]])

        goal = [world.goal]
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos]  + landmark_info + goal)

    def done(self, agent, world):
        return world.landmarks[-1].finish

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