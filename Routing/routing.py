import numpy as np
import matplotlib.pyplot as plt

class Router(object):
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.neighbor = []
        self.edge=[]

class Edge(object):
	def __init__(self, x, y, l):
		self.start = x
		self.end = y
		self.len = int(int(l*10)/2+1)
		self.load = 0

class Data(object):
	def __init__(self, x, y, size, priority):
		self.now = x
		self.target = y
		self.size = size
		self.priority = priority
		self.time = 0
		self.edge = -1
		self.neigh = [priority,-1,-1,-1]

class Routing:
    def __init__(self, n_agent=20, n_neighbor=3, n_data=20):
        self.n_router = n_agent
        self.n_neighbor = n_neighbor
        self.n_data = n_data
        self.n_obs = (5+n_neighbor*5+n_neighbor*5)
        self.router = []
        self.edges = []
        self.data = []        
        
    def reset(self):
        self.router = []
        self.edges = []
        self.data = []
        self.start()
        obs = self.observation()
        adj = self.get_adj()
        return obs, adj
    
    def start(self):
        self.build_graph()
        self.gen_data()
        
    def seed(self, _seed):
        np.random.seed(seed=_seed)
    
    def cal_dis_2d(self, a: Router, b: Router):
        dis = (a.x - b.x)**2 + (a.y - b.y)**2
        return dis
    
    def cal_dis_3d(self, a: Router, b: Router):
        dis = (a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2
        return dis

    def build_graph(self):
        ####build the graph####
        t_edge = 0
        for i in range(self.n_router):
            self.router.append(Router(np.random.random(),np.random.random()))

        for i in range(self.n_router):
            dis = []
            for j in range(self.n_router):
                dis.append([(self.router[j].x - self.router[i].x)**2 + (self.router[j].y - self.router[i].y)**2, j])
            dis.sort(key = lambda x: x[0],reverse = False)

            for j in range(self.n_router):

                if len(self.router[i].neighbor) == self.n_neighbor:
                    break
                if j == 0:   # self
                    continue
                
                # if len(self.router[dis[j][1]].neighbor) < self.n_neighbor:
    
                #     self.router[i].neighbor.append(dis[j][1])
                #     self.router[dis[j][1]].neighbor.append(i)

                #     if i<dis[j][1]:
                #         self.edges.append(Edge(i,dis[j][1],np.sqrt(dis[j][0])))
                #         self.router[i].edge.append(t_edge)
                #         self.router[dis[j][1]].edge.append(t_edge)
                #         t_edge += 1
                #     else:
                #         self.edges.append(Edge(dis[j][1],i,np.sqrt(dis[j][0])))
                #         self.router[dis[j][1]].edge.append(t_edge)
                #         self.router[i].edge.append(t_edge)
                #         t_edge += 1
                self.edges.append(Edge(i,dis[j][1],np.sqrt(dis[j][0])))
                self.router[i].neighbor.append(dis[j][1])
                self.router[i].edge.append(t_edge)
                if len(self.router[dis[j][1]].neighbor) < self.n_neighbor:
                    self.router[dis[j][1]].neighbor.append(i)
                    self.router[dis[j][1]].edge.append(t_edge)
                t_edge += 1

    def render(self):
        for i in range(self.n_router):
            plt.scatter(self.router[i].x, self.router[i].y, color = 'orange')
        for e in self.edges:
            plt.plot([self.router[e.start].x,self.router[e.end].x],[self.router[e.start].y,self.router[e.end].y],color='black')
        # plt.show()
        plt.savefig('graph.png')
    
    def testadj(self):
        for i in range(self.n_router):
            plt.scatter(self.router[i].x, self.router[i].y, color = 'orange')
        adj = self.get_adj()
        for i in range(self.n_router):
            for j in range(self.n_router):
                if adj[0][i][j] != 0:
                    plt.plot([self.router[i].x,self.router[i].x],[self.router[j].y,self.router[j].y],color='black')
        plt.savefig('graph1.png')

    def gen_data(self):
        for i in range(self.n_data):
            self.data.append(Data(np.random.randint(self.n_router),np.random.randint(self.n_router),np.random.random(),i))

    def observation(self):
        obs = []
        for i in range(self.n_data):
            ob=[]

            ####meta information####
            ob.append(self.data[i].now)
            ob.append(self.data[i].target)
            ob.append(self.data[i].edge)
            ob.append(self.data[i].size)
            ob.append(self.data[i].priority)

            ####edge information####
            for j in self.router[self.data[i].now].edge:
                ob.append(j)
                ob.append(self.edges[j].start)
                ob.append(self.edges[j].end)
                ob.append(self.edges[j].len)
                ob.append(self.edges[j].load)

            ####other datas####
            count = 0;
            self.data[i].neigh = []
            self.data[i].neigh.append(i)

            for j in range(self.n_data):
                if j==i:
                    continue
                if (self.data[j].now in self.router[self.data[i].now].neighbor)|(self.data[j].now == self.data[i].now):
                    count+=1
                    ob.append(self.data[j].now)
                    ob.append(self.data[j].target)
                    ob.append(self.data[j].edge)
                    ob.append(self.data[j].size)
                    ob.append(self.data[i].priority)
                    self.data[i].neigh.append(j)

                if count==self.n_neighbor:
                    break
            for j in range(self.n_neighbor-count):
                self.data[i].neigh.append(-1)
                # self.data[i].neigh.append(0)
                for k in range(5):
                    ob.append(-1) #invalid placeholder
                    # ob.append(0) #invalid placeholder

            obs.append(ob)

        return obs

    def get_adj(self):
        adj = np.zeros((1,self.n_router,self.n_router))
        for i in range(self.n_router):
            for j in self.router[i].neighbor:
                if j != -1:
                    adj[0][i][j] = self.cal_dis_2d(self.router[i], self.router[j])
        return adj

    def set_action(self, act):

        reward = [0]*self.n_data
        done = [False]*self.n_data

        for i in range(self.n_data):
            if self.data[i].edge != -1:
                self.data[i].time -= 1
                if self.data[i].time == 0:
                    self.edges[self.data[i].edge].load -= self.data[i].size
                    self.data[i].edge = -1

            elif act[i] == 0:
                continue

            else:
                t = self.router[self.data[i].now].edge[act[i]-1]
                # t = self.router[self.data[i].now].edge[act[i]]
                if self.edges[t].load + self.data[i].size >1:
                    reward[i] = -0.2 # -0.2
                else:
                    self.data[i].edge = t
                    self.data[i].time = self.edges[t].len
                    self.edges[t].load += self.data[i].size

                    if self.edges[t].start == self.data[i].now:
                        self.data[i].now = self.edges[t].end
                    else:
                        self.data[i].now = self.edges[t].start

            if self.data[i].now == self.data[i].target:
                reward[i] = 10
                done[i] = True

        return reward, done

    def step(self, act):
        reward, done = self.set_action(act)
        obs = self.observation()
        adj = self.get_adj()
        return obs, adj, reward, done
        
    
if __name__ == "__main__":
    env = Routing(20, 4, 20)
    obs, adj = env.reset()
    print(adj.shape)
    print(np.array(env.data).shape)    # (20,) -> (n_agent,)
    print(np.array(env.router).shape)  # (20,) -> (n_data,)
    print(np.array(env.edges).shape)   # (46,) -> 
    # obs = env.observation()
    print(np.array(obs).shape)  # (20, 45) -> (n_data, 5+n_neighbor*5+n_neighbor*5) 5是自身数据，n_neighbor*5:是邻居边的信息，n_neighbor*5是邻居数据信息
    env.render()
    # env.testadj()