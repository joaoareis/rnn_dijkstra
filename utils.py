from collections import defaultdict
import itertools
import random
from torch.autograd import Variable
import torch
import rnn_util

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance

    def check_edge(self,from_node,to_node):
        return (to_node in self.edges[from_node]) or (from_node in self.edges[to_node])

class GraphGenerator:
    def __init__(self,links):
        self.links = list(set(tuple(sorted(l)) for l in links))  #Removes permutations e.g if there is link (1,2) ans (2,1) removes (2,1)
        self.nodes = set([node for link in links for node in link]) #List of tuples to set
        self.graphs = []
        self.states = []

    def state_to_links(self,state):
        up_links = []
        for i in range(len(state)):
            if state[i]==1:
                up_links.append(self.links[i])
        return up_links

    def gen_graph(self,links):
        g = Graph()
        g.nodes = self.nodes
        for link in links:
            g.add_edge(link[0],link[1],1)
        return g

    def get_state_graph(self):
        return list(zip(self.states,self.graphs))


    def gen_all(self):
        states = list(itertools.product([0, 1], repeat=len(self.links))) #All possibilities of links up and down. Binary list where states[i]=1 means links[i] is up
        for state in states:
            links = self.state_to_links(state)
            g = self.gen_graph(links)
            self.graphs.append(g)
            self.states.append(state)
        return self.graphs

def nx2Graph(nx,distance=1):
    g = Graph()
    g.nodes = set(nx.nodes)
    for edge in nx.edges:
        g.add_edge(edge[0],edge[1],distance)
    return g

def dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
          if node in visited:
            if min_node is None:
              min_node = node
            elif visited[node] < visited[min_node]:
              min_node = node

        if min_node is None:
          break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
          weight = current_weight + graph.distances[(min_node, edge)]
          if edge not in visited or weight < visited[edge]:
            visited[edge] = weight
            path[edge] = min_node

    return visited, path

def path_to_node(dij,from_node,target_node,include_from=False):
    node = target_node
    path = [node]
    while True:
        node = dij[node]
        if node==from_node:
            break
        path.insert(0,node)
    if include_from:
        path.insert(0,from_node)
    return path


def BuildDataset(links,from_node,to_node):
    samples = []
    labels = []
    gg = GraphGenerator(links)
    gg.gen_all()
    for element in gg.get_state_graph():
        state = element[0]
        graph = element[1]
        samples.append(state)
        _,paths = dijsktra(graph,from_node)
        if to_node not in paths:
            next_hop= -1
        else:
            next_hop = path_to_node(paths,from_node,to_node)[0]
        labels.append(next_hop)
    return samples,labels

def BuildDatatasetRNN(g,nr_samples,verbose=False,torch_format=True):
    X = []
    y = []
    EOS_token = 1000
    for i in range(nr_samples):
        from_node = random.randint(0,len(g.nodes))
        n,p = dijsktra(g,from_node)
        to_node = random.randint(0,len(g.nodes))
        if to_node not in p:
            path = [989]
        else:
            path = path_to_node(p,from_node,to_node,include_from=True)
        path.append(EOS_token)
        x_path = [from_node,to_node,EOS_token]
        if torch_format:
            path = Variable(torch.LongTensor(path))
            x_path = Variable(torch.LongTensor(x_path))
        X.append(x_path)
        y.append(path)
        if i%100==0 and verbose:
            print(i/nr_samples*100)
    return X,y

def checkValidPath(g,path):
    curr_node = path[0]
    for i in range(1,len(path)):
        if path[i] in g.edges[curr_node]:
            curr_node = path[i]
        else:
             return False
    return True

def checkAccuracy(y_true,preds):
    corrects = 0
    for i in range(len(y_true)):
        if list(y_true[i])==preds[i]:
            corrects += 1
    return corrects/len(y_true)

def getPreds(encoder,decoder,X):
    preds = []
    for x in X:
        pred, _ = rnn_util.evaluate(encoder, decoder,x)
        pred = pred[:-1]
        preds.append(pred)
    return preds

def getNumberValids(g,preds):
    valids = 0
    for pred in preds:
        if checkValidPath(g,pred[:-1]):
            valids += 1
    return valids

def splitTrainTest(X,y,p=0.3):
    nr_train = int(len(X)*(1-p))
    return X[:nr_train],y[:nr_train],X[nr_train:],y[nr_train:]
