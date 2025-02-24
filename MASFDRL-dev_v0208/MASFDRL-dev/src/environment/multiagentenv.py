import torch
from torch_geometric.data import Data
from agent import Agent
from graph_utils import GraphGenerator, FeatureAssigner

class MultiAgentEnvironment:
    def __init__(self, num_nodes, num_edges, feature_dim, num_agents, max_steps=50):
        """
        Initialize a multi-agent environment.
        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges in the graph
            feature_dim: Dimensionality of node features
            num_agents: Number of agents (including fraudsters and detectors)
            max_steps: Maximum number of steps per episode
        """
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.feature_dim = feature_dim
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.step_count = 0

        # Initialize graph structure
        self.graph_generator = GraphGenerator(num_nodes, num_edges)
        self.graph = self.graph_generator.generate_graph()
        self.state = self.graph.x  # Initialize node features
        self.agents = self._initialize_agents()

    def _initialize_agents(self):
        """
        Initialize agents.
        Returns:
            A list containing fraudster and detector agents.
        """
        agents = []
        for i in range(self.num_agents):
            agent_type = "fraudster" if i % 2 == 0 else "detector"
            agents.append(Agent(agent_type, i))
        return agents

    def step(self, actions):
        """
        Execute actions of agents and update the environment.
        Args:
            actions: List of actions for each agent.
        Returns:
            new_state: New state of the environment.
            rewards: Rewards for each agent.
            done: Whether the episode is done.
        """
        self.step_count += 1

        # Update node features (simulate transaction changes)
        for action in actions:
            if action["type"] == "fraudster":
                target = action["target"]
                amount = action["amount"]
                self.state[target] += amount
            elif action["type"] == "detector":
                node = action["node"]
                action["reward"] = 1 if torch.sum(self.state[node]) > 1.5 else -0.5

        rewards = [agent.calculate_reward(self.state) for agent in self.agents]
        done = self.step_count >= self.max_steps
        return self.state, rewards, done

    def reset(self):
        """
        Reset the environment.
        Returns:
            Initial state.
        """
        self.graph = self.graph_generator.generate_graph()
        self.state = self.graph.x
        self.step_count = 0
        for agent in self.agents:
            agent.reset_reward()
        return self.state