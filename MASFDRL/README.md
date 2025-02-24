# MASFDRL - Multi-Agent Fraud Detection with Reinforcement Learning

## Overview
This repository contains the source code and deployment files for a reinforcement learning (RL) simulation environment designed for multi-agent fraud detection. The project aims to simulate a realistic environment where multiple agents (fraudsters and detectors) interact, and RL algorithms are used to detect fraudulent transactions. The system is designed to be deployed in a Kubernetes (K8S) environment with significant hardware resources.

## Main Code
Below are the primary code files included in this repository:

- **`trainmultiagent.py`**: The main training script that orchestrates the training process, including initializing the environment, executing actions, and updating policies.
- **`policynetwork.py`**: Defines the policy networks for the fraudster and detector agents, including the neural network architecture and methods for generating actions and updating policies.
- **`multiagentenv.py`**: Defines the multi-agent environment, including the simulation of transactions, state updates, and reward calculations.
- **`syntheticdata.py`**: Generates synthetic transaction data to simulate normal and fraudulent activities.
- **`masfd-rl-start-0.0.1-SNAPSHOT.jar`**: The compiled Java code for the RL simulation environment (if applicable).
- **`Dockerfile`**: Used to build a Docker image for the simulation environment.
- **`masfd-rl.yaml`**: Kubernetes deployment file for deploying the simulation environment in a Kubernetes cluster.

## Main Files
The repository contains the following key files:

- **`trainmultiagent.py`**: Main training script.
- **`policynetwork.py`**: Policy networks for agents.
- **`multiagentenv.py`**: Multi-agent environment definition.
- **`syntheticdata.py`**: Synthetic data generation script.
- **`masfd-rl-start-0.0.1-SNAPSHOT.jar`**: Compiled Java code (if applicable).
- **`Dockerfile`**: Docker image build file.
- **`masfd-rl.yaml`**: Kubernetes deployment file.

## Prerequisites
- A Kubernetes (K8S) environment with sufficient hardware resources.
- Docker installed on your machine to build the Docker image.
- Python 3.x installed to run the `trainmultiagent.py` script.

## Getting Started
### Building the Docker Image
1. Navigate to the root directory of the project.
2. Run the following command to build the Docker image:
   ```sh
   docker build -t masfd-rl .
   ```

### Running the Simulation
1. Deploy the simulation environment using the `masfd-rl.yaml` file:
   ```sh
   kubectl apply -f masfd-rl.yaml
   ```
2. Verify that the deployment is successful by running:
   ```sh
   kubectl get pods
   ```
3. Once the pod is running, you can interact with the simulation using the `masfd.py` script.

## Detailed Code Explanation
### policynetwork.py
- **PolicyNetwork**: Defines a neural network that outputs action probabilities based on the input state.
- **MultiAgentPolicy**: Manages the policy networks for fraudster and detector agents, including methods for generating actions and updating policies.

### trainmultiagent.py
- **train_multi_agent**: Contains the training loop, which initializes the policy manager, resets the environment, executes actions, updates policies, and records training rewards.

### multiagentenv.py
- **MultiAgentEnvironment**: Defines the environment where agents interact. It includes methods for initializing the environment, updating states based on agent actions, and calculating rewards.

### syntheticdata.py
- **generate_synthetic_data**: Generates synthetic transaction data with normal and fraudulent activities, which is used to simulate the environment.

## Validation Results
### Training Performance
The graph below illustrates the training performance of our multi-agent system, showcasing the cumulative rewards over the course of training. This metric is critical in evaluating the learning progress of our agents.

![Training Performance](docs/training_performance.png)

### Key Metrics
Our evaluation focuses on the following key metrics commonly used in fraud detection tasks:

- **Accuracy**: The proportion of transactions correctly identified as fraudulent or legitimate.
- **Precision**: The ratio of correctly identified fraudulent transactions to all transactions flagged as fraudulent.
- **Recall**: The ratio of correctly identified fraudulent transactions to all actual fraudulent transactions.

The results of these metrics after training are as follows:

- **Accuracy**
- **Precision**
- **Recall**

These results indicate that our MASFDRL system is highly effective in detecting fraudulent activities while maintaining a low rate of false positives.

## Usage Examples
Here is an example of how to use our code:

```python
# Initialize environment and policy manager
env = MultiAgentEnvironment(...)
policy_manager = MultiAgentPolicy(...)

# Start training
rewards_history = train_multi_agent(env, policy_manager)

## Contributing
We welcome contributions to this project. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Create a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or issues, please open an issue on this GitHub repository.

### Future Work
Looking ahead, we are excited to extend this project in several promising directions. Our future work will focus on:

- **Enhanced Multi-Agent Interactions**: We aim to explore more sophisticated models of interaction between fraudsters and detectors, potentially incorporating elements of game theory and advanced AI techniques.

- **Improved RL Algorithms**: We plan to investigate and implement more efficient reinforcement learning algorithms to enhance the speed and quality of learning.

- **Scalability and Robustness**: We will assess the scalability of our system and its robustness against various types of fraudulent activities.

- **Real-World Data Integration**: We intend to test our system with real-world transaction data to validate its practical applicability and effectiveness.

These enhancements will not only deepen our understanding of multi-agent systems in the context of fraud detection but also contribute to the broader field of AI and machine learning.
