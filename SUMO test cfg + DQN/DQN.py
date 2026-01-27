import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traci


# CONFIGURATION

SUMO_BINARY = "sumo"          # use "sumo-gui" to visualize
SUMO_CONFIG = "4-point.sumocfg"

# All in seconds

FIXED_YELLOW = 4.0
MIN_GREEN = 10.0
MAX_GREEN = 60.0
GREEN_STEP = 5.0


EPISODES = 300
SIM_TIME = 1800         


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN hyperparameters

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 50000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_PER_EPISODE = 0.990

TARGET_UPDATE = 1000

Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


# REPLAY MEMORY

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# DQN NETWORK

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# SUMO ENVIRONMENT

class SumoIntersectionEnv:

    def __init__(self):
        self.tls_id = None
        self.ns_lanes = []
        self.ew_lanes = []
        self.green_times = {0: 42.0, 2: 42.0}
        self.sim_time = SIM_TIME

    def start(self):
        traci.start([
            SUMO_BINARY,
            "-c", SUMO_CONFIG,
            "--start",
            "--no-step-log",
            "--log", "sumo.log"
        ])
        self._init_tls()


    def close(self):
        traci.close()

    def reset(self):
        try:
            traci.close()
        except:
            pass
        self.start()

        # Let vehicles load
        for _ in range(10):
            traci.simulationStep()

        return self.get_state()

    def _init_tls(self):
       
        self.tls_id = traci.trafficlight.getIDList()[0]

        self.ns_lanes = set()
        self.ew_lanes = set()

        links = traci.trafficlight.getControlledLinks(self.tls_id)

        # Arbitration with sumo versions, so:

        try: 
            logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.tls_id)[0]
        except Exception:
            logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]

        # Phase 0 -> NS green
        phase0 = logic.phases[0]
        for i, signal in enumerate(phase0.state):
            if signal in ("G", "g"):
                incoming_lane = links[i][0][0]
                self.ns_lanes.add(incoming_lane)

        # Phase 2 -> EW green
        phase2 = logic.phases[2]
        for i, signal in enumerate(phase2.state):
            if signal in ("G", "g"):
                incoming_lane = links[i][0][0]
                self.ew_lanes.add(incoming_lane)

        self.ns_lanes = list(self.ns_lanes)
        self.ew_lanes = list(self.ew_lanes)

    def get_state(self):

    # Queue / pressure
        ns_pressure = sum(
            traci.lane.getLastStepVehicleNumber(l)
            for l in self.ns_lanes
        )
        ew_pressure = sum(
            traci.lane.getLastStepVehicleNumber(l)
            for l in self.ew_lanes
        )

        # Current signal info
        phase = traci.trafficlight.getPhase(self.tls_id)
        green_time_left = traci.trafficlight.getNextSwitch(self.tls_id) \
                          - traci.simulation.getTime()

        # Normalize values
        ns_pressure /= 50.0
        ew_pressure /= 50.0
        green_time_left /= MAX_GREEN

        # One-hot encoding of current green
        ns_green = 1.0 if phase == 0 else 0.0
        ew_green = 1.0 if phase == 2 else 0.0

        return np.array(
           [
                ns_pressure,
                ew_pressure,
                ns_green,
                ew_green,
                green_time_left
            ],
            dtype=np.float32
        )


    def step(self, action):

        reward = 0.0
        done = False

        phase = traci.trafficlight.getPhase(self.tls_id)

        # Only act on green phases
        if phase not in [0, 2]:
            traci.simulationStep()
            return self.get_state(), 0.0, done, {}

        # Adjust green time
        if action == 0:
            self.green_times[phase] = max(MIN_GREEN, self.green_times[phase] - GREEN_STEP)
        elif action == 2:
            self.green_times[phase] = min(MAX_GREEN, self.green_times[phase] + GREEN_STEP)

        # GREEN
        for _ in range(int(self.green_times[phase])):
            traci.simulationStep()

            ns_pressure = sum(
                traci.lane.getLastStepVehicleNumber(l)
                for l in self.ns_lanes
            )

            ew_pressure = sum(
                traci.lane.getLastStepVehicleNumber(l)
                for l in self.ew_lanes
            )

            reward -= abs(ns_pressure - ew_pressure)

            if traci.simulation.getTime() >= self.sim_time:
                done = True
                break

        # YELLOW
        if not done:
            traci.trafficlight.setPhase(self.tls_id, phase + 1)
            for _ in range(int(FIXED_YELLOW)):
                traci.simulationStep()
                if traci.simulation.getTime() >= self.sim_time:
                    done = True
                    break

        # "Do nothing" penalization
        if action == 1:
            reward -= 2.0

        next_state = self.get_state()
        return next_state, reward, done, {}
    
# AGENT

class Agent:
    def __init__(self, state_dim):
        self.policy = DQN(state_dim).to(DEVICE)
        self.target = DQN(state_dim).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        self.eps = EPS_START
        self.learn_steps = 0

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 2)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            return self.policy(s).argmax().item()

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))
        s = torch.from_numpy(np.array(batch.state, dtype=np.float32)).to(DEVICE)
        a = torch.from_numpy(np.array(batch.action, dtype=np.int64)).unsqueeze(1).to(DEVICE)
        r = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).unsqueeze(1).to(DEVICE)
        s2 = torch.from_numpy(np.array(batch.next_state, dtype=np.float32)).to(DEVICE)


        q = self.policy(s).gather(1, a)
        with torch.no_grad():
            q2 = self.target(s2).max(1)[0].unsqueeze(1)
            target = r + GAMMA * q2

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.learn_steps % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.learn_steps += 1

    def decay_epsilon(self):
        self.eps = max(EPS_END, self.eps * EPS_DECAY_PER_EPISODE)


# TRAINING LOOP

def train():
    env = SumoIntersectionEnv()
    agent = Agent(state_dim=5)

    for ep in range(EPISODES):
        state = env.reset()
        episode_reward = 0

        while(traci.simulation.getTime() <= SIM_TIME):
            action = agent.act(state)
            next_state, reward, _, _ = env.step(action)

            agent.memory.push(state, action, reward, next_state, False)
            agent.learn()

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()

        print(
            f"Episode {ep+1:03d} | "
            f"Reward: {episode_reward:8.1f} | "
            f"Epsilon: {agent.eps:.3f} | "
            f"Avg Q {episode_reward / SIM_TIME:.3f}"
        )

    env.close()
    torch.save(agent.policy.state_dict(), "dqn_astc_4phase.pth")

if __name__ == "__main__":
    train()
