import learning.env as env
import learning.agent as agent

agent = agent.QAgent(env.Environment())
agent.learn(epochs=100, steps=100)