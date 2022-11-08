import learning.env as env
import learning.agent as agent
import torch

cuda = torch.cuda.is_available()
print(f"Is CUDA supported by this system? {cuda}")
if cuda:
    print(f"CUDA version: {torch.version.cuda}")
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

device = torch.device("cuda" if cuda else "cpu")
agent = agent.QAgent(env.Environment(), device=device)
agent.learn(epochs=1000)