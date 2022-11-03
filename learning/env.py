import game.wrapped_flappy_bird as game
import cv2

ACTION_SPACE = 2

class Environment:
    def __init__(self):
        self.game = game.GameState()
        self.action_space = ACTION_SPACE
        
    def step(self, action):
        next_frame, reward, done = self.game.frame_step(action)
        next_frame = cv2.cvtColor(cv2.resize(next_frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, next_frame = cv2.threshold(next_frame, 1, 255, cv2.THRESH_BINARY)
        return next_frame, reward, done