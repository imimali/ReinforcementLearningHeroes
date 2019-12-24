'''
    created on 23 June 2019
    
    @author: Gergely
'''
import gym

env=gym.make('Breakout-v0')
env.reset()
obss=[]
rews=[]
for _ in range(100):
    obs,rew,done,info=env.step(env.action_space.sample())
    rews.append(rew)
    img=env.render('rgb_array')
    obss.append(img)
    if done:
        env.reset()

import matplotlib.pyplot as plt
plt.imshow(obss[10])
plt.show()

