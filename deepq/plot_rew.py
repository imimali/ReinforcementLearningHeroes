'''
    created on 23 June 2019
    
    @author: Gergely
'''

with open('outputs.txt', 'r') as f:
    rewards = []
    line = f.readline()
    st='Avg Reward (Last 100): '
    while line != '':

        idx = line.find(st)
        if idx != -1:
            rewards.append(float(line[idx+len(st):idx+len(st)+ 5]))
        line = f.readline()

    xs=[i for i in range(len(rewards))]
    print(rewards)
    import matplotlib.pyplot as plt
    fig=plt.figure()
    fig.suptitle('Average reward in Breakout')
    plt.xlabel('no iter')
    plt.ylabel('reward')
    plt.plot(xs,rewards)
    plt.show()