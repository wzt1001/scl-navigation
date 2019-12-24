import matplotlib.pyplot as plt
import numpy as np

goals = [[-252.8, -26.6, 118.7], [-184.0, -25.9, 89.9], 
         [-121.3, -19.5, 3.8], [-162.1, -27.1, 13.3], [-202.0, -27.3, 11.1],
        [40.9, -20.75, -30.8], [75.5, -28.1, -38.7]]
goals = [[i[0] - 388.5, i[1] + 24.4, i[2] + 80.5] for i in goals]
final_goal = [-342.9, 3.65, 48.0]

def vis_paths(positions, goals=goals, final_goal=final_goal):
    start = [positions[0][0][0], positions[0][0][1], positions[0][0][2]]
    
    # rearrange paths
    pos = []
    for i in range(len(positions)):
        # ignore the last item, as it might start from the origin
        a = [[a[0], a[1], a[2]] for a in positions[i]]
        pos.append(np.asarray(a[:-2]))
    
    # plot the trajectories
    for i in range(len(pos)):
        plt.plot(pos[i][:,0], pos[i][:,2], linewidth=1)
        
    # plot the starting point
    plt.plot(start[0], start[2], marker="v", markersize=5, color="black")

    # plot the goals
    for i in range(len(goals)):
        plt.plot(goals[i][0], goals[i][2], marker='o', markersize=5, color="black")
        
    plt.plot([final_goal[0]], [final_goal[2]], marker='o', markersize=10, color="black")
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.gca()
    ax.set_aspect(1.0)
    plt.savefig('plot.png', dpi=100)
    plt.show()
    