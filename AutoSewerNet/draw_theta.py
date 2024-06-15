from theta_proc import pm
import numpy as np
import matplotlib.pyplot as plt

''' 查看theta值
for i in range(17):
    print(pm[i])   
'''

class Draw_theta(object):
    def __init__(self):
        self.thetas = np.array(pm).reshape((17, 9))

    def draw(self):
        # print(self.thetas)

        for i in range(17):
            y = self.thetas[i].reshape((9, ))
            print('layer'+str(i+1)+'& ', end='')
            for j in range(8):
                np.set_printoptions(precision=4, suppress=True)
                print(str(round(y[j], 4)) + '&', end=' ')
            np.set_printoptions(precision=4, suppress=True)
            print(str(round(y[8], 4)) + '\\\\')

        plt.show()


d = Draw_theta()
d.draw()


