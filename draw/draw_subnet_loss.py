import numpy as np
import matplotlib.pyplot as plt
import re

class Draw(object):
    def __init__(self):
        self.path = r'C:\Users\27966\Desktop\subnets_v3.txt'

    def get_content(self):
        f = open(self.path)
        lines = f.readlines()
        f.close()

        return lines

    def get_losses(self):
        print('extracting loss...')
        lines = self.get_content()
        pattern = 'loss: (\d.\d\d\d\d)'
        losses = []

        for line in lines:
            res = re.search(pattern=pattern, string=line)
            if res is not None:
                losses.append(eval(res.group(1)))

        return losses

    def draw_losses(self, interval=10):
        print('drawing...')
        losses = self.get_losses()
        y = []
        for i in range(len(losses)):
            if i % interval == 0:
                y.append(losses[i])
        x = np.arange(len(y))

        ax = plt.gca()
        c1 = '#000000'
        c2 = '#2F13CF'
        width = 1.2

        ax.spines['top'].set_color(c1)
        ax.spines['bottom'].set_color(c1)
        ax.spines['left'].set_color(c1)
        ax.spines['right'].set_color(c1)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)

        plt.plot(x, y, color=c2)
        plt.xlabel('50 steps', loc='right')
        plt.ylabel('losses', loc='top')
        plt.show()


if __name__ == "__main__":
    d = Draw()
    d.draw_losses()
