import matplotlib.pyplot as plt
import numpy as np

x = np.arange(24)   # 1-24epoch, 8 as boundary
x += 1

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

y_lr = np.array([0.01, 0.01, 0.0095, 0.0095, 0.009025, 0.009025, 0.00857375, 0.00857375,
               0.1, 0.1, 0.095, 0.095, 0.04, 0.04, 0.038, 0.038,
               0.005, 0.005, 0.0045, 0.0045, 0.002, 0.002, 0.0018, 0.0018])

y_loss = np.array([80.437, 46.539, 46.143, 45.772, 42.430, 41.734, 41.699, 41.716,
                   62.010, 37.415, 25.810, 31.341, 36.102, 21.388, 42.725, 39.900,
                   33.314, 38.085, 26.235, 14.710, 4.392, 2.184, 2.166, 2.088])

y_acc = np.array([0.8904, 0.9168, 0.9168, 0.9168, 0.9169, 0.9169, 0.9169, 0.9169,
                  0.6371, 0.6000, 0.8509, 0.8216, 0.7851, 0.8703, 0.7478, 0.7708,
                  0.8118, 0.7801, 0.8447, 0.9166, 0.8968, 0.9163, 0.9168, 0.9168])

def draw_lr():
    plt.plot(x, y_lr, c='#111111', linewidth=5)
    plt.xlabel('epoches')
    plt.ylabel('learning rate')
    plt.savefig('../outputs/SupernetLR.pdf', dpi=300)
    plt.close()

def draw_loss():
    plt.plot(x, y_loss, c='#111111', linewidth=5)
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.savefig('../outputs/SupernetLoss.pdf', dpi=300)
    plt.close()

def draw_acc():
    plt.plot(x, y_acc, c='#111111', linewidth=5)
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.savefig('../outputs/SupernetACC.pdf', dpi=300)
    plt.close()

draw_lr()
draw_acc()
draw_loss()
