import matplotlib.pyplot as plt


x = ['320x240', '640x480', '832x624', '1024x768', '1280x960', '1440x1080']
y1 = [0.04958, 0.09807, 0.15769, 0.23795, 0.37336, 0.47570]
y2 = [0.04193, 0.13818, 0.24460, 0.36614, 0.57220, 0.72777]
y3 = [0.08024, 0.24169, 0.37760, 0.54790, 0.87124, 1.11313]
y4 = [0.10610, 0.36903, 0.60164, 0.89698, 1.42379, 1.83285]

plt.plot(x, y1, label='ResNet-4')
plt.plot(x, y2, label='SegNet-4')
plt.plot(x, y3, label='ResNet')
plt.plot(x, y4, label='SegNet')
plt.ylabel('Time in sec')
plt.xlabel('Image resolution')
plt.legend()
plt.grid(True)
plt.show()
