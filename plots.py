import matplotlib.pyplot as plt
import pickle

stats = pickle.load(open('runs/stats.pkl', 'rb'))
scores = stats['scores']
losses = stats['losses']

fig, (ax1, ax2) = plt.subplots(2, 1)
plt.subplot(2, 1, 1)
plt.plot(losses)
# plt.xlim(left=1000)
# plt.yscale('log')
plt.ylabel('loss')

plt.subplot(2, 1, 2)
plt.plot(scores)
# plt.xlim(left=1000)
plt.ylabel('score')
plt.xlabel('Game')

fig.align_ylabels()
plt.show()
