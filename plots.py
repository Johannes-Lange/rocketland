import matplotlib.pyplot as plt
import pickle

scores = pickle.load(open('scores_0.pkl', 'rb'))

plt.figure()
plt.plot(scores)
plt.show()
