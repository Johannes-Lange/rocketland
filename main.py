from src.dql import DQL
import sys
import pickle

GAMES = 20000


def main(mode):
    if mode == 'example':
        state = pickle.load(open('runs/trained_net/state_wow.pkl', 'rb'))
        test = DQL()
        test.test(state)

    if mode == 'train':
        agent = DQL()
        # agent.policy_net.load_state(pickle.load(open('runs/trained_net/state_wow.pkl', 'rb')))
        agent.iteration(GAMES)
        agent.test()


if __name__ == '__main__':
    main(sys.argv[1])
