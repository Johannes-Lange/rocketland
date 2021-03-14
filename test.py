from dql import DQL

al = DQL()
al.policy_net.load_state()
al.iteration(20000)

