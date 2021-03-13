from dql import DQL

al = DQL()
al.model.load_state()
al.iteration(5000)
