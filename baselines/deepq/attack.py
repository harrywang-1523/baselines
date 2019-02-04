import tensorflow as tf
import numpy as np
import argparse
from baselines.common.cmd_util import make_env
from baselines.deepq.build_graph import build_act, build_adv
from baselines.deepq.utils import ObservationInput
from baselines.deepq.models import build_q_func
# from baselines.deepq.q_func import model, dueling_model
from baselines.common.tf_util import load_variables

def build_env(game_id):
    env = make_env(game_id, 'atari', seed=2321, wrapper_kwargs={'frame_stack': True})
    return env

def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN Model")
    parser.add_argument("--env", type=str, required=True, help="Name of the game")
    parser.add_argument("--path", type=str, default=None, help="Load model from this directory")
    parser.add_argument("--dueling", default=True, help="Whether or not to use dueling models")
    parser.add_argument("--stochastic", default=True,  help="whether or not to use stochastic actions according to models eps value")
    parser.add_argument("--attack", type=str, default=None, help="Method to attack the model")
    return parser.parse_args()


class DQNModel:

    def __init__(self, env, dueling, load_path):
        self.g = tf.Graph()
        self.dueling = dueling
        self.env = env
        self.sess = tf.Session(graph=self.g)
        with self.g.as_default():
            self.q_func = build_q_func(network="conv_only", dueling=self.dueling)
            self.act = build_act(
                 make_obs_ph=lambda name: ObservationInput(self.env.observation_space, name=name),
                 q_func= self.q_func,
                 num_actions=self.env.action_space.n
            )

        if load_path:
            print('Loading Model...')
            load_variables(load_path=load_path, sess=self.sess)

    def get_act(self):
        return self.act

    def get_session(self):
        return self.sess

    def craft_adv(self):
        with self.sess.as_default():
            with self.g.as_default():
                craft_adv_obs = build_adv(
                    make_obs_tf = lambda name: ObservationInput(self.env.observation_space, name=name),
                    q_func= self.q_func,
                    num_actions=self.env.action_space.n,
                    epsilon=1.0 / 255.0
                )
        return craft_adv_obs

def play(env, act, craft_adv_obs, stochastic, attack, m_target, m_adv):
    num_episodes = 0
    num_moves = 0
    num_transfer = 0

    obs = env.reset()
    while True:
        env.unwrapped.render()
        if attack:
            with m_adv.get_session().as_default():
                m_adv.get_session().run(tf.global_variables_initializer())
                # adv_obs = craft_adv_obs(np.array(obs)[None],stochastic_ph_adv=stochastic)[0]
                adv_obs = craft_adv_obs(np.array(obs)[None])[0]
                # print(adv_obs)
            with m_target.get_session().as_default():
                action = act(np.array(adv_obs)[None], stochastic=stochastic)[0]
                action2 = act(np.array(obs)[None], stochastic=stochastic)[0]
                num_moves = num_moves + 1
                if (action != action2):
                    num_transfer = num_transfer + 1
        else:
            action = act(np.array(obs)[None], stochastic=stochastic)[0]

        obs, rew, done, info = env.step(action)
        if done:
            obs = env.reset()

        # if len(info['rewards']) > num_episodes:
        #     print('Reward: ' + str(info["rewards"][-1]))
        #     num_episodes = len(info["rewards"])
        print('Episode: ' + str(num_episodes))
        success = float((num_transfer) / num_moves) * 100.0
        print("Percentage of successful attacks: " + str(success))
        num_moves = 0
        num_transfer = 0

if __name__ == '__main__':
    args = parse_args()
    env = build_env(game_id=args.env)
    # g1 = tf.Graph()
    # with g1.as_default():
    m = DQNModel(env, args.dueling, args.path)
    with m.g.as_default():
        with m.get_session().as_default():
            craft_adv_obs = m.craft_adv()
            play(env, m.get_act(), craft_adv_obs, args.stochastic, args.attack, m, m)
