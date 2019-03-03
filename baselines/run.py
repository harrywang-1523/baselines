import sys
import multiprocessing
import os.path as osp
import gym
import copy
from collections import defaultdict
import tensorflow as tf
import numpy as np
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from baselines.deepq.models import build_q_func
from baselines.deepq.build_graph import build_act # TODO: build_train??
from baselines.common.vec_env.vec_normalize import VecNormalize

from baselines.deepq.utils import ObservationInput
from baselines.deepq.build_graph import build_adv
import baselines.common.tf_util as U

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

# Import Cleverhans Components
from cleverhans.attacks import FastGradientMethod
import matplotlib.pyplot as plt
from PIL import Image
import csv

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model, debug = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env, debug # model is an ActWrapper


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            # print(env_id) #PongNoFrameskip-v4
            # print(env_type) #Atari
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
       config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
       config.gpu_options.allow_growth = True
       get_session(config=config)

       env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale)

       if env_type == 'mujoco':
           env = VecNormalize(env)

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    model, env, debug = train(args, extra_args)
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.adv:
        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:
                q_func = build_q_func(network='conv_only')
                craft_adv_obs = build_adv(
                    make_obs_tf=lambda name: ObservationInput(env.observation_space, name=name),
                    q_func=q_func, num_actions=env.action_space.n, epsilon= 0.005 * 255,
                    attack=args.adv
                )

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        action_meanings = env.unwrapped.get_action_meanings()
        def initialize_placeholders(nlstm=128,**kwargs):
            return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
        state, dones = initialize_placeholders(**extra_args)

        num_episodes = 0
        num_moves = 0
        num_transfer = 0
        step = 0
        q_value_dict = {}

    while True:
        step = step + 1 # Overall steps. Does not reset to 0 when an episode ends
        num_moves = num_moves + 1
        if args.adv:
            with g.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    adv_obs = craft_adv_obs([obs])
                    adv_obs = np.rint(adv_obs)
                    adv_obs = adv_obs.astype(np.uint8)
            # if step <= 10: # Visualize adversarial observation 
            #     img2 = Image.fromarray(np.asarray(adv_obs[:,:,0]), mode='L')
            #     img2.show()
            prev_state = np.copy(state)
            action, _, _, _ = model.step(obs,S=prev_state, M=dones)
            adv_action, _, state, _ = model.step(adv_obs,S=prev_state, M=dones)
            if (adv_action != action):
                print('Action before: {}, Action after: {}'.format(
                      action_meanings[action[0]], action_meanings[adv_action[0]]))
                num_transfer = num_transfer + 1
            obs, _, done, _ = env.step(adv_action)
        else:
            # Save the q_value in a csv file for analysis
            # q_values = debug['q_values']([obs])
            # diff = np.max(q_values) - np.min(q_values)
            # q_value_dict[step] = diff
            # with open('Breakout.csv', 'w') as f:
            #     for key in q_value_dict:
            #         f.write("%s,%s\n"%(key, q_value_dict[key]))
            # if (diff >= 1.2):
            #     print(diff)
            #     img = Image.fromarray(obs[:,:,0], mode='L')
            #     img.show()
            action, _, state, _ = model.step(obs,S=state, M=dones)

            # env_copy = env.unwrapped.clone_full_state()
            # reward_list = []
            # for i in range(env.action_space.n):
            #     reward_list.append(env.step(i)[1])
            #     env.unwrapped.restore_full_state(env_copy)
            # max_diff = max(reward_list) - min(reward_list)
            # print(max_diff)
            obs, _, done, _ = env.step(action)
        env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            obs = env.reset()
            print(f'Episode {num_episodes}')
            if args.adv:
                print('Percentage of successful attacks: {}'.format(100 * float(num_transfer) / num_moves))
            num_moves = 0
            num_transfer = 0
            num_episodes = num_episodes + 1

    env.close()

if __name__ == '__main__':
    main()
