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

    model, env, debug = train(args, extra_args) # Get the trained model
    env.close()

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.adv_alg: # If attack is applied, build the function for crafting adversarial observations
        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:
                q_func = build_q_func(network='conv_only')
                craft_adv_obs = build_adv(
                    make_obs_tf=lambda name: ObservationInput(env.observation_space, name=name),
                    q_func=q_func, num_actions=env.action_space.n, epsilon=args.epsilon,
                    attack=args.adv_alg
                )

    if args.save_info: # Save all the information in a csv filter
        name = args.info_name
        csv_file = open('/Users/harry/Documents/info/' + name, mode='a' )
        fieldnames = ['episode', 'diff_type', 'diff', 'epsilon', 'steps', 'attack rate', 'success rate', 'score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

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
        num_success_attack = 0
        num_attack = 0
        step = 0
        q_value_dict = {}
        old_diff = 0

        diff_type = args.diff_type
        print("Type of diff: {}. Threshold to launch attack: {}".format(diff_type, args.diff))
        print('-------------------------Episode 0 -------------------------')
        while True:
            step = step + 1 # Overall steps. Does not reset to 0 when an episode ends
            num_moves = num_moves + 1
            q_values = debug['q_values']([obs])
            q_values = np.squeeze(q_values)

            minus_diff = np.max(q_values) - np.min(q_values)
            div_diff = np.max(q_values) / np.min(q_values)
            sec_ord_diff = minus_diff - old_diff
            old_diff = minus_diff

            if args.save_q_value: # Save the q value to a file
                with open('/Users/harry/Documents/q_value_pong_ep' + str(num_episodes+1) + '_diff' + str(args.diff) + '.csv', 'a') as q_value_file:
                    q_value_writter = csv.writer(q_value_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    q_value_writter.writerow(q_values)

            if args.adv_alg:
                diff = minus_diff if args.diff_type == 'diff' else div_diff \
                                  if args.diff_type == 'div_diff' else sec_ord_diff \
                                  if args.diff_type == 'sec_ord_diff' else minus_diff

                if diff >= args.diff:
                    num_attack = num_attack + 1
                    with g.as_default():
                        with tf.Session() as sess:
                            sess.run(tf.global_variables_initializer())
                            adv_obs = craft_adv_obs([obs])[0] # Get the adversarial observation
                            adv_obs = np.rint(adv_obs)
                            adv_obs = adv_obs.astype(np.uint8)

                    if args.preview_image: # Show a few adversarial images on the screen
                        if num_attack >= 2 and num_attack <= 10:
                            adv_img = Image.fromarray(np.asarray(adv_obs[:,:,0]), mode='L')
                            adv_img.show()

                    if args.save_image: # Save one episode of adversarial images in a folder
                        if num_episodes == 0:
                            img = Image.fromarray(np.asarray(adv_obs[:,:,0]), mode='L')
                            img.save('/Users/harry/Documents/adv_19_99/adv_image_' + str(num_moves) + '.png')

                    prev_state = np.copy(state)
                    action, _, _, _ = model.step(obs,S=prev_state, M=dones)
                    adv_action, _, state, _ = model.step(adv_obs,S=prev_state, M=dones)
                    if (adv_action != action): # Count as a successful atttack
                        # print('Action before: {}, Action after: {}'.format(
                        #       action_meanings[action[0]], action_meanings[adv_action[0]]))
                        num_success_attack = num_success_attack + 1
                    obs, rew, done, info = env.step(adv_action)
                else:
                    action, _, state, _ = model.step(obs,S=state, M=dones)
                    obs, rew, done, info = env.step(action)
                    if args.save_image:
                        img = Image.fromarray(np.asarray(obs[:,:,0]), mode='L')
                        img.save('/Users/harry/Documents/adv_images_ep' + str(num_episodes+1) + '/' + str(num_moves) + '.png')
            else:
                if args.save_image: # Save one episode of normal images in a folder
                    if num_episodes == 0:
                        img = Image.fromarray(np.asarray(obs[:,:,0]), mode='L')
                        img.save('/Users/harry/Documents/normal_obs' + str(num_moves) + '.png')
                action, _, state, _ = model.step(obs,S=state, M=dones)
                obs, _, done, info = env.step(action)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                npc_score = info['episode']['r']
                score = 21 if npc_score < 0 else 21 - npc_score
                obs = env.reset()
                print('Episode {} takes {} time steps'.format(num_episodes, num_moves))
                print('NPC Score: {}'.format(npc_score))
                if args.adv_alg:
                    attack_rate = float(num_attack) / num_moves
                    success_rate = float(num_success_attack) / num_attack
                    print('Percentage of attack: {}'.format(100 * attack_rate))
                    print('Percentage of successful attacks: {}'.format(100 * success_rate))
                    info_dict = {'episode': num_episodes+1,'diff_type': args.diff_type, 'diff': args.diff, 'epsilon': args.epsilon,
                             'steps': num_moves, 'attack rate': attack_rate, 'success rate': success_rate, 'score': score}
                    writer.writerow(info_dict)

                num_moves = 0
                num_transfer = 0
                num_episodes = num_episodes + 1
                num_attack = 0
                num_success_attack = 0
                print(f'-------------------------Episode {num_episodes}-------------------------')

        env.close()

if __name__ == '__main__':
    main()
