import os
import shutil

import ray
import ray.rllib.agents.ppo as ppo
from PIL import Image
from ray import tune
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from modified_dungeon import ModifiedDungeon


def make_gif(agent, env, observation, title, num_actions=5):
    frames = []
    for _ in range(num_actions):
        action = agent.compute_single_action(observation)
        data = env._map.render(env._agent)
        frame = Image.fromarray(data).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
        frames.append(frame)

        observation, reward, done, info = env.step(action)
        if done:
            break

    frames[0].save(title, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)


def log_to_writer(writer, result, update_steps):
    writer.add_scalar('Rewards/min', result["episode_reward_min"], update_steps)
    writer.add_scalar('Rewards/mean', result["episode_reward_mean"], update_steps)
    writer.add_scalar('Rewards/max', result["episode_reward_max"], update_steps)

    writer.add_scalar('General/episode_len_mean', result["episode_len_mean"], update_steps)


if __name__ == "__main__":

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env("ModifiedDungeon", lambda config: ModifiedDungeon(**config))

    CHECKPOINT_ROOT = "tmp/ppo/dungeon"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    ray_results = os.getenv("HOME") + "/ray_results1/"
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "ModifiedDungeon"
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
        "observation_size": 11,
        "vision_radius": 5,
        "max_steps": 2000,
        "seed": 10
    }

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0

    log_each = 100
    n_iter = 500
    checkpoint_path = "tmp/ppo/dungeon"
    image_num_actions = 500

    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    # env = Dungeon(50, 50, 3)
    agent = ppo.PPOTrainer(config)

    writer = SummaryWriter(f'logs/ppo-{datetime.now():%Y%m%d-%H%M-%S}')
    for n in range(n_iter):
        result = agent.train()

        file_name = agent.save(CHECKPOINT_ROOT)
        log_to_writer(writer, result, n)

        # sample trajectory
        if n % log_each == 0 or n == n_iter - 1:
            env = ModifiedDungeon(**config['env_config'])
            obs = env.reset()
            make_gif(agent, env, obs, f'iter_{n}.gif', image_num_actions)
