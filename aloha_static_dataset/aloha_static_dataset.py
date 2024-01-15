from typing import Iterator, Tuple, Any

import copy
import cv2
import glob
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from aloha_static_dataset.conversion_utils import MultiThreadedDatasetBuilder

CAM_NAMES = ['cam_high', 'cam_left_wrist', 'cam_low', 'cam_right_wrist']
INSTRUCTIONS = {
    "12_01_ziploc_slide_50_compressed": "open the ziplog bag",
    "1_22_cups_open_compressed": "open the plastic cup",
    "3objects_test": "pack the objects in the bag",
    "aloha_coffee_compressed": "make coffee",
    "aloha_pingpong_test_compressed": "pour one ball from one cup into the other",
    "aloha_screwdriver": "pick up the screwdriver and move it to our left hand",
    "aloha_vinh_cup_compressed": "open the plastic cup",
    "aloha_coffee_new_compressed": "make coffee",
    "aloha_plate_sponge": "wipe the plate with the sponge",
    "candy_compressed": "unwrap the candy",
    "aloha_screw_driver_compressed": "pick up the screwdriver and put it in the cup",
    "thread_velcro_compressed": "close the velcro strap",
    "aloha_vinh_cup_left_compressed": "open the plastic cup",
    "aloha_fork_pick_up_compressed": "pick up the fork and place it on the plate",
    "aloha_pro_pencil_compressed": "pick up the pen and put it on the other side of the table",
    "aloha_towel_compressed": "tear off a paper towel and put it onto the spill",
    "battery_compressed": "put the battery into the remote control",
    "tape_compressed": "tear off a piece of adhesive tape and put it on the box",
}
CROP_DATASETS = ["aloha_pro_pencil_compressed", "aloha_screwdriver"]
FILE_PATH = '/nfs/kun2/datasets/aloha/static_aloha/**/episode_*.hdf5'


def crop_resize(image, crop_h=240, crop_w=320, resize_h=480, resize_w=640, resize=True):
    """
    Helper function to crop the bottom middle (offset by 20 pixels) and resize
    """
    h, w, _ = image.shape
    y1 = h - crop_h - 20  # Subtracting 20 to start 20 pixels above the bottom
    x1 = (w - crop_w) // 2
    cropped = image[y1:y1+crop_h, x1:x1+crop_w]
    return cv2.resize(cropped, (resize_w, resize_h)) if resize else cropped


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""

    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        try:
            with h5py.File(episode_path, "r") as root:
                qpos = root['/observations/qpos'][()]
                image_dict = {cam_name: root[f'/observations/images/{cam_name}'][()]
                              for cam_name in CAM_NAMES}
                action = root['/action'][()]
        except:
            print(f"Failed to load data for {episode_path}")
            return None

        # get language instruction
        instruction, dataset_name = None, None
        for key in INSTRUCTIONS:
            if key in episode_path:
                dataset_name = key
                instruction = INSTRUCTIONS[key]
        if instruction is None:
            raise ValueError(f"Couldn't find instruction for {episode_path}")

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(action.shape[0]):
            # real robot dataset
            try:
                imgs = {cam_name: cv2.imdecode(image_dict[cam_name][i], 1) for cam_name in CAM_NAMES}
            except:
                print(f"Skipping {episode_path}")
                return None
            if 'cam_high' in CAM_NAMES and dataset_name in CROP_DATASETS:
                imgs['cam_high'] = crop_resize(
                    imgs['cam_high'][..., ::-1])[..., ::-1]

            episode.append({
                'observation': {
                    **imgs,
                    'state': qpos[i],
                },
                'action': action[i],
                'discount': 1.0,
                'reward': float(i == (len(action) - 1)),
                'is_first': i == 0,
                'is_last': i == (len(action) - 1),
                'is_terminal': i == (len(action) - 1),
                'language_instruction': instruction,
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)


class AlohaStaticDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        **{cam_name: tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='RGB camera observation.',
                        ) for cam_name in CAM_NAMES},
                        'state': tfds.features.Tensor(
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint pos (two arms + grippers).',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot action for joints in two arms + grippers.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        return {
            'train': glob.glob(FILE_PATH),
        }

