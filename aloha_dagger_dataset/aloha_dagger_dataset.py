from typing import Iterator, Tuple, Any

import copy
import cv2
import glob
import json
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from aloha_dagger_dataset.conversion_utils import MultiThreadedDatasetBuilder

CAM_NAMES = ['cam_high', 'cam_left_wrist', 'cam_low', 'cam_right_wrist']

NO_CROP_DATASETS = ["aloha_plate_sponge"]
SKIP_DATASETS = ["aloha_plate_sponge"]
FILE_PATH = '/nfs/kun2/datasets/aloha/dagger_aloha/**/episode_*.hdf5'


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

    for episode_path in paths:
        # load raw data --> this should change for your dataset
        try:
            with h5py.File(episode_path, "r") as root:
                qpos = root['/observations/qpos'][()]
                image_dict = {cam_name: root[f'/observations/images/{cam_name}'][()]
                              for cam_name in CAM_NAMES}
                action = root['/action'][()]
        except:
            print(f"Can't load data for {episode_path}")
            yield None

        # load annotation data
        annotation_file = episode_path[:-4] + 'json'
        annotation_data = []
        try:
            with open(annotation_file, "r") as F:
                annotation_data = json.load(F)
        except:
            print(f"Can't load annotation data for {episode_path}")
            yield None

        for k, annotation_dict in enumerate(annotation_data):
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            next_step_lang_type = annotation_data[min(k, len(annotation_data)-1)]["type"]
            for i in range(annotation_dict['start_timestep'], annotation_dict['end_timestep'] + 1):
                # real robot dataset
                try:
                    imgs = {cam_name: cv2.imdecode(image_dict[cam_name][i], 1)[..., ::-1] for cam_name in CAM_NAMES}
                except:
                    print(f"Can't decode image for {episode_path} step {i}")
                    continue
                if 'cam_high' in CAM_NAMES and not any([name in episode_path for name in NO_CROP_DATASETS]):
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
                    'language_instruction': annotation_dict["command"],
                    'language_type': annotation_dict["type"],
                    'next_language_type': next_step_lang_type,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'trajectory_segment': k,
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            yield episode_path + '_' + str(k), sample


class AlohaDaggerDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 40   # number of paths converted & stored in memory before writing to disk
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
                    'language_type': tfds.features.Text(
                        doc='Whether language is instruction or correction.'
                    ),
                    'next_language_type': tfds.features.Text(
                        doc='Language type of next annotated segment.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'trajectory_segment': tfds.features.Scalar(
                        dtype=np.int_,
                        doc='Index of language segment in trajectory.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        paths = glob.glob(FILE_PATH)
        print(len(paths))
        filtered_paths = []
        for path in paths:
            if any([k in path for k in SKIP_DATASETS]):
                continue
            filtered_paths.append(path)
        #filtered_paths = filtered_paths[:int(len(filtered_paths) * 0.8)]
        print(len(filtered_paths))

        return {
            'train': filtered_paths,
        }

