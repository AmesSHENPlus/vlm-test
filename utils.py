import os
import numpy as np
import torch

from datasets import load_dataset, concatenate_datasets
import av

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# from visualize import *

def load_data(task, data_num, data_path):
    if task == "VideoDetailCaption":
        data_video = load_dataset(
            "/workspace/VideoDetailCaption",
            split="test",
            # cache_dir=cache_dir,
        ).shuffle(seed=42).select(range(data_num))

        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video_name']}.mp4")
            return os.path.exists(video_path)

        video_dir = os.path.join(data_path, "Test_Videos")
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVBench':
        data_video_1 = load_dataset(
            "",
            'action_sequence',
            split="train",
            cache_dir=cache_dir,
        ).shuffle(seed=42).select(range(data_num))

        data_video_2 = load_dataset(
            "",
            'action_prediction',
            split="train",
            cache_dir=cache_dir,
        ).shuffle(seed=42).select(range(data_num))

        data_video = concatenate_datasets([data_video_1, data_video_2])
        data_video = data_video.shuffle(seed=42)

        def video_exists(example):
            video_path = os.path.join(video_dir, f"{example['video']}")
            return os.path.exists(video_path)

        video_dir = ""
        filtered_data = data_video.filter(video_exists)
        data_video = filtered_data
    elif task == 'MVLU':
        data_video = load_dataset(
            "",
            split="train",
            cache_dir=cache_dir,
        ).shuffle(seed=42).select(range(data_num))
    elif task == 'LongVideoBench':
        data_video = load_dataset(
            "",
            split="test",
            cache_dir=cache_dir,
        ).shuffle(seed=24).select(range(data_num))
    elif task == 'MMBench':
        data_video = load_dataset(
            "",
            split="train",
            cache_dir=cache_dir,
        ).shuffle(seed=42).select(range(data_num))
    elif task == 'COCO_caption':
        cache_dir = ''
        os.makedirs(cache_dir, exist_ok=True)
        data_video = load_dataset(
            "",
            split="test",
            cache_dir=cache_dir,
        ).shuffle(seed=42).select(range(100))
    else:
        data_video = None

    # print(data_video)
    return data_video



def read_video_pyav(container, indices=None):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)

    if indices is None:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        print(f"INFO: {len(frames)} frames are decoded.")
        return np.stack(frames)
    else:
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        print(f"INFO: {len(frames)} frames are decoded.")
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def get_last_video_idx(input_ids, video_token_id):
    #reverse order
    last_video_idx = -1
    for i in range(len(input_ids)-1, -1, -1):
        if input_ids[i] == video_token_id:
            last_video_idx = i
            break
    return last_video_idx

def clip_input_video(task, data_instance, frame_num=64, model_type='qwen2_5_vl',data_path=None):
    if task == "VideoDetailCaption":
        video_path = os.path.join(data_path, "Test_Videos/")
        video_name = data_instance["video_name"]
        video_path = video_path + video_name + ".mp4"
        question = data_instance["question"]

    elif task == "MVBench":
        video_path = data_path
        video_name = data_instance["video"]
        video_path = video_path + video_name
        question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."

    elif task == 'LongVideoBench':
        video_path = data_path
        video_name = data_instance["video_path"]
        video_path = video_path + video_name
        question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."

    # elif task == "MVLU":
    #     video_reader = data_instance['video']
    #     total_frames = len(video_reader)
    #     print("Total frames:", total_frames)

    #     indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
    #     frames = video_reader.get_batch(indices).asnumpy()


    container = av.open(video_path)
    total_frames = container.streams.video[0].frames

    if total_frames == 0:
        return None, None

    indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
    video = read_video_pyav(container, indices)

    placeholder = "<|video_pad|>"
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt, video