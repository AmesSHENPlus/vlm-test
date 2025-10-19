import os
import numpy as np
import torch

from datasets import load_dataset, concatenate_datasets
import av

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# from visualize import *


def load_model(base_model_path):
    processor = AutoProcessor.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)
    model = AutoProcessor.from_pretrained(
        base_model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        attn_implementation = "sdpa",
    )
    video_token_id = 151656

    # video_token_id = model.config.video_token_id
    # print("video_token_id:",video_token_id)

    return model, processor, video_token_id


def load_data(task, data_num, data_path):
    if task == "VideoDetailCaption":
        data_video = load_dataset(
            "/workspace/vlm-test/VideoDetailCaption",
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

def clip_input(processor, data_instance):
    image = data_instance["image"]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please provide a detailed description of the image, focusing on the main subjects, their actions, and the background scenes."},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    print("input id length:", inputs['input_ids'].shape[1])
    return inputs


def clip_input_video(processor, task, data_instance, frame_num=64, model_type='llava_ov',data_path=None):
    if model_type == 'llava_ov':
        if task == "VideoDetailCaption":
            video_path = os.path.join(data_path, "Test_Videos/")
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"

            question = data_instance["question"]
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == "MVBench":
            video_path = data_path
            video_name = data_instance["video"]
            video_path = video_path + video_name

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == 'LongVideoBench':
            video_path = data_path
            video_name = data_instance["video_path"]
            video_path = video_path + video_name

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": question},
                    ],
                },
            ]

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            if total_frames == 0:
                return None
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == "MVLU":
            video_reader = data_instance['video']

            total_frames = len(video_reader)
            # print("Total frames:", total_frames)
            indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
            video = video_reader.get_batch(indices).asnumpy()

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
                {

                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": question},
                    ],
                },
            ]

        # display_frame_grid(video)
        # save_frames(video)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda")

    elif model_type == 'qwen2_5_vl':
        def calculate_fps_for_target_frames(container, target_frames):
            video_stream = container.streams.video[0]
            duration = container.duration / 1000000
            if duration <= 0:
                return 1.0

            required_fps = target_frames / duration
            print(f"INFO: Duration: {duration:.2f}s, frame_num: {target_frames}, fps: {required_fps:.2f}")
            return required_fps

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
        # print("Total frames:", total_frames)

        if total_frames == 0:
            return None

        fps = calculate_fps_for_target_frames(container, frame_num)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 448*448,
                        "fps": fps,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")

    print("INFO: Input length:", inputs['input_ids'].shape[1])
    return inputs