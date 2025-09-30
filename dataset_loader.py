import os
from datasets import load_dataset
from PIL import Image
import numpy as np

def load_inst_it_dataset(
        dataset_local_path="Inst-IT-Dataset",
        dataset_config_name="inst_it_dataset_video_21k.json",
        dataset_split="train",
):
    """
    Loads the Inst-IT dataset from local files.

    It first loads the metadata from a local JSON file, then filters the dataset
    to include only the samples for which video frame directories exist locally.

    Args:
        dataset_local_path (str): The local base path for the video files and metadata.
        dataset_config_name (str): The name of the local JSON metadata file.
        dataset_split (str): The dataset split to use (e.g., 'train', 'test').

    Returns:
        datasets.Dataset: The filtered dataset with an added 'full_video_path' column.
    """
    metadata_path = os.path.join(dataset_local_path, dataset_config_name)
    print(f"Loading dataset metadata from local file '{metadata_path}' - split '{dataset_split}'...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata JSON file not found at: {metadata_path}. "
            f"Please ensure --dataset-local-path is set correctly and the file '{dataset_config_name}' exists there."
        )

    # Load metadata from local JSON file
    dataset = load_dataset('json', data_files=metadata_path, split=dataset_split)

    print("Filtering dataset based on locally available video directories...")

    # Create a new column for the full local path
    def add_full_path(sample):
        sample['full_video_path'] = os.path.join(dataset_local_path, sample['video_path'])
        return sample

    dataset = dataset.map(add_full_path)

    # Filter the dataset to keep only samples with existing video directories
    original_size = len(dataset)
    dataset = dataset.filter(lambda sample: os.path.isdir(sample['full_video_path']))
    filtered_size = len(dataset)

    print(f"Filtering complete. Found {filtered_size} local video directories out of {original_size} total samples in the metadata.")

    return dataset

def load_video_frames(video_path, num_frames=16):
    """
    Load a specified number of frames from a directory of video frames.
    The frames are expected to be in JPG format.

    Args:
        video_path (str): Path to the directory containing video frames.
        num_frames (int): This argument is ignored as we load all available frames.

    Returns:
        numpy.ndarray: An array of video frames.
    """
    # List all JPG files in the directory and sort them
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])

    if not frame_files:
        raise FileNotFoundError(f"No JPG frames found in directory: {video_path}")

    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_path, frame_file)
        with Image.open(frame_path) as img:
            frames.append(np.array(img))

    return np.array(frames)

def get_middle_frame_as_image(frames):
    """
    Get the middle frame from a sequence of video frames as an image.

    Args:
        frames (numpy.ndarray): An array of video frames.

    Returns:
        PIL.Image.Image: The middle frame as an image.
    """
    if len(frames) == 0:
        raise ValueError("No frames available to select the middle frame from.")

    # Select the middle frame
    middle_index = len(frames) // 2
    middle_frame = frames[middle_index]

    # Convert the frame (numpy array) to an image
    return Image.fromarray(middle_frame)

def preprocess_video_directory(video_dir, output_dir, target_size=(224, 224)):
    """
    Preprocess all videos in a directory and save the middle frame of each as an image.

    Args:
        video_dir (str): Path to the directory containing video frames.
        output_dir (str): Path to the directory where output images will be saved.
        target_size (tuple): The target size for the output images.
    """
    # Load video frames
    frames = load_video_frames(video_dir)

    # Get the middle frame as an image
    middle_frame_image = get_middle_frame_as_image(frames)

    # Resize the image
    resized_image = middle_frame_image.resize(target_size, Image.ANTIALIAS)

    # Save the image
    output_path = os.path.join(output_dir, "middle_frame.jpg")
    resized_image.save(output_path)

    print(f"Processed video in {video_dir}, saved middle frame to {output_path}")

def preprocess_videos_in_directory(video_directory, output_directory, target_size=(224, 224)):
    """
    Preprocess all videos in the given directory and save the middle frame of each video as an image.

    Args:
        video_directory (str): Path to the directory containing video frames.
        output_directory (str): Path to the directory where output images will be saved.
        target_size (tuple): The target size for the output images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all items in the video directory
    for item in os.listdir(video_directory):
        item_path = os.path.join(video_directory, item)

        # Check if the item is a directory (a video folder)
        if os.path.isdir(item_path):
            # Process the video directory
            preprocess_video_directory(item_path, output_directory, target_size)
        else:
            print(f"Skipped non-directory item: {item_path}")

def validate_dataset(args):
    """
    Validates the dataset loading process by loading the dataset metadata,
    filtering for local files, and attempting to load a few video samples.
    """
    print("--- Starting Dataset Validation ---")

    # Early exit if the base path doesn't exist
    if not os.path.isdir(args.dataset_local_path):
        print(f"\n❌ Validation Failed: The specified dataset local path does not exist or is not a directory.")
        print(f"  - Checked path: {os.path.abspath(args.dataset_local_path)}")
        print(f"  - Please ensure the '--dataset-local-path' argument points to the correct location.")
        return

    # Load the dataset using the loader function
    try:
        dataset = load_inst_it_dataset(
            dataset_local_path=args.dataset_local_path,
            dataset_config_name=args.dataset_config_name,
            dataset_split=args.dataset_split,
        )
    except FileNotFoundError as e:
        print(f"\n❌ Validation Failed: {e}")
        return


    if len(dataset) == 0:
        print("\n❌ Validation Failed: No local video files were found or matched.")
        print("Please check the following:")
        print(f"  - The local path exists: {args.dataset_local_path}")
        print(f"  - The dataset name and config are correct: {args.dataset_name}, {args.dataset_config_name}")
        return

    print(f"\n✅ Dataset loaded successfully with {len(dataset)} samples.")

    num_samples_to_check = min(args.num_samples_to_check, len(dataset))
    print(f"Attempting to load video frames for {num_samples_to_check} sample(s)...")

    for i in range(num_samples_to_check):
        sample = dataset[i]
        video_path = sample['full_video_path']
        print(f"\n--- Checking sample {i+1}/{num_samples_to_check} ---")
        print(f"  Video: {video_path}")

        try:
            # Try to load the video frames
            frames = load_video_frames(video_path)
            print(f"  ✅ Successfully loaded {len(frames)} frames.")

            # Check for question-answer pairs
            if 'question_answer_pairs' in sample and sample['question_answer_pairs']:
                print(f"  ✅ Found {len(sample['question_answer_pairs'])} question-answer pairs.")
            else:
                print("  ⚠️  Warning: No question-answer pairs found for this sample.")

        except Exception as e:
            print(f"  ❌ Failed to load video: {e}")

    print("\n--- Validation Complete ---")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Dataset loader and validator for the Inst-IT dataset.")
    parser.add_argument("--dataset-local-path", type=str, default="Inst-IT-Dataset", help="Local base path for the video files and metadata JSON.")
    parser.add_argument("--dataset-config-name", type=str, default="inst_it_dataset_video_21k.json", help="Name of the local JSON metadata file.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'test').")
    parser.add_argument("--num-samples-to-check", type=int, default=5, help="Number of video samples to try loading.")

    args = parser.parse_args()

    validate_dataset(args)
