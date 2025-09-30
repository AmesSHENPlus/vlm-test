import argparse
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
from PIL import Image
import os
from vllm import LLM, SamplingParams

from dataset_loader import load_inst_it_dataset, load_video_frames, get_middle_frame_as_image


def main(args):
    # Initialize vLLM
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
    )

    # Load data using the new dataset loader
    dataset = load_inst_it_dataset()

    total_inference_time = 0
    total_preprocessing_time = 0
    total_postprocessing_time = 0
    num_samples = 0
    correct_predictions = 0

    profiler_context = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir_vllm')
    ) if args.torch_profile else nullcontext()

    with profiler_context as prof:
        for i, sample in enumerate(dataset):
            if i >= args.num_samples:
                break

            # The dataset is now pre-filtered, so we can directly use the full path
            video_path = sample['full_video_path']
            question_answer_pairs = sample['question_answer_pairs']

            # --- Pre-processing (load video once) ---
            start_preprocessing = time.time()
            try:
                frames = load_video_frames(video_path, num_frames=args.num_frames)
                image = get_middle_frame_as_image(frames)
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                continue
            end_preprocessing = time.time()
            total_preprocessing_time += (end_preprocessing - start_preprocessing)

            for qa_pair in question_answer_pairs:
                with record_function(f"sample_{i}_qa") if args.torch_profile else nullcontext():
                    question = qa_pair['question']
                    ground_truth_answer = qa_pair.get('answer', '').lower()

                    # --- Model Inference with vLLM ---
                    with record_function("model_inference_vllm") if args.torch_profile else nullcontext():
                        start_inference = time.time()

                        prompt = "USER: <image>\n" + question + "\nASSISTANT:"

                        sampling_params = SamplingParams(max_tokens=100)

                        outputs = llm.generate({
                            "prompt": prompt,
                            "multi_modal_data": {"image": image},
                        }, sampling_params=sampling_params)

                        end_inference = time.time()
                        total_inference_time += (end_inference - start_inference)

                    # --- Post-processing ---
                    with record_function("postprocessing") if args.torch_profile else nullcontext():
                        start_postprocessing = time.time()
                        generated_text = outputs[0].outputs[0].text.strip().lower()
                        end_postprocessing = time.time()
                        total_postprocessing_time += (end_postprocessing - start_postprocessing)

                    # --- Accuracy Calculation ---
                    if ground_truth_answer and ground_truth_answer in generated_text:
                        correct_predictions += 1

                    print(f"Question: {question}")
                    print(f"Generated Answer: {generated_text}")
                    if ground_truth_answer:
                        print(f"Ground Truth: {ground_truth_answer}")
                    print("-" * 20)

                    num_samples += 1


    # --- Print Benchmark Results ---
    if num_samples > 0:
        avg_preprocessing_time = total_preprocessing_time / num_samples
        avg_inference_time = total_inference_time / num_samples
        avg_postprocessing_time = total_postprocessing_time / num_samples
        accuracy = (correct_predictions / num_samples) * 100 if 'answer' in dataset.column_names else -1

        print("\n--- Benchmark Results (vLLM) ---")
        print(f"Number of samples: {num_samples}")
        print(f"Average Pre-processing time: {avg_preprocessing_time:.4f}s")
        print(f"Average Model Inference time: {avg_inference_time:.4f}s")
        print(f"Average Post-processing time: {avg_postprocessing_time:.4f}s")
        print(f"Average Total time per sample: {(avg_preprocessing_time + avg_inference_time + avg_postprocessing_time):.4f}s")
        if accuracy != -1:
            print(f"Accuracy: {accuracy:.2f}%")

    if args.torch_profile:
        print("\n--- PyTorch Profiler Results ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        print("Trace file saved to ./log_dir_vllm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dataset-name", type=str, default="Inst-IT/Inst-It-Dataset")
    parser.add_argument("--dataset-config-name", type=str, default="1_video_21k")
    parser.add_argument("--dataset-local-path", type=str, default="/workspace/vlm-test/Inst-IT-Dataset")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--torch-profile", action="store_true", help="Enable PyTorch profiling.")
    args = parser.parse_args()
    main(args)
