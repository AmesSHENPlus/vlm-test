import os
import argparse
import torch
import time
from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.profiler import profile, record_function, ProfilerActivity
from utils import load_model, load_data, clip_input_video


def run_eval(model_type, llm, data_video, task, frame_num, evaluation_num, max_new_tokens, drop_rate, video_token_id, save_path=None, data_path=None, processor=None):
    # Run evaluation
    results = {}
    results['ar_two_stage_decode'] = []

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=max_new_tokens,
    )

    for i in tqdm(range(evaluation_num)):
        with record_function("data_iteration"):
            data_instance = data_video[i]

            # Prepare model inputs
            with record_function("clip_input_video"):
                prompt, video_data = clip_input_video(processor, task, data_instance, frame_num=frame_num, model_type=model_type, data_path=data_path)
            if prompt is None:
                continue

            torch.cuda.synchronize()
            tic = time.time()

            # Generate text from the prompts.
            with record_function("llm.generate"):
                outputs = llm.generate(prompt, sampling_params, multi_modal_data=video_data)

            torch.cuda.synchronize()
            toc = time.time()

            decoding_time = toc - tic

            print("\n")
            print("-------Autoregressive Decoding with vLLM-------")
            print("Decoding Time:", decoding_time)
            output_text = outputs[0].outputs[0].text
            print("Output:")
            print(output_text)
            print("\n")
            results['ar_two_stage_decode'].append(decoding_time)


if __name__ == "__main__":
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Run video model evaluation')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='qwen2_5_vl',
                        choices=['llava_ov', 'qwen2_5_vl'],
                        help='Model type: llava_ov or qwen2_5_vl')
    parser.add_argument('--base_model_path', type=str,
                        default=None,
                        help='Path to the base model')

    parser.add_argument('--data_path', type=str,
                        default='/data',
                        help='Path to the data directory')

    # Evaluation parameters
    parser.add_argument('--task', type=str, default='VideoDetailCaption',
                        choices=['VideoDetailCaption', 'MVBench', 'MVLU', 'LongVideoBench', 'MMBench'],
                        help='Evaluation task type')
    parser.add_argument('--frame_num', type=int, default=8,
                        help='Number of frames per video')
    parser.add_argument('--evaluation_num', type=int, default=1,
                        help='Number of evaluation samples')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--drop_rate', type=float, default=0.9,
                        help='Pruning rate')
    parser.add_argument('--data_num', type=int, default=100,
                        help='Number of data samples to load')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save results. If not specified, a default path will be used')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3,4,5",
                        help='GPU IDs to use')
    parser.add_argument('--setting', type=str, default='standard',
                        choices=['self', 'standard'],
                        help='Speculative Decoding setting')


    # Parse command line arguments
    args = parser.parse_args()

    # Set GPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Initialize the vLLM engine.
    print("--- Initializing vLLM Engine ---")
    llm = LLM(
        model=args.base_model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True, # Enforce eager execution for multi-modal models
        trust_remote_code=True,
    )
    print("--- vLLM Engine Initialized ---")
    # Load models
    _, processor, video_token_id = load_model(args.base_model_path)


    # Load data
    data_video = load_data(args.task, args.data_num, args.data_path)

    # Set save path
    if args.save_path is None:
        save_path = f"results/{args.model_type}_{args.task}_{args.drop_rate}"
    else:
        save_path = args.save_path

    # Wrap the evaluation with the profiler
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir_vllm')
    ) as prof:
        run_eval(
            args.model_type,
            llm=llm,
            data_video=data_video,
            task=args.task,
            frame_num=args.frame_num,
            evaluation_num=args.evaluation_num,
            max_new_tokens=args.max_new_tokens,
            drop_rate=args.drop_rate,
            video_token_id=video_token_id,
            save_path=save_path,
            data_path=args.data_path,
            processor=processor,
        )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("profile_trace.json")
    print("Profiler trace saved to profile_trace.json")
