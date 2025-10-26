import os
import argparse
import torch
import time
from datetime import datetime
from multiprocessing import Process, Event
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from torch.profiler import profile, record_function, ProfilerActivity
from utils import load_data, clip_input_video


def run_prefill(prefill_done, decode_done, model_path, prompt, multi_modal_data, trace_dir):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_parallel_size=2,
    )
    llm = LLM(
        model=model_path,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1},
        disable_log_stats=False,
    )

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
    ) as prof:
        llm.generate(prompt, sampling_params, multi_modal_data=multi_modal_data)

    print("Prefill node is finished.")

    if trace_dir:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"prefill_profile_trace_{timestamp}.json"
        trace_path = os.path.join(trace_dir, trace_filename)
        os.makedirs(trace_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"Prefill profiler trace saved to {trace_path}")

    prefill_done.set()
    print("Prefill node is waiting for decode to finish...")
    decode_done.wait()
    print("Prefill node is exiting.")


def run_decode(prefill_done, decode_done, model_path, prompt, sampling_params, trace_dir):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ktc = KVTransferConfig(
        kv_connector="P2pNcclConnector",
        kv_role="kv_consumer",
        kv_rank=2,
        kv_parallel_size=2,
    )
    llm = LLM(
        model=model_path,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"video": 1},
        disable_log_stats=False,
    )
    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
    ) as prof:
        torch.cuda.synchronize()
        tic = time.time()
        outputs = llm.generate(prompt, sampling_params)
        torch.cuda.synchronize()
        toc = time.time()

    decoding_time = toc - tic

    if trace_dir:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"decode_profile_trace_{timestamp}.json"
        trace_path = os.path.join(trace_dir, trace_filename)
        os.makedirs(trace_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"Decode profiler trace saved to {trace_path}")

    print("\n")
    print("-------Disaggregated Decoding with vLLM-------")
    print("Decoding Time:", decoding_time)

    output = outputs[0]
    input_token_length = len(output.prompt_token_ids)
    output_token_length = len(output.outputs[0].token_ids)
    print(f"Input token length: {input_token_length}")
    print(f"Output token length: {output_token_length}")

    output_text = outputs[0].outputs[0].text
    print("Output:")
    print(output_text)
    print("\n")
    decode_done.set()


def run_eval(model_type, llm, data_video, task, frame_num, evaluation_num, max_new_tokens, drop_rate, save_path=None, data_path=None, trace_dir=None, use_ncu=False, use_pd_disagg=False, base_model_path=None):
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
    ) as prof:
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
                    prompt, video_data = clip_input_video(task, data_instance, frame_num=frame_num, model_type=model_type, data_path=data_path)
                if prompt is None:
                    continue

                inputs = {
                    "prompt": prompt,
                    "multi_modal_data": {"video": video_data},
                }

                if use_pd_disagg:
                    prefill_done = Event()
                    decode_done = Event()
                    prefill_process = Process(target=run_prefill, args=(prefill_done, decode_done, base_model_path, inputs["prompt"], inputs["multi_modal_data"], trace_dir))
                    decode_process = Process(target=run_decode, args=(prefill_done, decode_done, base_model_path, inputs["prompt"], sampling_params, trace_dir))
                    prefill_process.start()
                    decode_process.start()
                    decode_process.join()
                    prefill_process.join()
                    continue

                torch.cuda.synchronize()
                tic = time.time()

                # Generate text from the prompts.
                if use_ncu:
                    torch.cuda.profiler.start()

                with record_function("llm.generate"):
                    outputs = llm.generate(inputs, sampling_params)

                if use_ncu:
                    torch.cuda.profiler.stop()

                torch.cuda.synchronize()
                toc = time.time()

                decoding_time = toc - tic

                for i, output in enumerate(outputs):
                    print(f"\n--- Result for Evaluation {i+1} ---")
                    print("-------Video QA with vLLM-------")
                    print(f"Eval Time for batch: {decoding_time}")

                    input_token_length = len(output.prompt_token_ids)
                    output_token_length = len(output.outputs[0].token_ids)
                    print(f"Input token length: {input_token_length}")
                    print(f"Output token length: {output_token_length}")

                    output_text = output.outputs[0].text
                    print("Output:")
                    print(output_text)
                    print("\n")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    if trace_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_filename = f"profile_trace_{timestamp}.json"
        trace_path = os.path.join(trace_dir, trace_filename)

        os.makedirs(trace_dir, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"Profiler trace saved to {trace_path}")


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
    parser.add_argument('--trace_dir', type=str, default='.',
                        help='Directory to save the profiler trace.')
    parser.add_argument('--use_ncu', action='store_true',
                        help='Enable NCU profiling markers.')
    parser.add_argument('--use_pd_disagg', action='store_true',
                        help='Use prefill/decode disaggregation.')


    # Parse command line arguments
    args = parser.parse_args()

    # Set GPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # Initialize the vLLM engine.
    llm = None
    if not args.use_pd_disagg:
        print("--- Initializing vLLM Engine ---")
        llm = LLM(
            model=args.base_model_path,
            enforce_eager=True, # Enforce eager execution for multi-modal models
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1},
        )
        print("--- vLLM Engine Initialized ---")
    # else:
    #     print("--- Initializing vLLM Engine for PD Disagg ---")
    #     llm = LLM(
    #         model=args.base_model_path,
    #         tensor_parallel_size=2,
    #         gpu_memory_utilization=0.85,
    #         enable_chunked_prefill=True,
    #         kv_cache_dtype="fp8",
    #         enforce_eager=True, # Enforce eager execution for multi-modal models
    #         trust_remote_code=True,
    #         limit_mm_per_prompt={"video": 1},
    #     )
    #     print("--- vLLM Engine Initialized for PD Disagg ---")

    # Load data
    data_video = load_data(args.task, args.data_num, args.data_path)

    # Set save path
    if args.save_path is None:
        save_path = f"results/{args.model_type}_{args.task}_{args.drop_rate}"
    else:
        save_path = args.save_path

    # Wrap the evaluation with the profiler
    run_eval(
        args.model_type,
        llm=llm,
        data_video=data_video,
        task=args.task,
        frame_num=args.frame_num,
        evaluation_num=args.evaluation_num,
        max_new_tokens=args.max_new_tokens,
        drop_rate=args.drop_rate,
        save_path=save_path,
        data_path=args.data_path,
        trace_dir=args.trace_dir,
        use_ncu=args.use_ncu,
        use_pd_disagg=args.use_pd_disagg,
        base_model_path=args.base_model_path,
    )
