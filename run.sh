export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# Define parameters with comments
MODEL_TYPE="qwen2_5_vl"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="/workspace/vlm-test/Qwen2.5-VL-7B-Instruct"  # Path to base model\

TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
DATA_PATH="/workspace/vlm-test/VideoDetailCaption"  # Path to dataset

EVAL_NUM=2                        # Number of evaluation samples
MAX_NEW_TOKENS=256                # Number of new tokens to generate
DATA_NUM=100                      # Number of data samples to load
DROP_RATE=0.9                     # Pruning ratio
GPU_IDS="0,1,2,3"                 # GPU IDs to use

# A larger number of frames is generally recommended, as permitted by your GPU memory capacity and bandwidth. Memory bottlenecks are typically triggered by long visual sequence.
# Example:
#   - On NVIDIA A100 GPUs, we recommend using 128 frames for the LLaVA-OV 7B target
#     model and 64 frames for the 72B model.
#   - On NVIDIA H200 GPUs, we recommend 256 and 192 frames for the 7B and 72B models, respectively.
# Qwen2.5-VL currently does not support specifying input length directly. To control the input length, you will need to adjust the frame number accordingly.
FRAME_NUM=128

LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
    --model_type $MODEL_TYPE \
    --base_model_path $BASE_MODEL_PATH \
    --data_path $DATA_PATH \
    --task $TASK \
    --frame_num $FRAME_NUM \
    --evaluation_num $EVAL_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --drop_rate $DROP_RATE \
    --data_num $DATA_NUM \
    --gpu_ids $GPU_IDS \
    --save_path "results/${MODEL_TYPE}_${TASK}_drop_rate_${DROP_RATE}" \
    --setting "standard" \
    --trace_dir "log_dir_vllm" \
    --use_ncu \
    --use_pd_disagg \
    2>&1 | tee -a $LOG_FILE
