import multiprocessing
import os
import shutil
from application import create_app
from transformers import AutoProcessor, TextIteratorStreamer, Qwen2_5_VLForConditionalGeneration
import torch

loaded_models_cache = {}

# --- Model Configurations ---
MODEL_CONFIGS = [
    { "id": "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 3B (Instruct, 4-bit Unsloth)" },
    { "id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 7B (Instruct, 4-bit Unsloth)" },
    { "id": "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 32B (Instruct, 4-bit Unsloth)" }
]
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS[1]
DEFAULT_MODEL_ID = DEFAULT_MODEL_CONFIG["id"]

# --- Performance & Loading Defaults ---
DEFAULT_LOAD_PARAMS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": os.getenv("DEFAULT_BNB_4BIT_QUANT_TYPE", "nf4"),
    "bnb_4bit_use_double_quant": os.getenv("DEFAULT_BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == 'true',
    "bnb_4bit_compute_dtype_str": os.getenv("DEFAULT_BNB_4BIT_COMPUTE_DTYPE", "auto"),
    "llm_int8_enable_fp32_cpu_offload": os.getenv("DEFAULT_LLM_INT8_ENABLE_FP32_CPU_OFFLOAD", "false").lower() == 'true',
}

def recreate_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        
def get_dtype_from_string(dtype_str: str):
    if dtype_str == "bfloat16": return torch.bfloat16
    if dtype_str == "float16": return torch.float16
    if dtype_str == "float32": return torch.float32
    return "auto"

def get_model_and_processor(model_id: str, load_in_4bit: bool, bnb_4bit_quant_type: str, bnb_4bit_use_double_quant: bool, bnb_4bit_compute_dtype_str: str, llm_int8_enable_fp32_cpu_offload: bool):
    param_tuple = (model_id, f"4bit-{load_in_4bit}", f"quant-{bnb_4bit_quant_type}", f"doubleq-{bnb_4bit_use_double_quant}", f"compute-{bnb_4bit_compute_dtype_str}", f"offload-{llm_int8_enable_fp32_cpu_offload}")
    cache_key = "_".join(param_tuple)
    if cache_key in loaded_models_cache: return loaded_models_cache[cache_key]

    print(f"Initiating load for model '{model_id}'. Cache key: {cache_key}")
    model_kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        actual_compute_dtype = get_dtype_from_string(bnb_4bit_compute_dtype_str)
        if actual_compute_dtype == "auto":
            actual_compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model_kwargs.update({
            "load_in_4bit": load_in_4bit,
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": actual_compute_dtype
        })
        if llm_int8_enable_fp32_cpu_offload:
            model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    else:
        print(f"CUDA NOT available. Model '{model_id}' will be loaded on CPU.")
        model_kwargs.update({"device_map": "cpu", "torch_dtype": torch.float32})
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        loaded_models_cache[cache_key] = (model, processor)
        print(f"Successfully loaded model and processor for: {model_id}")
        return model, processor
    except Exception as e:
        print(f"Error loading model '{model_id}': {e}")
        if cache_key in loaded_models_cache: del loaded_models_cache[cache_key]
        raise e


def preload_models_on_startup():
    if True:
        print(f"Attempting to preload default model: {DEFAULT_MODEL_ID}...")
        try:
            get_model_and_processor(DEFAULT_MODEL_ID, **DEFAULT_LOAD_PARAMS)
        except Exception as e:
            print(f"Failed to preload default model {DEFAULT_MODEL_ID}: {e}")

app = create_app()

if __name__ == '__main__':
    print("Starting Flask application...")
    multiprocessing.set_start_method('spawn', force=True)

    preload_models_on_startup()
    print("Starting Qwen-VL Flask API Server...")

    # Directories to recreate
    directories_to_recreate = ['frames', 'frames_processed']
    recreate_directories(directories_to_recreate)

    app.run(debug=True, host='0.0.0.0', port=8000)