# --- Flask API Imports ---
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# --- Original Core Logic Imports (Now used by Flask) ---
import logging
import re
from PIL import Image
import requests
from io import BytesIO
import base64
import os
import uuid
import json
import time
from threading import Thread, Lock
import csv
import datetime
import torch
from transformers import AutoProcessor, TextIteratorStreamer, Qwen2_5_VLForConditionalGeneration
import numpy # For qwen_model_handler

# --- Global Configuration ---
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO").upper()

# --- Model Configurations ---
MODEL_CONFIGS = [
    { "id": "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 3B (Instruct, 4-bit Unsloth)" },
    { "id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 7B (Instruct, 4-bit Unsloth)" },
    { "id": "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 32B (Instruct, 4-bit Unsloth)" }
]
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS[1]
DEFAULT_MODEL_DISPLAY_NAME = DEFAULT_MODEL_CONFIG["display_name"]
MODEL_DISPLAY_NAME_TO_ID_MAP = {config["display_name"]: config["id"] for config in MODEL_CONFIGS}
DEFAULT_LOAD_PARAMS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": os.getenv("DEFAULT_BNB_4BIT_QUANT_TYPE", "nf4"),
    "bnb_4bit_use_double_quant": os.getenv("DEFAULT_BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == 'true',
    "bnb_4bit_compute_dtype_str": os.getenv("DEFAULT_BNB_4BIT_COMPUTE_DTYPE", "auto"),
    "llm_int8_enable_fp32_cpu_offload": os.getenv("DEFAULT_LLM_INT8_ENABLE_FP32_CPU_OFFLOAD", "false").lower() == 'true',
}
# --- Model Configurations ---
MODEL_CONFIGS = [
    { "id": "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 3B (Instruct, 4-bit Unsloth)" },
    { "id": "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 7B (Instruct, 4-bit Unsloth)" },
    { "id": "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit", "display_name": "Qwen2.5-VL 32B (Instruct, 4-bit Unsloth)" }
]
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS[1]
DEFAULT_MODEL_ID = DEFAULT_MODEL_CONFIG["id"]
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.1))

# --- Core Setup ---
logging.basicConfig(level=getattr(logging, LOGGING_LEVEL, logging.INFO), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flask_qwen_vl_app")
LOGS_DIRECTORY = os.path.join(os.getcwd(), 'logs')
TEMP_IMAGE_DIRECTORY = os.path.join(LOGS_DIRECTORY, 'temp_images')
os.makedirs(TEMP_IMAGE_DIRECTORY, exist_ok=True)
USAGE_LOG_FILE = os.path.join(LOGS_DIRECTORY, 'usage_log_flask.csv')
USAGE_LOG_HEADERS = ["timestamp", "user_identifier", "request_uuid", "model_id", "temperature", "system_prompt_key_used", "prompt_text", "image_provided", "image_details", "ai_response_length", "ai_response_preview", "ttft_ms", "generation_time_ms", "total_stream_handler_time_ms", "error_message", "load_params_used"]
usage_log_lock = Lock()
loaded_models_cache = {}

def _load_image_from_url(url):
    try:
        if url.startswith('http'):
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        elif os.path.exists(url):
            img = Image.open(url)
        else: return None
        return img.convert('RGB') if img.mode != 'RGB' else img
    except Exception as e:
        logger.error(f"Error loading image {url[:60]}...: {e}")
        return None

def extract_pil_images_from_messages(messages):
    pil_images = []
    if not messages: return pil_images
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image_url":
                    url_data = item.get("image_url", {})
                    url = url_data.get("url")
                    if url and (img := _load_image_from_url(url)):
                        pil_images.append(img)
    return pil_images

DEFAULT_SYSTEM_PROMPTS = {
    "ocr_digit_only": "You are an expert Optical Character Recognition (OCR) assistant. Your sole task is to meticulously extract ONLY THE DIGITS (0-9) visible in the provided image. Present these digits clearly, for example, separated by spaces or newlines. If no digits are found, explicitly state 'No digits found'. Do not provide any other text, explanation, or commentary. If no image is provided, state 'Please provide an image for digit extraction.'",
    "ocr_general": "You are an expert OCR assistant. Your primary task is to meticulously extract ALL text, numbers, and symbols visible in any provided image or described scene. Transcribe the text exactly as it appears. Only output the extracted text. If no image is clearly referenced or uploaded, state that you need an image or image URL to perform OCR.",
    "ocr_receipt": "You are an expert OCR assistant specializing in receipts. Extract all items, quantities, and prices. Also identify the store name, date, and total amount. Present the information in a structured format if possible.",
    "chat_general_helper": "You are a helpful AI assistant. Analyze the provided image and respond to the user's query."
}
PRIMARY_DEFAULT_SYSTEM_PROMPT_KEY = "ocr_digit_only"

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
        print(f"Error loading model '{model_id}': {e}", exc_info=True)
        if cache_key in loaded_models_cache: del loaded_models_cache[cache_key]
        raise e


def generate_chat_response_blocking(model_id_param: str, messages_for_model, temperature, **load_params):
    model, processor = get_model_and_processor(model_id_param, **load_params)
    pil_images = extract_pil_images_from_messages(messages_for_model)
    text_prompt = processor.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=pil_images or None, return_tensors="pt").to(model.device)
    
    generation_kwargs = dict(inputs, max_new_tokens=2048, temperature=max(temperature, 0.01))
    outputs = model.generate(**generation_kwargs)
    
    # Decode only the newly generated tokens, excluding the prompt
    generated_tokens = outputs[:, inputs.input_ids.shape[-1]:]
    response_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return response_text

def generate_chat_response_stream(model_id_param: str, messages_for_model, temperature, **load_params):
    model, processor = get_model_and_processor(model_id_param, **load_params)
    pil_images = extract_pil_images_from_messages(messages_for_model)
    text_prompt = processor.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=pil_images or None, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2048, temperature=max(temperature, 0.01))
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    
    buffer = ""
    for chunk in streamer:
        if chunk: buffer += chunk
        if ' ' in buffer or '\n' in buffer or len(buffer) > 5:
            yield buffer
            buffer = ""
    if buffer: yield buffer

def prepare_qwen_messages_for_model(current_message_dict, system_prompt_text, request_uuid, system_prompt_key):
    qwen_messages = [{"role": "system", "content": system_prompt_text}]
    logged_image_path = None
    
    current_user_prompt_text = current_message_dict.get("text", "")
    current_qwen_user_content_parts = []
    image_part_added = False

    if current_message_dict.get("files"):
        temp_image_path = current_message_dict["files"][0]
        current_qwen_user_content_parts.append({"type": "image_url", "image_url": {"url": temp_image_path}})
        image_part_added = True
        if pil_image_to_log := _load_image_from_url(temp_image_path):
            logged_image_path = os.path.join(LOGS_DIRECTORY, f"log_img_{request_uuid}.png")
            pil_image_to_log.save(logged_image_path)

    if current_user_prompt_text.strip():
        current_qwen_user_content_parts.append({"type": "text", "text": current_user_prompt_text.strip()})
    elif image_part_added:
        default_text = "ocr digit" if system_prompt_key == "ocr_digit_only" else "Describe the image."
        current_qwen_user_content_parts.append({"type": "text", "text": default_text})

    if current_qwen_user_content_parts:
        qwen_messages.append({"role": "user", "content": current_qwen_user_content_parts})
    
    return qwen_messages, logged_image_path

def process_chat_request_blocking(**args):
    request_uuid = str(uuid.uuid4())[:8]
    actual_model_id = MODEL_DISPLAY_NAME_TO_ID_MAP.get(args['model_display_name'], DEFAULT_MODEL_ID)
    system_prompt_text = DEFAULT_SYSTEM_PROMPTS.get(args['system_prompt_key'], DEFAULT_SYSTEM_PROMPTS[PRIMARY_DEFAULT_SYSTEM_PROMPT_KEY])

    if not args['current_message'].get("text", "").strip() and not args['current_message'].get("files", []):
        return {"error": "Please provide some input or an image."}, 400

    messages_for_model, logged_image_path = prepare_qwen_messages_for_model(args['current_message'], system_prompt_text, request_uuid, args['system_prompt_key'])
    if len(messages_for_model) < 2:
        return {"error": "Could not form a valid message to send."}, 400

    load_params = {k: v for k, v in args.items() if k in DEFAULT_LOAD_PARAMS}
    user_prompt_text_for_log = next((item["text"] for item in messages_for_model[-1]["content"] if item["type"] == "text"), "")
    log_payload = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "user_identifier": "api_user", "request_uuid": request_uuid, "model_id": actual_model_id, "temperature": args['temperature'], "system_prompt_key_used": args['system_prompt_key'], "prompt_text": user_prompt_text_for_log, "image_provided": bool(logged_image_path), "image_details": logged_image_path or "none", "load_params_used": json.dumps(load_params)}
    
    full_ai_response = ""
    start_time = time.monotonic()
    try:
        full_ai_response = generate_chat_response_blocking(
            model_id_param=actual_model_id,
            messages_for_model=messages_for_model,
            temperature=args['temperature'],
            **load_params
        )
        end_time = time.monotonic()
        total_time_ms = round((end_time - start_time) * 1000)
        
        log_payload.update({"ai_response_length": len(full_ai_response), "ai_response_preview": full_ai_response[:200], "ttft_ms": total_time_ms, "generation_time_ms": total_time_ms, "total_stream_handler_time_ms": total_time_ms, "error_message": ""})
        
        response_data = {"content": full_ai_response, "request_uuid": request_uuid, "model_id": actual_model_id}
        return response_data, 200

    except Exception as e:
        logger.error(f"Error during blocking model generation for {request_uuid}: {e}", exc_info=True)
        total_time_ms = round((time.monotonic() - start_time) * 1000)
        log_payload.update({"error_message": str(e)[:200], "total_stream_handler_time_ms": total_time_ms})
        return {"error": f"Error during model generation: {str(e)[:200]}", "request_uuid": request_uuid}, 500
    finally:
        print(**log_payload)

def handle_blocking_chat_request():
    def get_form_val(key, default, converter):
            val = request.form.get(key)
            return converter(val) if val is not None else default

    load_params = {
        "load_in_4bit": get_form_val("load_in_4bit", DEFAULT_LOAD_PARAMS["load_in_4bit"], lambda v: v.lower() == 'true'),
        "bnb_4bit_quant_type": request.form.get("bnb_4bit_quant_type", DEFAULT_LOAD_PARAMS["bnb_4bit_quant_type"]),
        "bnb_4bit_use_double_quant": get_form_val("bnb_4bit_use_double_quant", DEFAULT_LOAD_PARAMS["bnb_4bit_use_double_quant"], lambda v: v.lower() == 'true'),
        "bnb_4bit_compute_dtype_str": request.form.get("bnb_4bit_compute_dtype_str", DEFAULT_LOAD_PARAMS["bnb_4bit_compute_dtype_str"]),
        "llm_int8_enable_fp32_cpu_offload": get_form_val("llm_int8_enable_fp32_cpu_offload", DEFAULT_LOAD_PARAMS["llm_int8_enable_fp32_cpu_offload"], lambda v: v.lower() == 'true')
    }
    args = {
        "model_display_name": request.form.get("model_display_name", DEFAULT_MODEL_DISPLAY_NAME),
        "system_prompt_key": request.form.get("system_prompt_key", PRIMARY_DEFAULT_SYSTEM_PROMPT_KEY),
        "temperature": get_form_val("temperature", DEFAULT_TEMPERATURE, float),
        **load_params
    }
    response_data, status_code = process_chat_request_blocking(**args)
    return jsonify(response_data), status_code

def test_endpoint_handle_blocking_chat_request(frame):
    def get_form_val(key, default, converter):
            val = request.form.get(key)
            return converter(val) if val is not None else default

    load_params = {
        "load_in_4bit": get_form_val("load_in_4bit", DEFAULT_LOAD_PARAMS["load_in_4bit"], lambda v: v.lower() == 'true'),
        "bnb_4bit_quant_type": request.form.get("bnb_4bit_quant_type", DEFAULT_LOAD_PARAMS["bnb_4bit_quant_type"]),
        "bnb_4bit_use_double_quant": get_form_val("bnb_4bit_use_double_quant", DEFAULT_LOAD_PARAMS["bnb_4bit_use_double_quant"], lambda v: v.lower() == 'true'),
        "bnb_4bit_compute_dtype_str": request.form.get("bnb_4bit_compute_dtype_str", DEFAULT_LOAD_PARAMS["bnb_4bit_compute_dtype_str"]),
        "llm_int8_enable_fp32_cpu_offload": get_form_val("llm_int8_enable_fp32_cpu_offload", DEFAULT_LOAD_PARAMS["llm_int8_enable_fp32_cpu_offload"], lambda v: v.lower() == 'true')
    }
    args = {
        "current_message": {"text": request.form.get("message", ""), "files": frame},
        "model_display_name": request.form.get("model_display_name", DEFAULT_MODEL_DISPLAY_NAME),
        "system_prompt_key": request.form.get("system_prompt_key", PRIMARY_DEFAULT_SYSTEM_PROMPT_KEY),
        "temperature": get_form_val("temperature", DEFAULT_TEMPERATURE, float),
        **load_params
    }
    response_data, status_code = process_chat_request_blocking(**args)
    return jsonify(response_data), status_code