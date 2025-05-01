"""
Performs OCR on selected image files to detect digits using EasyOCR.

Allows users to select image files via a dialog. For each image, it:
- Applies preprocessing optimized for low-resolution white text.
- Optionally runs an experimental mode to find the best preprocessing combination.
- Detects digits using EasyOCR.
- Filters detections based on a confidence threshold.
- Saves the recognized digits (one per line) to a text file.
- Saves cropped images of each detected digit.
- Saves a copy of the original image with bounding boxes and recognized digits annotated.
- Outputs results to an 'output' subdirectory within the script's directory.
- Displays progress using tqdm.
"""

import os
import tkinter as tk
from tkinter import filedialog
import itertools
import cv2
import easyocr
import numpy as np


def draw_bounding_box(image: np.ndarray, bbox: list, text: str, 
                      color: tuple = (0, 255, 0), thickness: int = 2):
    """Draws a bounding box and recognized text on an image.

    Args:
        image: The image canvas (NumPy array) to draw on.
        bbox: The bounding box coordinates from EasyOCR,
              formatted as [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]].
        text: The recognized text (digit) to display.
        color: The color of the bounding box and text (BGR). Defaults to green.
        thickness: The thickness of the bounding box lines and text. Defaults to 2.
    """
    p1 = (int(bbox[0][0]), int(bbox[0][1]))
    p2 = (int(bbox[2][0]), int(bbox[2][1]))
    cv2.rectangle(image, p1, p2, color, thickness)
    text_origin = (p1[0], max(0, p1[1] - 10)) # Place text above the box
    cv2.putText(image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, color, thickness)


def preprocess_for_white_text(image: np.ndarray, 
                              use_grayscale: bool = True, 
                              use_tophat: bool = True, 
                              use_upscale: bool = True, upscale_factor: int = 3,
                              use_normalize: bool = True, 
                              use_otsu: bool = True,
                              use_erosion: bool = False, erosion_iterations: int = 1,
                              use_morphology: bool = True,
                              use_dilation: bool = False, dilation_iterations: int = 1
                              ) -> tuple[np.ndarray, float]:
    """Applies a sequence of preprocessing steps optimized for OCR on white text.

    Args:
        image: Input image (NumPy array).
        use_grayscale: Convert to grayscale. Defaults to True.
        use_tophat: Apply top hat transformation. Defaults to True.
        use_upscale: Upscale the image. Defaults to True.
        upscale_factor: Factor for upscaling. Defaults to 3.
        use_normalize: Normalize the image pixel values. Defaults to True.
        use_otsu: Apply OTSU thresholding. Defaults to True.
        use_erosion: Apply erosion. Defaults to False.
        erosion_iterations: Number of erosion iterations. Defaults to 1.
        use_morphology: Apply morphological opening and closing. Defaults to True.
        use_dilation: Apply dilation after morphology. Defaults to False.
        dilation_iterations: Number of dilation iterations. Defaults to 1.

    Returns:
        A tuple containing the preprocessed image (NumPy array) and the scaling
        factor applied during upscaling (1.0 if no upscaling).
    """
    processed = image.copy()
    scale_factor = 1.0
    
    # Step 1: Grayscale
    if use_grayscale and len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Top-Hat
    if use_tophat:
        kernel_size = max(5, int(min(processed.shape[:2]) / 20))
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, kernel)
        processed = cv2.addWeighted(processed, 0.7, tophat, 1.3, 0)
            
    # Step 3: Upscale
    if use_upscale:
        h, w = processed.shape[:2]
        new_h, new_w = h * upscale_factor, w * upscale_factor
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        scale_factor = float(upscale_factor)
    
    # Step 4: Normalize
    if use_normalize:
        processed = cv2.normalize(processed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Step 5: OTSU Thresholding
    if use_otsu:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 6: Erosion
    if use_erosion:
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.erode(processed, kernel, iterations=erosion_iterations)
    
    # Step 7: Morphological Operations (Opening then Closing)
    if use_morphology:
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_large = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_small)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_large)

    # Step 7.5: Dilation (after Morphology)
    if use_dilation:
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=dilation_iterations)

    return processed, scale_factor


def calculate_score(results: list) -> tuple[float, int]:
    """Calculates a score based on OCR results, primarily average confidence.

    Args:
        results: A list of tuples from EasyOCR `readtext` 
                 (bbox, text, confidence).

    Returns:
        A tuple containing the average confidence score of detected digits 
        (or -1.0 if no results, 0.0 if no digits) and the number of digits found.
    """
    if not results:
        return -1.0, 0 # Penalize no results
    
    valid_results = [(bbox, text, prob) for bbox, text, prob in results if text.isdigit()]
    if not valid_results:
        return 0.0, 0 # No valid digits found

    total_confidence = sum(prob for _, _, prob in valid_results)
    avg_confidence = total_confidence / len(valid_results)
    num_digits = len(valid_results)
    
    return avg_confidence, num_digits


def generate_experimental_params(max_combinations: int = 50) -> list[dict]:
    """Generates preprocessing parameter combinations for experimental mode.

    Focuses on varying erosion, dilation, morphology, and OTSU, keeping
    grayscale, top-hat, upscale, and normalize enabled.

    Args:
        max_combinations: The maximum number of combinations to generate.

    Returns:
        A list of dictionaries, where each dictionary represents a set of 
        preprocessing parameters.
    """
    erosion_iters = [0, 1, 2]
    dilation_iters = [0, 1, 2] # Dilation after morphology
    morph_options = [False, True]
    otsu_options = [False, True] 
    
    fixed_params = {
        'use_grayscale': True,
        'use_tophat': True,
        'use_upscale': True, 'upscale_factor': 3,
        'use_normalize': True,
    }

    combinations = []
    param_product = itertools.product(otsu_options, erosion_iters, dilation_iters, morph_options)

    for otsu, erosion_iter, dilation_iter, morph in param_product:
        # Base case: Tophat, Upscale, Normalize (Otsu T/F)
        is_base_case = erosion_iter == 0 and dilation_iter == 0 and not morph
        if not is_base_case: # Add variations
             pass # Allow the base cases too
        
        params = fixed_params.copy()
        params.update({
            'use_otsu': otsu,
            'use_erosion': erosion_iter > 0, 'erosion_iterations': erosion_iter,
            'use_morphology': morph,
            'use_dilation': dilation_iter > 0, 'dilation_iterations': dilation_iter,
        })
        combinations.append(params)
        
        if len(combinations) >= max_combinations:
            break
            
    return combinations


def detect_number_from_frame(frame):
    """
    Main execution function. Sets up configuration, selects files, initializes OCR,
    runs processing (standard or experimental), and saves results.
    """
    ocrResult = None

    # --- Configuration ---
    CONFIDENCE_THRESHOLD = 0.0 # Minimum confidence score for detected digits

    # --- Mode Selection ---
    USE_EXPERIMENTAL_MODE = True # Master switch for experimental mode
    MAX_EXPERIMENTAL_COMBINATIONS = 50 # Limit for experimental mode

    # --- Standard Preprocessing Configuration (if USE_EXPERIMENTAL_MODE is False) ---
    USE_PREPROCESSING = True  # Master switch for standard preprocessing
    STANDARD_PREPROCESSING_PARAMS = {
        'use_grayscale': True,
        'use_tophat': True,
        'use_upscale': True, 'upscale_factor': 3,
        'use_normalize': True,
        'use_otsu': True,
        'use_erosion': False, 'erosion_iterations': 1,
        'use_morphology': True,
        'use_dilation': False, 'dilation_iterations': 1,
    }
    
    if not USE_EXPERIMENTAL_MODE:
        if USE_PREPROCESSING:
            for key, value in STANDARD_PREPROCESSING_PARAMS.items():
                print(f"{key}: {value}")
    else:
        print(f"  └─ Max Combinations: {MAX_EXPERIMENTAL_COMBINATIONS}")

    try:
        # Attempt GPU first, fallback to CPU
        reader = easyocr.Reader(['en'], gpu=True, recognizer='digit')
    except Exception:
        try:
            reader = easyocr.Reader(['en'], gpu=False, allowlist='0123456789')
        except Exception as e:
            print(f"Fatal Error initializing EasyOCR: {e}", flush=True)
            print("Please ensure EasyOCR and its dependencies (like PyTorch) are correctly installed.", flush=True)
            return

    # --- Generate Experimental Parameters (if needed) ---
    experimental_params_list = []
    if USE_EXPERIMENTAL_MODE:
        experimental_params_list = generate_experimental_params(MAX_EXPERIMENTAL_COMBINATIONS)


        try:
            results = []
            scale_factor = 1.0
            
            # --- Determine Best Preprocessing (Experimental or Standard) ---
            if USE_EXPERIMENTAL_MODE:
                best_results_exp = []
                best_score_exp = -1.0
                best_num_digits_exp = 0
                best_scale_factor_exp = 1.0
                best_params_exp = {}
                final_preprocessed_img = None

                # Iterate through generated parameter combinations
                for params in experimental_params_list:
                    preprocessed_img, current_scale_factor = preprocess_for_white_text(frame, **params)
                    current_results = reader.readtext(preprocessed_img, detail=1, paragraph=False, allowlist='0123456789')
                    current_score, current_num_digits = calculate_score(current_results)

                    # Update best score if current is better
                    if current_score > best_score_exp or \
                       (current_score == best_score_exp and current_num_digits > best_num_digits_exp):
                        best_score_exp = current_score
                        best_num_digits_exp = current_num_digits
                        best_results_exp = current_results
                        best_scale_factor_exp = current_scale_factor

                results = best_results_exp
                scale_factor = best_scale_factor_exp
                
            elif USE_PREPROCESSING: # Standard Preprocessing Mode
                preprocessed_img, scale_factor = preprocess_for_white_text(frame, **STANDARD_PREPROCESSING_PARAMS)
                results = reader.readtext(preprocessed_img, detail=1, paragraph=False, allowlist='0123456789')
                
                # Fallback to original if no results
                if not results:
                    scale_factor = 1.0
                    results = reader.readtext(frame, detail=1, paragraph=False, allowlist='0123456789')
            else: # No Preprocessing
                scale_factor = 1.0
                results = reader.readtext(frame, detail=1, paragraph=False, allowlist='0123456789')

            for (bbox, text, prob) in results: 
                ocrResult = text
        except Exception as e:
            print(f"An unexpected error occurred while processing: {e}")

    return ocrResult
