# OCR Digits with EasyOCR

This script performs Optical Character Recognition (OCR) on selected image files to detect and extract digits using the EasyOCR library. It includes preprocessing steps optimized for low-resolution images with white text and an optional experimental mode to find the best preprocessing parameters.

## Features

*   **File Selection:** Uses a graphical file dialog to select one or multiple image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`).
*   **Preprocessing:**
    *   Applies a standard preprocessing pipeline optimized for low-resolution white text (Grayscale, Top-Hat, Upscaling, Normalization, OTSU Thresholding, Morphological Operations).
    *   Optionally disables standard preprocessing.
*   **Experimental Mode:**
    *   If enabled, automatically tests multiple combinations of preprocessing parameters (varying OTSU, Erosion, Dilation, Morphology) to find the set that yields the best OCR results (based on average confidence and number of digits).
    *   Saves the best-performing preprocessed image.
*   **OCR Engine:** Uses EasyOCR for digit recognition. Attempts to use GPU acceleration if available, falling back to CPU.
*   **Filtering:** Filters detected results based on a configurable confidence threshold and ensures only digits are processed.
*   **Output Generation:** For each processed image, it saves:
    *   A text file (`*_digits.txt`) containing recognized digits, one per line.
    *   Cropped images (`*_digit_N.png`) of each detected digit.
    *   A copy of the original image (`*_annotated.png`) with bounding boxes and recognized digits drawn on it.
    *   The preprocessed image used for OCR (`*_preprocessed.png` or `*_preprocessed_best.png`).
*   **Organized Output:** Saves all generated files into an `output` subdirectory within the script's location.
*   **Progress Indication:** Uses `tqdm` to display a progress bar during image processing.

## Requirements

*   Python 3.x
*   Required Python libraries:
    *   `easyocr`
    *   `opencv-python` (cv2)
    *   `numpy`
    *   `tqdm`
    *   `tk` (usually included with standard Python installations)

*   **EasyOCR Dependencies:** EasyOCR relies on PyTorch. Ensure you have a compatible version of PyTorch installed. Installation might differ based on your system and whether you have a CUDA-enabled GPU. Refer to the official [PyTorch](https://pytorch.org/get-started/locally/) and [EasyOCR](https://github.com/JaidedAI/EasyOCR) documentation for detailed installation instructions.

## Installation

1.  **Clone or download the script.**
2.  **Install Python dependencies:**
    ```bash
    pip install easyocr opencv-python numpy tqdm
    ```

## Usage

1.  **Navigate** to the directory containing `main.py` in your terminal.
2.  **Run the script:**
    ```bash
    python main.py
    ```
3.  A **file dialog** will appear. Select the image files you want to process. You can select multiple files.
4.  The script will process each selected image, displaying progress using `tqdm`.
5.  Output files will be saved in the `output` subdirectory.

## Configuration

You can modify the script's behavior by changing the configuration variables at the beginning of the `main()` function in `main.py`:

*   `CONFIDENCE_THRESHOLD`: (Float, 0.0 to 1.0) Minimum confidence score required for a detected digit to be considered valid. Default: `0.0` (accepts all).
*   `OUTPUT_SUBDIR`: (String) Name of the subdirectory where output files will be saved. Default: `"output"`.
*   `USE_EXPERIMENTAL_MODE`: (Boolean) Set to `True` to enable experimental mode, `False` to use standard preprocessing or no preprocessing. Default: `True`.
*   `MAX_EXPERIMENTAL_COMBINATIONS`: (Integer) Maximum number of preprocessing combinations to test in experimental mode. Default: `50`.
*   `USE_PREPROCESSING`: (Boolean) Master switch for standard preprocessing. Only effective if `USE_EXPERIMENTAL_MODE` is `False`. Default: `True`.
*   `STANDARD_PREPROCESSING_PARAMS`: (Dictionary) Contains boolean flags and parameters for each step in the standard preprocessing pipeline. Only used if `USE_EXPERIMENTAL_MODE` is `False` and `USE_PREPROCESSING` is `True`.

## Experimental Mode

When `USE_EXPERIMENTAL_MODE` is set to `True`, the script ignores the `USE_PREPROCESSING` and `STANDARD_PREPROCESSING_PARAMS` settings. Instead, it generates multiple variations of preprocessing parameters (primarily focusing on `use_otsu`, `use_erosion`, `use_dilation`, `use_morphology`) up to `MAX_EXPERIMENTAL_COMBINATIONS`.

For each image, it applies every generated parameter set, runs OCR, and calculates a score based on the average confidence and number of digits found. The results (bounding boxes, text file, annotated image, cropped digits) corresponding to the parameter set with the highest score are saved. The specific best-performing preprocessed image is also saved as `*_preprocessed_best.png`.

This mode is useful for finding optimal settings for a specific type of image but takes significantly longer to run.

## Output Files

For an input image named `example.png`, the following files might be generated in the `output` directory:

*   `example_digits.txt`: Contains the extracted digits, one per line.
*   `example_annotated.png`: A copy of `example.png` with bounding boxes and recognized digits drawn.
*   `example_7_1.png`, `example_3_1.png`, etc.: Cropped images of each detected digit (e.g., the first detected '7', the first detected '3').
*   `example_preprocessed.png`: The image after applying the standard preprocessing pipeline (if standard mode is used).
*   `example_preprocessed_best.png`: The image after applying the best-performing preprocessing pipeline found during experimental mode (if experimental mode is used).

## Recommendations & Notes

*   **Preprocessing Tuning:** The default standard preprocessing parameters are optimized for low-resolution images with white text on a darker background. You may need to adjust `STANDARD_PREPROCESSING_PARAMS` or rely on `USE_EXPERIMENTAL_MODE` if your images have different characteristics (e.g., dark text on a light background, different resolutions, noise levels).
*   **Performance:** OCR processing, especially in experimental mode, can be computationally intensive. Using a CUDA-enabled GPU significantly speeds up EasyOCR.
*   **Confidence Threshold:** Adjust `CONFIDENCE_THRESHOLD` based on the quality of your images and the desired accuracy. A higher threshold reduces false positives but might miss some valid digits.
*   **Allowlist:** The script currently uses `allowlist='0123456789'` when initializing EasyOCR (or the 'digit' recognizer which implies digits), focusing solely on digits.
*   **Error Handling:** The script includes basic error handling for file reading and directory creation, but complex image issues or EasyOCR errors might require further investigation.
