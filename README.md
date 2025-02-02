# MedShieldAI
MedShieldAI is an AI-powered tool that utilizes object detection and text recognition to extract and analyze textual information from medical documents.


## Features
- **Object Detection:** Uses YOLO to detect specific regions in documents.
- **Text Extraction:** Utilizes EasyOCR to extract text from detected regions.
- **Semantic Analysis:** Compares extracted text using a transformer-based language model.
- **Similarity Scoring:** Computes a similarity score between different textual components.

## Installation

1. Clone the repository:
   ```sh
   git clone [https://github.com/karmast/MedShieldAI.git](https://github.com/Karmast/MedShieldAI.git)
   cd MedShieldAI
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Download the YOLO model and place it in the `model/` directory.

## Usage

1. Place input images in the `input/` directory.
2. Run the main script:
   ```sh
   python main.py
   ```
3. Extracted data and results will be saved in the `output/` directory.

## Configuration
- Modify `model_path`, `output_dir`, and `image_dir` in the script to adjust directories.
- Adjust similarity thresholds as needed.

## Future Enhancements
- Improve text extraction accuracy.
- Enhance object detection for better precision.
- Optimize performance for large-scale document processing.

---

For any issues or contributions, feel free to open a discussion or pull request!

