import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import easyocr
import json
import numpy as np
import re
from difflib import SequenceMatcher
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class StampDetector:
    def __init__(self, model_path, output_dir):
        # Initialize the YOLO model
        self.model = YOLO(model_path)
        # Create the output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['fa'])  # Persian language

        # Initialize ParsBERT for semantic similarity
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model_bert = BertModel.from_pretrained("bert-base-multilingual-cased")

    def detect_and_save_stamps(self, image_dir):
        """Detect stamps in all images in the directory, save them, and extract text."""
        results_dict = {}  # Dictionary to store results

        # Loop through the images in the input directory
        for image_filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_filename)
            if image_path.endswith(('.jpg', '.png')):  # Process only image files
                # Run YOLO inference
                results = self.model(image_path)

                # Load the original image
                image = cv2.imread(image_path)

                image_results = []  # List to store results for the current image
                stamps_boxes = []  # List to store bounding boxes of detected stamps

                for i, result in enumerate(results):
                    for box in result.boxes.xyxy:  # Extract bounding box coordinates
                        x_min, y_min, x_max, y_max = map(int, box)  # Convert to integers

                        # Add margin to bounding box for better cropping
                        margin = 10
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(image.shape[1], x_max + margin)
                        y_max = min(image.shape[0], y_max + margin)

                        # Store the bounding boxes for further use
                        stamps_boxes.append((x_min, y_min, x_max, y_max))

                        # Crop the detected stamp
                        cropped = image[y_min:y_max, x_min:x_max]

                        # Use EasyOCR to extract text from the cropped stamp
                        text_easyocr = self.reader.readtext(cropped, detail=0)
                        text_easyocr_combined = " ".join(text_easyocr)  # Combine text without newlines

                        # Save cropped stamp
                        output_path = self.output_dir / f"stamp_{i}.jpg"
                        cv2.imwrite(str(output_path), cropped)

                        # Add the extracted text to the image results
                        image_results.append({
                            'stamp_id': i,
                            'text': text_easyocr_combined
                        })

                # Store the results for this image in the dictionary
                results_dict[image_filename] = image_results

                # Extract text excluding stamp regions
                prescription_text = self.extract_prescription_text(image, stamps_boxes)
                image_results.append({
                    'prescription_text': prescription_text
                })

                # Compare text from stamp with prescription text
                for result in image_results:
                    if 'stamp_id' in result:
                        stamp_text = result['text']
                        similarity_percentage, confidence = self.compare_texts(stamp_text, prescription_text)
                        result['similarity_percentage'] = similarity_percentage
                        result['confidence'] = confidence

        # Save results as a JSON file
        json_output_path = self.output_dir / "extracted_text.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {json_output_path}")

    def extract_prescription_text(self, image, stamps_boxes, max_length=500):
        """Extract text from the prescription excluding the stamp regions."""
        # Create a mask that excludes the regions of the stamps
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255  # White mask

        # Set stamp areas to black (i.e., exclude them from OCR)
        for (x_min, y_min, x_max, y_max) in stamps_boxes:
            mask[y_min:y_max, x_min:x_max] = 0  # Set stamp areas to 0 (black)

        # Apply the mask to the original image (white areas are kept, black is excluded)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Use EasyOCR to extract text from the remaining part of the image
        text_easyocr = self.reader.readtext(masked_image, detail=0)
        text_easyocr_combined = " ".join(text_easyocr)  # Combine text without newlines

        # Optional: Truncate the text if it's too long
        if len(text_easyocr_combined) > max_length:
            text_easyocr_combined = text_easyocr_combined[:max_length] + "..."

        # Optional: Clean the text from unwanted characters or symbols
        cleaned_text = re.sub(r"[^\w\s]", "", text_easyocr_combined)  # Removing special characters

        return cleaned_text

    def compare_texts(self, stamp_text, prescription_text):
        """Compare the text from stamp and prescription and return similarity percentage and confidence."""
        # Encode texts using ParsBERT tokenizer
        inputs_stamp = self.tokenizer(stamp_text, return_tensors="pt", padding=True, truncation=True)
        inputs_prescription = self.tokenizer(prescription_text, return_tensors="pt", padding=True, truncation=True)

        # Get embeddings from ParsBERT
        with torch.no_grad():
            embeddings_stamp = self.model_bert(**inputs_stamp).last_hidden_state.mean(dim=1)
            embeddings_prescription = self.model_bert(**inputs_prescription).last_hidden_state.mean(dim=1)

        # Calculate cosine similarity between the embeddings
        similarity = cosine_similarity(embeddings_stamp.cpu().numpy(), embeddings_prescription.cpu().numpy())

        # Convert similarity ratio to a percentage
        similarity_percentage = f"{similarity[0][0] * 100:.2f}%"  # Format as percentage with two decimal places

        # Define a confidence threshold (you can adjust this value as needed)
        confidence = "Similar" if similarity[0][0] > 0.4 else "Not Similar"  # 80% as threshold for "Similar"

        return similarity_percentage, confidence


# Example usage
if __name__ == "__main__":
    model_path = "./model/best.pt"
    output_dir = "./output"
    image_dir = "./input"

    # Ensure the input directory exists
    if not os.path.exists(image_dir):
        print(f"Input directory {image_dir} does not exist.")
    else:
        detector = StampDetector(model_path, output_dir)
        detector.detect_and_save_stamps(image_dir)