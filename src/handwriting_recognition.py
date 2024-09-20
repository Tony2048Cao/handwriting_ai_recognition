import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

import cv2
from PIL import Image, ImageOps
import numpy as np
import os, re
from tempfile import mkdtemp
from math import atan2, degrees
import json
from typing import Dict, List



TROCR_BASE_HANDWRITTEN = 'handwriting_ai_recognition/source/modules/trocr_base_handwritten'


class ImageAnalyzer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rotation_angle = 0

    def detect_and_draw(self):
        # 1. Detect text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(self.gray_image)
        
        # 2. Create mask and dilate text horizontally
        mask = np.zeros(self.gray_image.shape, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], 0, (255), -1)
        
        # Horizontal dilation
        kernel = np.ones((1, 150), np.uint8)  # Adjust the size of the dilation kernel to suit your needs
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Create a colored dilated image
        dilated_color = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        dilated_color[dilated != 0] = [0, 255, 255]  # Set dilated area to yellow (BGR format)
    
        # Overlay the dilated image on the original image
        overlay = cv2.addWeighted(self.image, 0.7, dilated_color, 0.3, 0)
        
        # 3. Perform Hough transform on the dilated image
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        # Calculate rotation angle
        angles = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                angle = degrees(theta) - 90  # Convert to angle with horizontal line
                angles.append(angle)
        else:
            print("No lines detected")
        
        self.rotation_angle = np.median(angles) if angles else 0
        
        # 4. Identify boundaries of each text line
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Mark left and right boundaries
            cv2.line(overlay, (x, y), (x, y+h), (255, 0, 0), 2)  # Left boundary
            cv2.line(overlay, (x+w, y), (x+w, y+h), (255, 0, 0), 2)  # Right boundary
            
            # Mark top boundary
            cv2.line(overlay, (x, y), (x+w, y), (0, 0, 255), 2)  # Top boundary

        # Add rotation angle text
        cv2.putText(overlay, f"Rotation Angle: {self.rotation_angle:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return overlay


class ImageLineSegmenter:
    def __init__(self, image_path='./', kernel_width=150):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rotated_image = None
        self.rotation_angle = 0
        self.temp_path = self._create_temp_folder()
        self.kernel_width = kernel_width  # New: Configurable dilation kernel width

    def _create_temp_folder(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        temp_folder_name = f"{base_name}_temp"
        temp_path = os.path.join(os.path.dirname(self.image_path), temp_folder_name)
        os.makedirs(temp_path, exist_ok=True)
        return temp_path

    def _detect_angle(self):
        # Try to detect horizontal lines in the image
        angle = self._detect_horizontal_lines()
        if angle is None:
            angle = self._detect_text_orientation()
        return angle

    def _detect_horizontal_lines(self):
        edges = cv2.Canny(self.gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            angles = [degrees(atan2(y2 - y1, x2 - x1)) for line in lines for x1, y1, x2, y2 in line if abs(degrees(atan2(y2 - y1, x2 - x1))) < 10]
            return np.median(angles) if angles else None
        return None

    def _detect_text_orientation(self):
        # MSER algorithm is used to detect text regions in the image. It can find stable connected regions in grayscale images.
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(self.gray_image)
        
        # Create a convex hull for each detected text region and draw it on the mask.
        mask = np.zeros(self.gray_image.shape, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], 0, (255), -1)
        
        # Dilate the mask horizontally to connect text regions in the same line.
        dilated = cv2.dilate(mask, np.ones((1, 150), np.uint8), iterations=1)

        # Use Canny edge detection algorithm to detect edges in the dilated image.
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)

        # Use Hough transform to detect lines.
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        # Calculate angles:
        if lines is not None:
            angles = [degrees(theta) - 90 for line in lines for _, theta in line]
            return np.median(angles)
        return 0

    def _create_text_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], 0, (255), -1)
        
        # Use configurable kernel width
        kernel = np.ones((1, self.kernel_width), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        return dilated

    # New: Method to adjust dilation kernel width
    def set_kernel_width(self, width):
        self.kernel_width = width
        print(f"Dilation kernel width adjusted to: {width}")

    def get_rotation_angle(self):
        return self.rotation_angle        

    def correct_image_rotation(self):
        self.rotation_angle = self._detect_angle()
        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        self.rotated_image = cv2.warpAffine(self.image, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)      

    def visualize_text_regions(self):
        # Use the rotated image
        image_to_process = self.rotated_image

        dilated = self._create_text_mask(image_to_process)
        dilated_color = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        dilated_color[dilated != 0] = [0, 255, 255]  # Yellow
        
        overlay = cv2.addWeighted(image_to_process, 0.7, dilated_color, 0.3, 0)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green boundary box
            cv2.line(overlay, (x, y), (x, y+h), (255, 0, 0), 2)  # Blue left boundary
            cv2.line(overlay, (x+w, y), (x+w, y+h), (255, 0, 0), 2)  # Blue right boundary
            cv2.line(overlay, (x, y), (x+w, y), (0, 0, 255), 2)  # Red top boundary

        return overlay

    def process_and_visualize(self):
        # Ensure the image is rotated
        if self.rotated_image is None:
            self.correct_image_rotation()
            print(f'Detected rotation angle for the image: {self.rotation_angle}')
        
        # Perform boundary detection and visualization on the rotated image
        visualized_image = self.visualize_text_regions()
        output_path = os.path.join(self.temp_path, "visualized_text_regions.jpg")
        cv2.imwrite(output_path, visualized_image)
        return output_path, self.rotation_angle

    def segment_image_into_lines(self, boundary_overflow=5):
        if self.rotated_image is None:
            self.correct_image_rotation()

        visualized_path, _ = self.process_and_visualize()
        print(f"Boundary detection results for the rotated image saved at: {visualized_path}")

        dilated = self._create_text_mask(self.rotated_image)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_images = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            y_start = max(0, y - boundary_overflow)
            y_end = min(self.rotated_image.shape[0], y + h + boundary_overflow)
            line_image = self.rotated_image[y_start:y_end, :]
            line_images.append(line_image)
            
            line_image_path = os.path.join(self.temp_path, f"line_{idx + 1}.jpg")
            cv2.imwrite(line_image_path, line_image)

        return self.temp_path, self.rotation_angle, len(line_images)

    # to do 
    def segment_image_into_characters(self, boundary_overflow=5):
        if self.rotated_image is None:
            self.correct_image_rotation()

        gray_rotated = cv2.cvtColor(self.rotated_image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding to binarize the image
        binary = cv2.adaptiveThreshold(gray_rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Perform dilation to connect adjacent character parts
        kernel = np.ones((3,20), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        char_images = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small contours
            if w * h < 500:  # Adjust this threshold as needed
                continue

            # Apply boundary overflow parameter
            y_start = max(0, y - boundary_overflow)
            y_end = min(self.rotated_image.shape[0], y + h + boundary_overflow)
            x_start = max(0, x - boundary_overflow)
            x_end = min(self.rotated_image.shape[1], x + w + boundary_overflow)
            
            # Crop character image
            char_image = self.rotated_image[y_start:y_end, x_start:x_end]
            char_images.append(char_image)

            # Save character image
            char_image_path = os.path.join(self.temp_path, f"char_{idx + 1}.jpg")
            cv2.imwrite(char_image_path, char_image)

        return self.temp_path, self.rotation_angle, len(char_images)

    

# need to fix
class HandwritingRecognizer:
    def __init__(self, model_path: str = TROCR_BASE_HANDWRITTEN):
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        # Check if GPU is available and move model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")

    def recognize_image(self, image_path: str, kernel_width:int|None) -> Dict[str, str]:
        self.segmenter = ImageLineSegmenter(image_path=image_path, kernel_width=kernel_width)
        if kernel_width:
            self.segmenter.set_kernel_width(width=kernel_width)
            
        output_dir, _, _ = self.segmenter.segment_image_into_lines(boundary_overflow=10)
        results = {}

        for root, _, files in os.walk(output_dir):
            sorted_files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else float('inf'), reverse=True)
            for file in sorted_files:
                if file.startswith("line_") and file.endswith(".jpg"):
                    line_image_path = os.path.join(root, file)
                    predicted_text = self._recognize_single_image(line_image_path)
                    if predicted_text:
                        results[line_image_path] = predicted_text

        return results

    def save_results_to_json(self, results: Dict[str, Dict[str, str]], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    def _recognize_single_image(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Move input to the same device as the model
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return predicted_text.strip() or None


    def recognize_characters(self, image_path: str, kernel_width: int | None = None) -> Dict[str, str]:
        self.segmenter = ImageLineSegmenter(image_path=image_path, kernel_width=kernel_width)
        if kernel_width:
            self.segmenter.set_kernel_width(width=kernel_width)
            
        output_dir, _, _ = self.segmenter.segment_image_into_characters(boundary_overflow=5)
        results = {}

        for root, _, files in os.walk(output_dir):
            sorted_files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else float('inf'), reverse=True)
            for file in sorted_files:
                if file.startswith("char_") and file.endswith(".jpg"):
                    char_image_path = os.path.join(root, file)
                    predicted_text = self._recognize_single_image(char_image_path)
                    if predicted_text:
                        results[char_image_path] = predicted_text

        return results


def data_cleaning(input_raw_data: str) -> str | None:
    if not input_raw_data:
        return None
    
    cleaned_words = [re.sub(r'[^a-zA-Z]', '', word) for word in input_raw_data.split()]
    cleaned_text = ' '.join(filter(bool, cleaned_words))
    
    return cleaned_text or None
    

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    
    # Recognize a single image
    # image_path = "./test/hand_writing/full_handwriting_test000r.jpg"
    # recognizer = HandwritingRecognizer()
    # results = recognizer.recognize_image(image_path, kernel_width= 30)
    # # results = recognizer.recognize_characters(image_path, kernel_width=100)
    # print(results)
    # for key, re_txt in results.items():
    #     print('-'*40)
    #     print(f'the raw data is -> {re_txt}')
    #     results[key] = data_cleaning(re_txt) # cleaning the raw data 
    #     print(f'the cleaned data is -> {results[key]}')
    # recognizer_result_json = {'recognizer_result': results}
    # json_output = json.dumps(recognizer_result_json, ensure_ascii=False, indent=4)
    # print('-'*40)
    # print(f'Image recognition results:\n{json_output}')

