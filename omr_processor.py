import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image, ImageEnhance
import gc

class OMRProcessor:
    def __init__(self, model_path="model_mbv3_iou_mix_2C049.pth"):
        """
        Initialize OMR Processor with the trained model
        
        Args:
            model_path (str): Path to the trained model weights file
        """
        self.device = torch.device("cpu")  # Using CPU only for hosting
        self.model_path = model_path
        self.trained_model = None
        self.preprocess_transforms = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model for document extraction"""
        try:
            self.trained_model = self._load_model_weights(num_classes=2, model_name="mbv3", 
                                                        checkpoint_path=self.model_path, device=self.device)
            self.preprocess_transforms = self._image_preprocess_transforms()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _image_preprocess_transforms(self, mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
        """Create image preprocessing transforms"""
        return torchvision_T.Compose([
            torchvision_T.ToTensor(), 
            torchvision_T.Normalize(mean, std)
        ])
    
    def _load_model_weights(self, num_classes=1, model_name="mbv3", checkpoint_path=None, device=None):
        """Load the trained model weights"""
        if model_name == "mbv3":
            model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
        else:
            raise ValueError("Only mbv3 model is supported")
        
        model.to(device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
        
        checkpoints = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoints, strict=False)
        model.eval()
        
        # Test the model with a dummy input
        _ = model(torch.randn((2, 3, 384, 384)))
        
        return model
    
    def _order_points(self, points):
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        points = np.array(points, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        
        # Calculate sums and differences
        s = points.sum(axis=1)  # x + y
        
        # Assign corners
        rect[0] = points[np.argmin(s)]  # Top-left: smallest sum
        rect[2] = points[np.argmax(s)]  # Bottom-right: largest sum
        
        # Remove assigned points
        remaining = [i for i in range(len(points)) if i not in [np.argmin(s), np.argmax(s)]]
        remaining_points = points[remaining]
        
        # Assign remaining corners based on difference
        diff_remaining = np.diff(remaining_points, axis=1).flatten()
        rect[1] = remaining_points[np.argmin(diff_remaining)]  # Top-right: smallest difference
        rect[3] = remaining_points[np.argmax(diff_remaining)]  # Bottom-left: largest difference
        
        return rect.astype("int").tolist()
    
    def _find_dest(self, pts):
        """Find destination points for perspective transform"""
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return self._order_points(destination_corners)
    
    def extract_document(self, image_true=None, image_size=384, BUFFER=10):
        """
        Extract and straighten the document from the image using the trained model
        
        Args:
            image_true: Input image (RGB format)
            image_size: Size for model processing
            BUFFER: Buffer for edge detection
            
        Returns:
            Processed and straightened document image
        """
        if image_true is None:
            raise ValueError("Input image is required")
        
        IMAGE_SIZE = image_size
        half = IMAGE_SIZE // 2
        
        imH, imW, C = image_true.shape
        
        # Resize image for model processing
        image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        
        scale_x = imW / IMAGE_SIZE
        scale_y = imH / IMAGE_SIZE
        
        # Preprocess image for model
        image_model = self.preprocess_transforms(image_model)
        image_model = torch.unsqueeze(image_model, dim=0)
        
        # Run inference
        with torch.no_grad():
            out = self.trained_model(image_model)["out"].cpu()
        
        del image_model
        gc.collect()
        
        # Process model output
        out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
        r_H, r_W = out.shape
        
        _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
        _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
        out = _out_extended.copy()
        
        del _out_extended
        gc.collect()
        
        # Edge Detection
        canny = cv2.Canny(out.astype(np.uint8), 225, 255)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            raise ValueError("No contours found in the image")
        
        page = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        
        # Approximate corners
        epsilon = 0.02 * cv2.arcLength(page, True)
        corners = cv2.approxPolyDP(page, epsilon, True)
        
        if len(corners) < 4:
            raise ValueError("Could not detect 4 corners of the document")
        
        corners = np.concatenate(corners).astype(np.float32)
        
        # Scale corners back to original image coordinates
        corners[:, 0] -= half
        corners[:, 1] -= half
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y
        
        # Check if corners are inside image bounds
        if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):
            # Expand image if needed
            left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0
            
            rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
            box = cv2.boxPoints(rect)
            box_corners = np.int32(box)
            
            box_x_min = np.min(box_corners[:, 0])
            box_x_max = np.max(box_corners[:, 0])
            box_y_min = np.min(box_corners[:, 1])
            box_y_max = np.max(box_corners[:, 1])
            
            if box_x_min <= 0:
                left_pad = abs(box_x_min) + BUFFER
            if box_x_max >= imW:
                right_pad = (box_x_max - imW) + BUFFER
            if box_y_min <= 0:
                top_pad = abs(box_y_min) + BUFFER
            if box_y_max >= imH:
                bottom_pad = (box_y_max - imH) + BUFFER
            
            # Create extended image
            image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)
            image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image_true
            image_extended = image_extended.astype(np.float32)
            
            # Adjust corners
            box_corners[:, 0] += left_pad
            box_corners[:, 1] += top_pad
            corners = box_corners
            image_true = image_extended
        
        # Order corners and find destination
        corners = sorted(corners.tolist())
        corners = self._order_points(corners)
        destination_corners = self._find_dest(corners)
        
        # Apply perspective transform
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
        final = np.clip(final, a_min=0., a_max=255.)
        
        return final
    
    def crop_image(self, image, roi_coordinates):
        """Crop image using ROI coordinates"""
        return image[roi_coordinates['y1']:roi_coordinates['y2'], roi_coordinates['x1']:roi_coordinates['x2']]
    
    def detect_vertical_lines_with_endpoints(self, image, min_line_length=50):
        """Detect vertical lines in the image and return their endpoints"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        binary_vertical = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, 15, 10)
        
        # Use vertical structuring element to detect lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vertical_lines = cv2.morphologyEx(binary_vertical, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours of detected vertical lines
        contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_endpoints = [(x, y, x, y + h) for x, y, w, h in [cv2.boundingRect(c) for c in contours] if h >= min_line_length]
        
        return line_endpoints
    
    def crop_between_lines_by_points(self, image, sorted_endpoints, line1, line2):
        """Crop image between two vertical lines"""
        x1_start, y1_start, _, y1_end = sorted_endpoints[line1 - 1]
        x2_start, y2_start, _, y2_end = sorted_endpoints[line2 - 1]
        x_min, x_max = min(x1_start, x2_start), max(x1_start, x2_start)
        y_min, y_max = min(y1_start, y2_start), max(y1_end, y2_end)
        return image[y_min:y_max, x_min:x_max]
    
    def analyze_omr_section(self, image, num_questions, options_per_question=4):
        """
        Analyze OMR bubbles in a section and return selected answers
        
        Args:
            image: Input image section
            num_questions: Number of questions in this section
            options_per_question: Number of options per question (default 4)
            
        Returns:
            Dictionary with question numbers as keys and selected options as values
        """
        # Convert to PIL and enhance contrast
        image_pil = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(2.0)
        
        # Convert to grayscale and apply thresholding
        gray = np.array(image_pil.convert("L"))
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 10)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        
        # Calculate bubble dimensions
        question_number_width = binary_image.shape[1] // 10
        bubble_width = (binary_image.shape[1] - question_number_width) // options_per_question
        bubble_height = binary_image.shape[0] // num_questions
        
        results = {}
        
        for q in range(num_questions):
            max_intensity = 0
            selected_option = None
            
            for opt in range(options_per_question):
                x1 = question_number_width + (opt * bubble_width)
                y1 = q * bubble_height
                x2 = x1 + bubble_width
                y2 = y1 + bubble_height
                
                bubble_roi = binary_image[y1:y2, x1:x2]
                mean_intensity = cv2.mean(bubble_roi)[0]
                
                if mean_intensity > 105 and mean_intensity > max_intensity:
                    max_intensity = mean_intensity
                    selected_option = opt + 1
            
            if selected_option is not None:
                results[q + 1] = selected_option
        
        return results
    
    def process_omr(self, image_path, roi_coordinates=None):
        """
        Main function to process OMR sheet and return results
        
        Args:
            image_path: Path to the input image
            roi_coordinates: Dictionary with x1, y1, x2, y2 coordinates for cropping
            
        Returns:
            Dictionary containing OMR results for all sections
        """
        # Default ROI coordinates if not provided
        if roi_coordinates is None:
            roi_coordinates = {'x1': 250, 'y1': 50, 'x2': 755, 'y2': 900}
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract and straighten document
        processed_image = self.extract_document(image_true=image_rgb)
        
        # Convert back to BGR for OpenCV operations
        processed_image_bgr = cv2.cvtColor(processed_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Crop to OMR region
        cropped_image = self.crop_image(processed_image_bgr, roi_coordinates)
        
        # Detect vertical lines
        endpoints = self.detect_vertical_lines_with_endpoints(cropped_image, min_line_length=200)
        endpoints = sorted(endpoints, key=lambda p: p[0])
        
        if len(endpoints) < 20:
            raise ValueError(f"Expected at least 20 vertical lines, found {len(endpoints)}")
        
        # Define line pairs for different sections
        line_pairs = [(5, 7), (9, 11), (13, 15), (17, 19),  # Sections 1-4 (35 questions)
                      (6, 8), (10, 12), (14, 16), (18, 20)] # Sections 5-8 (15 questions)
        
        all_section_results = {}
        
        for idx, (start_line, end_line) in enumerate(line_pairs):
            try:
                cropped_section = self.crop_between_lines_by_points(cropped_image, endpoints, start_line, end_line)
                num_questions = 35 if idx < 4 else 15  # First 4 sections have 35 questions, last 4 have 15
                
                section_results = self.analyze_omr_section(cropped_section, num_questions=num_questions)
                all_section_results[f"Section_{idx + 1}"] = section_results
                
            except Exception as e:
                print(f"Error processing section {idx + 1}: {e}")
                all_section_results[f"Section_{idx + 1}"] = {}
        
        return all_section_results

def main():
    """Main function for testing the OMR processor"""
    # Initialize processor
    processor = OMRProcessor()
    
    # Test with a default image (you can change this path)
    test_image_path = "test_omr.jpg"  # Replace with your test image path
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please provide a valid image path for testing")
        return
    
    try:
        # Process OMR
        results = processor.process_omr(test_image_path)
        
        print("OMR Processing Results:")
        print("=" * 50)
        for section, section_results in results.items():
            print(f"\n{section}:")
            for question, answer in section_results.items():
                print(f"  Q{question}: Option {answer}")
        
        print(f"\nTotal questions answered: {sum(len(section) for section in results.values())}")
        
    except Exception as e:
        print(f"Error processing OMR: {e}")

if __name__ == "__main__":
    main() 