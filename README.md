# OMR Processor

This project provides a complete Python-based pipeline for reading and evaluating NEET-style OMR sheets using classical computer vision techniques (no pretrained models). It includes document detection, alignment, line detection, and filled bubble analysis using OpenCV and PIL.

## 🔍 Features

- Automatically detects and extracts the document region from the scanned image
- Detects vertical grid lines to localize answer sections
- Identifies filled answer bubbles using adaptive thresholding and region intensity
- Handles real-world distortions and skew using perspective transforms

## 🛠️ Core Dependencies
- OpenCV
- NumPy
- PIL
- Torch (used only to load a document segmentation model)

---

## 🧠 Key Functions

### `extract_document(image_true)`
Detects the edges of the scanned sheet and applies a perspective transform to straighten and crop it.

### `detect_vertical_lines_with_endpoints(image)`
Uses adaptive thresholding and morphology to find vertical grid lines on the OMR sheet.

### `analyze_omr_section(image, num_questions)`
Splits a section into question rows and option columns, evaluates each bubble based on intensity to identify the selected choice.

### `process_omr(image_path)`
Main function that integrates the entire pipeline—document detection, cropping, line detection, and final answer extraction.

---

## 📂 Examples

| Stage                              | Output                                          |
|------------------------------------|-------------------------------------------------|
| Original Scan                      | ![](examples/omr_brown.jpg)                 |
| Document Detection and cropping    | ![](examples/resized_image.jpg)             |
| Line Detection                     | ![](examples/lines_drawn.png)                |
| Bubble Analysis                    | ![](examples/bubbles_drawn.jpg)               |

---

## 📌 Note
- This was built as a lightweight alternative to heavy OCR or deep learning-based OMR systems.
- No pretrained bubble detection models are used — everything is built from first principles using OpenCV.

---

## 👤 Author
Shiva Dhanush
Mohammed Hamza  
For demo purposes.  
Feel free to fork, reuse, or improve!

