# TextRemovalPipeline

A sophisticated text removal system using Stable Diffusion for high-quality inpainting, inspired by Meta's TextStyleBrush. This project aims to build a foundation for text style transfer with special focus on Indian languages.

## 🚧 Project Status: Under Construction 🚧

The text removal component is currently under active development. While the architecture is in place, some components are still being optimized for better performance.

## Architecture

The system follows a comprehensive pipeline for text detection and removal:

![Text Removal System Architecture](assets/architecture_diagram.png)

### Key Components

#### 1. Input Processing
- Image loading and preprocessing
- Bounding box data parsing
- Inpainting model preparation
- Debug environment setup

#### 2. Box Handling
- Coordinate extraction and scaling
- Figure box preservation
- Region of interest extraction

#### 3. Text Detection Pipeline
Multiple detection methods are employed for robustness:
- Grayscale conversion
- Adaptive thresholding (Gaussian/Mean methods)
- Simple thresholding (High/Medium options)
- Edge detection (Canny)
- Color-based detection (Dark/Light/Colored text)

#### 4. Mask Processing
- Mask combination and refinement
- Morphological operations
- Dilation for coverage
- Validation checks

#### 5. Background Analysis
- Background extraction and analysis
- Color and texture profiling
- Dominant color identification
- Context-aware prompt generation

#### 6. Inpainting Processes
**Primary**: Diffusion-Based Inpainting
- Stable Diffusion 2 model
- Context-aware custom prompting
- 30 inference steps with guidance scale 1.5

**Fallback**: OpenCV Inpainting
- Navier-Stokes algorithm
- Fast Marching Method (TELEA)
- Bilateral filtering for texture preservation

#### 7. Output Generation
- Result compositing
- Visualization generation
- Debug image output
- Final result saving

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TextRemovalPipeline.git
cd TextRemovalPipeline

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the models (large files)
python download_models.py
