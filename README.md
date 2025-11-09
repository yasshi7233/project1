# Smart Product Pricing Challenge - ML Solution

This repository contains a complete machine learning solution for predicting product prices based on catalog content and product images.

## Project Structure

```
├── src/
│   ├── utils.py              # Image download and preprocessing utilities
│   └── predict_test.py       # Main prediction script
├── notebooks/
│   ├── data_exploration.ipynb    # Data analysis and preprocessing
│   ├── baseline_model.ipynb      # Text-only baseline model
│   ├── image_features.ipynb      # Image feature extraction
│   └── final_model.ipynb         # Final ensemble model
├── dataset/
│   ├── train.csv             # Training data (75k samples)
│   ├── test.csv              # Test data (75k samples)
│   └── [generated files]     # Preprocessed data and features
├── models/                   # Trained models and preprocessing objects
├── images/                   # Downloaded product images
├── submission/
│   └── test_out.csv         # Final predictions
├── requirements.txt          # Python dependencies
├── sample_code.py           # Sample dummy implementation
└── Documentation_template.md # Documentation template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Data Exploration

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

### 3. Train Baseline Model

```bash
jupyter notebook notebooks/baseline_model.ipynb
```

### 4. Extract Image Features (Optional)

```bash
jupyter notebook notebooks/image_features.ipynb
```

### 5. Train Final Model

```bash
jupyter notebook notebooks/final_model.ipynb
```

### 6. Generate Predictions

```bash
python src/predict_test.py
```

## Methodology

### Data Preprocessing

1. **Text Processing**:
   - Clean catalog content (lowercase, remove special characters)
   - Extract Item Pack Quantity (IPQ) using regex patterns
   - Generate text statistics (length, word count)

2. **Image Processing**:
   - Download product images from URLs
   - Extract features using pretrained ResNet50
   - Handle missing/failed image downloads

### Feature Engineering

1. **Text Features**:
   - TF-IDF vectorization (5000 features, 1-2 grams)
   - Numerical features (IPQ, text length, word count)

2. **Image Features**:
   - ResNet50 features (2048 dimensions)
   - Zero-padding for missing images

### Model Architecture

1. **Base Models**:
   - LightGBM with optimized hyperparameters
   - XGBoost with optimized hyperparameters

2. **Ensemble**:
   - Simple averaging of base model predictions
   - Ensures positive price predictions

### Performance

- **Validation SMAPE**: [To be filled after training]
- **Training Time**: [To be filled after training]
- **Model Size**: Under 8B parameters (compliant with requirements)

## Key Features

- ✅ **Multi-modal Approach**: Combines text and image features
- ✅ **Robust Preprocessing**: Handles missing data and edge cases
- ✅ **Ensemble Learning**: Combines multiple models for better performance
- ✅ **Compliance**: No external price lookup, uses only provided data
- ✅ **Scalable**: Efficient feature extraction and model training
- ✅ **Reproducible**: Fixed random seeds and versioned dependencies

## File Descriptions

### Core Scripts

- `src/utils.py`: Utilities for image downloading and preprocessing
- `src/predict_test.py`: Main script for generating test predictions

### Notebooks

- `data_exploration.ipynb`: Data analysis, preprocessing, and feature engineering
- `baseline_model.ipynb`: Text-only baseline model implementation
- `image_features.ipynb`: Image feature extraction using ResNet50
- `final_model.ipynb`: Final ensemble model with text + image features

### Sample Files

- `sample_code.py`: Dummy implementation for reference
- `Documentation_template.md`: Template for methodology documentation

## Usage Examples

### Generate Sample Predictions

```python
python sample_code.py
```

### Train and Predict (Full Pipeline)

```python
# 1. Run data exploration notebook
# 2. Run baseline model notebook
# 3. Run image features notebook (optional)
# 4. Run final model notebook
# 5. Generate predictions
python src/predict_test.py
```

## Dependencies

- **Core ML**: pandas, numpy, scikit-learn
- **Gradient Boosting**: lightgbm, xgboost
- **Deep Learning**: torch, torchvision
- **Image Processing**: Pillow, opencv-python
- **Utilities**: requests, tqdm, matplotlib, seaborn

## Compliance

This solution strictly adheres to the challenge requirements:

- ❌ **No External Price Lookup**: No web scraping or external price databases
- ✅ **Training Data Only**: Uses only provided train.csv and test.csv
- ✅ **Image Processing**: Downloads and processes images from provided URLs
- ✅ **Model Constraints**: Models under 8B parameters
- ✅ **Output Format**: Generates test_out.csv in required format

## Performance Optimization

- **Memory Efficient**: Sparse matrices for text features
- **GPU Support**: CUDA acceleration for image processing
- **Batch Processing**: Efficient image feature extraction
- **Caching**: Saves preprocessed features to avoid recomputation

## Troubleshooting

### Common Issues

1. **Image Download Failures**: 
   - Check internet connection
   - Increase retry attempts in utils.py
   - Some URLs may be invalid

2. **Memory Issues**:
   - Reduce TF-IDF max_features
   - Process images in smaller batches
   - Use CPU instead of GPU if needed

3. **Model Loading Errors**:
   - Ensure all model files are in models/ directory
   - Check file paths in predict_test.py

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact the team or create an issue in the repository.
