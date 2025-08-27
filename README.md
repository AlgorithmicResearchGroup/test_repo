# MNIST MLP Training

A simple Multi-Layer Perceptron (MLP) implementation for MNIST digit classification using PyTorch.

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**
   ```bash
   cd test_repo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test your installation**
   ```bash
   python test.py
   ```

4. **Start training**
   ```bash
   python train.py
   ```

## 📦 Dependencies

The project requires the following packages (as specified in `requirements.txt`):

- **torch >= 1.9.0** - PyTorch deep learning framework
- **torchvision >= 0.10.0** - Computer vision utilities and datasets
- **numpy >= 1.21.0** - Numerical computing library

### Alternative Installation Methods

**Using a virtual environment (recommended):**
```bash
# Create virtual environment
python -m venv mnist_env

# Activate virtual environment
# On Windows:
mnist_env\Scripts\activate
# On macOS/Linux:
source mnist_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Using conda:**
```bash
# Create conda environment
conda create -n mnist_env python=3.9

# Activate environment
conda activate mnist_env

# Install PyTorch (recommended method for conda)
conda install pytorch torchvision -c pytorch

# Install remaining dependencies
pip install numpy
```

## 🧪 Testing Installation

Before training, run the test script to verify everything is installed correctly:

```bash
python test.py
```

The test script will:
- ✅ Verify all packages are installed with correct versions
- ✅ Test PyTorch functionality (tensors, neural networks, optimization)
- ✅ Test torchvision functionality (transforms, datasets)
- ✅ Test NumPy functionality (arrays, math operations)
- ✅ Check CUDA availability (if applicable)

**Expected output on success:**
```
🎉 All tests passed! Your environment is ready for MNIST MLP training.

You can now run: python train.py
```

## 🏃‍♂️ Running the Training

Once installation is verified, start training:

```bash
python train.py
```

The script will:
1. **Download MNIST dataset** automatically (first run only)
2. **Train the MLP** for 10 epochs
3. **Display progress** with loss and accuracy metrics
4. **Save the trained model** as `mnist_mlp_model.pth`

**Expected performance:** ~97-98% accuracy on MNIST test set

## 🧠 Model Architecture

- **Input Layer:** 784 neurons (28×28 flattened MNIST images)
- **Hidden Layer 1:** 512 neurons + ReLU activation + Dropout (0.2)
- **Hidden Layer 2:** 512 neurons + ReLU activation + Dropout (0.2)  
- **Output Layer:** 10 neurons (digit classes 0-9)
- **Loss Function:** Cross-entropy
- **Optimizer:** Adam (lr=0.001)

## 📁 Project Structure

```
test_repo/
├── requirements.txt    # Package dependencies
├── test.py            # Installation verification script
├── train.py           # MLP training script
└── README.md          # This file
```

## 🔧 Troubleshooting

### Common Issues

**1. Import Error: No module named 'torch'**
```bash
# Make sure you've installed the requirements
pip install -r requirements.txt
```

**2. CUDA out of memory (if using GPU)**
```bash
# The script automatically uses CPU if CUDA is unavailable
# To force CPU usage, you can modify train.py
```

**3. Permission denied when installing packages**
```bash
# Use --user flag or virtual environment
pip install --user -r requirements.txt
```

**4. Old PyTorch version**
```bash
# Upgrade PyTorch
pip install --upgrade torch torchvision
```

### Getting Help

1. **Run the test script** first: `python test.py`
2. **Check Python version**: `python --version` (requires 3.7+)
3. **Verify pip**: `pip --version`
4. **Check installed packages**: `pip list`

## 📊 Results

After training, you should see output similar to:

```
Final Test Accuracy: 97.85%
Model saved as 'mnist_mlp_model.pth'
```

The trained model will be saved and can be loaded later for inference.

## 🤝 Usage Notes

- **First run**: May take longer due to MNIST dataset download (~11MB)
- **Subsequent runs**: Will use cached dataset
- **Training time**: ~2-5 minutes on CPU, faster on GPU
- **Memory usage**: Minimal, works on most systems

## 📝 License

This project is for educational purposes. MNIST dataset is freely available for research and education.
