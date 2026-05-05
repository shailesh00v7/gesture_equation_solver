# ✍️Hand Gesture-Based Equation Solver (OpenCV + CNN + OCR)

This project allows users to **draw digits and operators** using OpenCV, then recognizes them with a **CNN model** for digits and **EasyOCR** for operators.  
Finally, it reconstructs and evaluates the full mathematical expression using **SymPy**.

## 🚀 Features
- Hand gesture-based equation drawing (`painter.py`)
- CNN digit classifier trained on MNIST + custom operators (`sys.py`)
- Operator recognition with EasyOCR (`join.py`)
- Equation parsing & solving (SymPy)
- Undo/redo functionality for drawing

## 📂 Repository Structure
- `painter.py` → OpenCV drawing & main loop
- `join.py` → Digit & operator recognition
- `sys.py` → CNN training script
- `models/digit_operator_model.h5` → Pretrained CNN
- `dataset/` → Custom operator dataset + optional digit samples

## ⚙️ Installation
```bash
git clone https://github.com/shailesh00v7/gesture-equation-solver.git
cd gesture-equation-solver
pip install -r requirements.txt
