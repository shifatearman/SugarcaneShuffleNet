# SugarcaneShuffleNet: A Fast, Lightweight CNN for Diagnosing 17 Sugarcane Leaf Diseases

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository delivers a complete pipeline—data → model → web app—for real-time, on-device diagnosis of **17** sugarcane leaf conditions (15 diseases + Healthy + Dried Leaves). It includes training notebooks, optimized lightweight models, and a field-deployable Progressive Web Application (PWA) with Grad-CAM explainability and LLM-powered recommendations.

<p align="center">
  <img alt="Pipeline overview" src="https://github.com/user-attachments/assets/a3396039-b3a2-471b-93e6-df1f8e4bc7fe" width="90%">
</p>

---

## 🌟 Overview

SugarcaneShuffleNet is an optimized ShuffleNet backbone that achieves **98.02%** accuracy and **0.98** macro-F₁ while running in **≈ 4 ms** on mid-range mobile CPUs. The end-to-end system helps farmers and agronomists obtain quick, explainable diagnoses directly in the field.

## ✨ Key Features

- **🗂️ SugarcaneLD-BD Dataset** – 638 expertly annotated Bangladeshi images merged with two Indian datasets, yielding **9,908** raw images (17 classes) and **7,037** unique photos after de-duplication; class-balanced augmentation expands the training set to **11,313** images
- **⚡ Lightweight Models** – Implementations and trained weights for ShuffleNet, SqueezeNet, MobileNet V3-Small, MNASNet, EfficientNet-Lite, and EdgeNeXt
- **🎯 Bayesian Hyperparameter Optimization** – Optuna (TPE) search over eight key hyperparameters produces state-of-the-art accuracy–speed trade-offs
- **📱 Real-time PWA (SugarcaneAI)** – Capture or upload leaf images, view Grad-CAM heatmaps, and receive Gemini-powered management advice—all offline-friendly

---

## 📁 Repository Structure

```
.
├── App/
│   └── app.py                          # Progressive Web Application
├── Models/
│   ├── edgenext.ipynb                  # EdgeNeXt training notebook
│   ├── efficientnet-lite.ipynb        # EfficientNet-Lite training notebook
│   ├── mnasnet.ipynb                   # MNASNet training notebook
│   ├── mobilenet.ipynb                 # MobileNet V3-Small training notebook
│   ├── shufflenet.ipynb                # ShuffleNet training notebook
│   └── squeezenet.ipynb                # SqueezeNet training notebook
└── Results/
    ├── *_training_history_plot.png     # Training curves for each model
    └── best_*_model.pth                # Trained model weights
```

---

## 📊 Dataset Summary

| Source | Images | Classes |
|--------|-------:|--------:|
| **SugarcaneLD-BD** (Bangladesh) | 638 | 5 |
| Thite *et al.* (India) | 6,748 | 11 |
| Daphal & Koli (India) | 2,522 | 5 |
| **Total raw** | **9,908** | **17** |
| Unique after de-duplication | 7,037 | 17 |
| Augmented training set | 11,313 | 17 |

### Disease Classes
The dataset includes 15 disease classes plus Healthy and Dried Leaves categories, representing the most common sugarcane leaf conditions found in Bangladesh and India.

---

## 🏆 Model Performance (17-class Test Set)

| Model | Accuracy (%) | Macro F₁ | Size (MB) | Latency (ms/img) |
|-------|-------------:|---------:|----------:|-----------------:|
| **ShuffleNet (SugarcaneShuffleNet)** | **98.02** | **0.98** | **9.26** | **4.14** |
| EfficientNet-Lite | 98.02 | 0.98 | 18.65 | 3.26 |
| MNASNet | 98.51 | 0.98 | 17.59 | 2.91 |
| EdgeNeXt | 98.23 | 0.97 | 22.02 | 3.02 |
| MobileNet V3-Small | 96.53 | 0.96 | 10.25 | 2.55 |
| SqueezeNet | 97.31 | 0.97 | 7.00 | 2.69 |

> **💡 Why ShuffleNet?**  
> Despite MNASNet's slightly higher accuracy, ShuffleNet offers a superior accuracy–latency–size balance, making it ideal for low-power edge devices.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- PyTorch
- Additional dependencies listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/shifatearman/SugarcaneShuffleNet.git
cd SugarcaneShuffleNet
                                                            
# Run the Progressive Web Application
cd App
python app.py
```

The application will be available at `http://localhost:5000`

---

## 🔍 Explainability with Grad-CAM

Grad-CAM heatmaps are generated automatically during inference, highlighting lesion regions that drive predictions. This provides crucial visual feedback for farmers and agronomists to understand what the model is focusing on.

<p align="center">
  <img alt="Grad-CAM example" src="https://github.com/user-attachments/assets/bb432b48-3f31-4359-bce4-773d3df37cdc" width="90%">
</p>

---

## 🤖 AI-Powered Recommendations

After diagnosis, the PWA invokes a Gemini LLM endpoint to return concise, three-part agronomic advice:
1. **Disease Cause** - Understanding the root cause
2. **Immediate Actions** - Steps to take right away
3. **Long-term Control** - Prevention and management strategies

<p align="center">
  <img alt="LLM recommendation example" src="https://github.com/user-attachments/assets/396040ad-c352-4b9d-a823-6a621c559fe6" width="90%">
</p>

---

## 📚 Usage

### Training Models
Each model has its own Jupyter notebook in the `Models/` directory. Simply open and run the desired notebook to train a model from scratch or fine-tune existing weights.

### Using the Web Application
1. Launch the app using `python app.py`
2. Upload or capture a sugarcane leaf image
3. View the diagnosis results with confidence scores
4. Examine the Grad-CAM heatmap for explainability
5. Read AI-generated management recommendations

---

## 📖 Dataset Access

The complete SugarcaneLD-BD dataset will be made available through Mendeley Data upon publication. The link will be updated here once available.

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{arman2024sugarcane,
  title={SugarcaneShuffleNet: A Fast, Lightweight CNN for Diagnosing 17 Sugarcane Leaf Diseases},
  author={Arman, Shifat E. and Abdullah, Hasan M. and Sakib, Syed Nazmus and Saiem, R. M. and Asha, S. N. and Hassan, M. M. and Amin, S. B. and Abrar, S. M. Mahin},
  year={2024}
}
```

---

## 👥 Authors

**Shifat E. Arman** • **Hasan M. Abdullah** • **Syed Nazmus Sakib** • **R. M. Saiem** • **S. N. Asha** • **M. M. Hassan** • **S. B. Amin** • **S. M. Mahin Abrar**

---

## 🙏 Acknowledgements

- Funded by the **Ministry of Science and Technology**, Government of Bangladesh
- Data collected with support from the **Bangladesh Sugarcrop Research Institute (BSRI)** and collaborating farmers
- Special thanks to the farming communities who contributed to this research

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📞 Contact

For questions or collaborations, please reach out through the repository issues or contact the corresponding authors.
