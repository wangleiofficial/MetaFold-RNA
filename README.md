# 🧬 **MetaFold-RNA**

> **MetaFold-RNA** is a deep learning toolkit for RNA secondary structure prediction, featuring meta-learning and advanced neural architectures.

---

## ✨ **Features**

- 📚 **Multi-dataset support:** CASP, PDB, and more
- 🧠 **Modular meta-learner & model architectures**
- ⚡ **GPU acceleration** for fast inference
- 📄 **Standard output formats**

---

## 📁 **Directory Structure**

```
MetaFold-RNA/
├── run_metafold.py
├── dataset/
├── MetaFold/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── postprocess.py
│   └── utils.py
├── model_checkpoint/
└── README.md
```

---

## 🛠️ **Requirements**

| Package   | Version    |
|-----------|------------|
| Python    | 3.7+       |
| PyTorch   | latest     |
| numpy     | latest     |
| tqdm      | latest     |

**Install dependencies:**
```sh
pip install torch numpy tqdm
```

---

## 🚀 **Quick Start**

1. 📦 **Prepare your dataset**  
   Place files like `CASP15.pickle`, `PDB.pickle` in the `dataset/` directory.

2. 💾 **Download or place model weights**  
   Put files like `model_pdb.pth` in `model_checkpoint/`.

3. 🔮 **Run prediction:**
   ```sh
   python run_metafold.py --casp_pdb_path ./dataset --model-path ./model_checkpoint/model_pdb.pth --device cuda:0 --output-path ./output
   ```

   **Or use a FASTA file:**
   ```sh
   python run_metafold.py --fasta-path ./dataset/test.fasta --model-path ./model_checkpoint/model_pdb.pth --device cuda:0 --output-path ./output
   ```

---

## ⚙️ **Arguments**

| Argument           | Description                                 |
|--------------------|---------------------------------------------|
| `--casp_pdb_path`  | Path to dataset folder                      |
| `--fasta-path`     | Path to input FASTA file                    |
| `--model-path`     | Path to model weights                       |
| `--device`         | Device for inference (`cuda:0` or `cpu`)    |
| `--output-path`    | Output directory                            |

---

## 📖 **Citation**

If you use MetaFold-RNA in your research, please cite the original paper.
