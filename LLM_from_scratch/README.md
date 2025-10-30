# GenAI

From‑scratch **GPT-style language model** in PyTorch with a minimal tokenizer, attention, and Transformer blocks — plus **fine‑tuning** for **SMS spam classification** and **instruction tuning**. This repo is centered around the notebook `hands_on_llm_fine_tuned.ipynb`.

## ✨ Highlights
- **Custom tokenizer** (basic + Byte‑Pair Encoding) and token→ID pipeline
- **Embeddings + positional encodings**
- **Self‑Attention / Causal Attention / Multi‑Head Attention**
- **Transformer blocks** (LayerNorm, GELU, residual/shortcut connections)
- **GPT‑style model** with text generation
- **Training loop** with train/val loss evaluation & weight saving/loading
- **Fine‑tuning for classification** on the UCI SMS Spam Collection dataset
- **Instruction fine‑tuning** with a small JSON instruction dataset
- Utility cells for **reproducibility** (seeds) and **data loaders**

> Built to be educational and hackable — great as a learning scaffold or a base for small experiments.

---

## 📁 Project Structure
```
hands_on_llm_fine_tuned.ipynb   # Main notebook with full pipeline
data/                           # (Optional) place datasets here
├─ sms_spam_collection.zip      # UCI SMS Spam Collection (or extracted CSVs)
├─ instruction-data.json        # Instruction tuning data
└─ the-verdict.txt              # Sample text for pretraining demos
```

Datasets are downloaded in‑notebook from:
- UCI SMS Spam: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- Instruction data (example path used in notebook): `instruction-data.json`
- Sample text: `the-verdict.txt` (from the LLMs‑from‑scratch examples)

---

## 🛠 Requirements
- Python 3.9+
- PyTorch
- NumPy, pandas, matplotlib, tqdm
- (Optional) `tiktoken`, `gensim`, `tensorflow` (used in some cells)
- Jupyter

You can capture exact versions with:
```bash
pip freeze > requirements.txt
```

---

## 🚀 Quickstart
1. **Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install packages**
```bash
pip install torch numpy pandas matplotlib tqdm tiktoken gensim tensorflow jupyter
```

3. **Run the notebook**
```bash
jupyter notebook hands_on_llm_fine_tuned.ipynb
```
Follow the sections in order (tokenizer → attention → GPT → training → fine‑tuning).

---

## 🧪 Fine‑Tuning for SMS Spam Classification
The notebook defines a `SpamDataset` and fine‑tunes the model for a binary classifier.

**Data**  
- Source: UCI SMS Spam Collection  
- Files (after extraction): `train.csv`, `validation.csv`, `test.csv` (or adapt paths used in the notebook)

**Run**  
Execute the section **“Fine‑Tune for Classification”**:
- Preprocess & tokenize
- Create `DataLoader`s
- Initialize model head for classification
- Train and track **loss & accuracy**
- Evaluate on test set

---

## 📝 Instruction Fine‑Tuning
A lightweight **`InstructionDataset`** is used to fine‑tune the base GPT for instruction–response behavior.

**Run**  
Execute the section **“Instruction Fine‑Tuning”**:
- Load `instruction-data.json` (prompt/response pairs)
- Build training batches & `DataLoader`
- Load the pretrained (base) model
- Fine‑tune and sample generations

---

## 💾 Saving / Loading Weights
The notebook includes utilities to:
- `torch.save(state_dict)` after training
- `model.load_state_dict(...)` to resume

Use the **“Loading and Saving Model Weights”** section to persist experiments.

---

## 📊 Evaluation & Generation
- Evaluate **train/val loss** with helper functions
- Generate text from random or trained weights (sampling top‑k/temperature depending on your edits)

---

## ⚠️ Notes
- The code is designed for **clarity over speed**; expect educational readability.
- Some sections use optional libraries (`tiktoken`, `tensorflow`, `gensim`). If not needed, you can skip those cells.
- GPU recommended for faster training (configure your PyTorch device).

---

## 🧾 License
MIT (see `LICENSE` file).

## 🙏 Acknowledgments
Parts of the data paths and sample text are inspired by the “LLMs from Scratch” educational materials.
