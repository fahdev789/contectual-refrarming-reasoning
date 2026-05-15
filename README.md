# Quantifying Verification Drop in Reasoning LLMs

This repository contains the empirical framework, dataset loops, and evaluation engines used to isolate and track **Verification Drop** ($D_v$) under structural input noise distributions.

## 🔬 Core Finding
Contextual reframing and background prompt noise compress an LLM's latent reasoning trace footprint. This causes the internal attention engine to drop active logical self-verification phrases ($D_v \to 0$) well before hitting absolute context allocation bounds.

![Empirical Profile of Verification Drop Behavior](paper/verification_drop_profile.png)

## 🛠️ Rapid Replication Pipeline

### 1. Environment Initialization
Ensure your background local execution server is running ([Ollama Engine Framework](https://ollama.com)), pull the target model, and install the library requirements:
```bash
ollama pull deepseek-r1:1.5b
pip install -r requirements.txt
```

### 2. Execution of Experiment Loop
To run the automated prompt insertion grid and export raw telemetry files to the local file system:
```bash
python src/evaluate.py
```

### 3. Generate Evaluation Analytics
To parse your local results file and generate the publication-ready dual-axis analytical curves:
```bash
python src/plot_results.py
```

## 📄 Manuscript Configuration
The complete LaTeX templates, structural abstract variations, and configuration blocks are organized within the `/paper` folder path.

## 🤝 Citation Layout
```bibtex
@misc{anonymous2026verification,
  title={Quantifying Verification Drop: How Contextual Reframing Erases Latent Self-Correction in Reasoning LLMs},
  author={Independent AI Researcher},
  year={2026},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{https://github.com}}
}
```
