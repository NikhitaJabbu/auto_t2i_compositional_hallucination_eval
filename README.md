# Scene Graph-Based Hallucination Evaluation for Text-to-Image Models

This project evaluates how well text-to-image diffusion models follow compositional prompts. Each prompt is parsed into a scene graph that captures objects, their attributes, and the spatial relations between them. The pipeline then generates an image, detects what is actually in it, and scores five hallucination metrics automatically.

The evaluation was run on SD 1.5, SDXL, SD 3.5, and Flux Schnell using 150 compositional prompts sourced from T2I-CompBench.

# What Gets Measured

| Metric | Description |
|---|---|
| **H_obj** | Object hallucination scored through scene graph QA using Llama 3.1 |
| **H_missed** | Objects mentioned in the prompt that were not detected in the image |
| **H_extra** | Objects detected in the image that were not part of the prompt |
| **H_attr** | Attribute errors across color, material, and shape using Qwen2-VL and OpenCLIP |
| **H_rel** | Spatial relation errors computed from bounding box geometry and SAM masks |

All scores range from 0 to 1. Lower means fewer hallucinations.

---

# System Requirements

- Linux with an NVIDIA GPU (tested on RTX 3090 with 24 GB VRAM)
- CUDA 11.8 or 13.0
- Python 3.10 or higher
- ComfyUI running locally at `http://127.0.0.1:8188`

---

# Setup

Clone the repository and run the setup script. It handles all package installs, model downloads, and third-party repos automatically.

```bash
git clone https://github.com/your-username/Project_H.git
cd Project_H
chmod +x setup.sh
./setup.sh
```

The setup script does the following in order:

1. Installs all Python packages including torch, transformers, open-clip, and segment-anything
2. Downloads the SpaCy language model `en_core_web_sm`
3. Downloads SAM ViT-H weights into `weights/` (around 2.4 GB)
4. Clones and installs GroundingDINO from IDEA-Research
5. Clones and installs CountGD from niki-amini-naieni
6. Installs Ollama and pulls the Llama 3.1 8B model (around 5 GB)
7. Pre-downloads Qwen2-VL-7B (around 15 GB) and OpenCLIP ViT-L/14

**ComfyUI is not included in this repo and needs to be installed separately.** Once installed, load the workflow files from the `Models/` folder. The [ComfyUI GitHub page](https://github.com/comfyanonymous/ComfyUI) has the full installation guide.

---

# Running the Pipeline

Make sure ComfyUI and Ollama are both running before starting. Then run:

```bash
python run.py
```

The script asks to pick a model and a mode:

```
Select model:
  1. sdxl
  2. flux_schnell
  3. sd15
  4. sd35

Select mode:
  0. Regular      — full baseline run
  1. Experiment 1 — latent resolution comparison across 5 sizes
  2. Experiment 2 — steps and CFG sensitivity sweep
  3. Experiment 3 — seed variance across 20 runs per prompt
```

# Experiment Details

**Baseline (Mode 0)**
Each of the 150 prompts is parsed into a scene graph, an image is generated at the model's default settings, and all five hallucination metrics are computed. This is the standard run used for cross-model comparison.

**Experiment 1 — Latent Resolution**
The same 150 prompts are run at five different latent sizes: 256x256, 1024x1024, 2048x2048, 1048x256, and 256x2048. All other settings stay the same. Hallucination scores are compared across sizes to see how resolution changes what the model renders.

**Experiment 2 — Steps and CFG Sensitivity**
Two separate sweeps are run. In the steps sweep, images are generated at 6, 35, 65, and 90 sampling steps while CFG is held at 8. In the CFG sweep, CFG is set to 4, 17, 24, and 35 while steps are held at 20. Each configuration runs all 150 prompts and produces a full set of hallucination scores.

**Experiment 3 — Seed Variance**
Each of the 150 prompts is generated 20 times using different random seeds. The hallucination scores from all 20 runs are averaged per prompt. This shows which prompts are reliably hard for the model across different generations and which ones vary a lot depending on the seed.

---

## Output Structure

Results are saved under `Outputs/` and also written to `RESULTS.md` after each run.

```
Outputs/
  shared/                  # parsed prompts and QA pairs, generated once and reused
  models/{model}/
    baseline/
    experiment_1/
    experiment_2/
    experiment_3/
  comparison/              # cross-model summary tables
```

---

## Baseline Results

Evaluated on 150 prompts. All scores are hallucination rates, lower is better.

| Model | H_obj | H_attr | H_rel | H_missed | H_extra |
|---|---|---|---|---|---|
| SD 1.5 | 0.1800 | 0.4767 | 0.4333 | 0.1229 | 0.3684 |
| SDXL | 0.2033 | 0.5700 | 0.5400 | 0.1462 | 0.2345 |
| SD 3.5 | 0.1200 | 0.2233 | 0.3467 | 0.0831 | 0.2596 |
| Flux Schnell | 0.1367 | 0.3033 | 0.3867 | 0.0897 | 0.3224 |

SD 3.5 scores the lowest hallucination rate overall. Attribute prediction is the hardest metric across all four models.

---

## Project Files

```
Project_H/
  run.py                  # entry point, model and mode selection
  prompt_triples.py       # parses prompts into scene graph triples
  qa_generator.py         # generates QA pairs from scene graphs using Llama 3.1
  ImgGeneration.py        # sends workflows to ComfyUI and retrieves images
  ObjectDetection.py      # two-pass GroundingDINO detection and SAM segmentation
  attributeprediction.py  # attribute scoring with Qwen2-VL and OpenCLIP
  relretry.py             # spatial relation prediction from geometry and masks
  hallucination_eval.py   # computes all five hallucination metrics
  experiment_runner.py    # runs and organizes the three experiments
  update_results.py       # writes RESULTS.md from saved JSON outputs

  data/
    promts_150.jsonl      # 150 T2I-CompBench prompts used in the study
    prompts_500.jsonl     # extended set of 500 prompts

  Models/                 # ComfyUI workflow JSON files per model
  weights/                # SAM ViT-H weights downloaded by setup.sh
  vocab/                  # vocabulary files used by OpenCLIP normalization
```

---

## External Repos Used

These are not included here. The setup script clones and installs them automatically.

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — open-vocabulary object detection
- [CountGD](https://github.com/niki-amini-naieni/CountGD) — counting objects from text queries
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — mask generation from detected boxes

---

## Prompts Reference

The 150 evaluation prompts come from:

> Huang, K., Sun, K., Xie, E., Li, Z., & Liu, X. (2023). T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation. NeurIPS 2023.
