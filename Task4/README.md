# Task 4 — Cross-Instrument Note Timbre Transfer (NSynth)

This task (CSE 153/253 Assignment 2, Task 4: *continuous, conditioned generation*)
builds a generative model that performs **cross-instrument timbre transfer at the
single-note level**. Given an audio recording of a single note played by a source
instrument *X* and a target instrument label *Y*, the model generates a new waveform of
the same note — preserving pitch and approximate loudness — rendered with the timbre of
*Y*. We train a conditional convolutional encoder–decoder that maps a source log-mel
spectrogram plus a learned target-instrument embedding to a predicted target log-mel,
then render audio with a Griffin-Lim vocoder. The dataset is
[NSynth](https://magenta.tensorflow.org/datasets/nsynth) (Engel et al., 2017).

## Setup

```bash
pip install -r requirements.txt
jupyter notebook task4_workbook.ipynb   # then Run All
```

The notebook downloads `nsynth-valid` (~4.5 GB) into `data/` on first run (idempotent —
skipped if already present). Trained weights are cached in `checkpoints/`; audio and the
`continuous_conditioned.mp3` deliverable are written to `outputs/`. Set `RETRAIN = True`
in the setup cell to retrain from scratch.

## One-notebook constraint

Per the assignment, **all pipeline logic lives in `task4_workbook.ipynb`** — there is no
`src/` directory and no separate `.py` files. Helper functions are defined in notebook
cells next to where they are used, and the markdown narrative is written for peer graders
who read the exported HTML rather than executing the code.

## Layout

```
Task4/
├── task4_workbook.ipynb   # the deliverable — all code lives here
├── requirements.txt
├── README.md
├── data/                  # NSynth (gitignored)
├── checkpoints/           # model + classifier weights (gitignored)
└── outputs/               # sanity-check audio + continuous_conditioned.mp3
```
