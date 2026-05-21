# GPTAPPS

# CAD AI Suite

Prototype software for building a text-to-CAD workflow:

- **Trainer app**: import STEP/STP files, label them, extract CAD-oriented features, and train a PyTorch model.
- **Generator app**: turn text prompts into parametric engineering shapes and export CAD artifacts.

This first version is intentionally practical. It trains on structured STEP metadata and labels instead of trying to generate raw STEP text directly. The generation side uses parametric builders, with optional CadQuery support for real STEP export.

## Project Layout

```text
apps/
  trainer_app.py       Label STEP files and launch training
  generator_app.py     Generate CAD from text prompts

cad/
  step_io.py           Lightweight STEP reader
  feature_extraction.py
  parametric_builders.py

data/
  label_store.py       SQLite label database

ml/
  dataset.py
  model.py
  train.py
  inference.py
```

## Quick Start

The workspace already has this run layout:

```text
step_inbox/          Drop STEP/STP files here
data/raw_step/       Imported STEP files used for labeling/training
data/manifests/      Import inventory files
data/generated/      Generated CAD outputs
scripts/             One-command launchers
```

If you need to rebuild the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Start the full GUI workbench:

```bash
./scripts/run_workbench.sh
```

The workbench has tabs for importing STEP files, labeling, training, and generating new CAD.
In the Label tab, select one imported STEP file at a time to preview it and save an individual label.
In the Generate tab, generated STEP files are previewed in the browser after export.

For training with PyTorch:

```bash
pip install -e ".[ml]"
```

For real STEP generation through CadQuery:

```bash
pip install -e ".[cad]"
```

## Label STEP Files

Drop STEP/STP files into:

```text
step_inbox/
```

Then import them into the training workspace:

```bash
./scripts/import_steps.sh
```

The importer copies files into `data/raw_step/`, moves originals into
`step_inbox/imported/`, and writes an inventory manifest at
`data/manifests/step_inventory.json`.

Run the trainer app:

```bash
./scripts/run_trainer.sh
```

Or label a file from the command line:

```bash
cad-ai-trainer label data/raw_step/example.step --label "l_bracket" --tags "bracket,mounting,holes"
```

Labels are stored in:

```text
data/labels.db
```

## Train a Model

```bash
cad-ai-train --db data/labels.db --out data/models/part_classifier.pt
```

Or use:

```bash
./scripts/train_model.sh
```

The first model is a small classifier that learns from STEP entity counts and file-level features. This is the foundation for later models that learn richer geometry, sketches, constraints, or feature trees.

## Generate a Shape

```bash
cad-ai-generator "Generate an aluminum L bracket 80mm by 120mm, 6mm thick, with four M6 holes and a triangular gusset"
```

Or open the generator UI:

```bash
./scripts/run_generator.sh
```

Generate one STEP file directly:

```bash
./scripts/generate_step.sh "Generate a steel plate 100mm by 50mm, 8mm thick, with two M8 holes"
```

Outputs are written under:

```text
data/generated/
```

If CadQuery is installed, the generator will attempt STEP export. Without CadQuery, it writes a JSON design specification that can still be inspected, tested, and used by later exporters.

## Recommended Build Path

1. Label 50 to 100 STEP files across a few part families.
2. Train the first classifier and verify that labels are consistent.
3. Add richer extraction using OpenCascade/CadQuery:
   - exact bounding boxes
   - face areas
   - hole axes
   - wall thickness
   - fillets/chamfers
   - sketch-like profiles
4. Add more parametric builders:
   - L brackets
   - flat plates
   - gusseted brackets
   - simple fixtures
   - hole-pattern plates
5. Train models to predict parametric design intent from labels and prompts.

The most reliable long-term architecture is not "AI emits STEP text." It is "AI predicts clean CAD intent, then a CAD kernel creates valid geometry."
