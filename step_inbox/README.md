# STEP Inbox

Drop `.step` or `.stp` CAD files directly into this folder.

Then run:

```bash
./scripts/import_steps.sh
```

The importer will:

- copy valid STEP files into `data/raw_step/`
- move the originals into `step_inbox/imported/`
- write an inventory manifest to `data/manifests/step_inventory.json`

After importing, run the trainer:

```bash
./scripts/run_trainer.sh
```

Use the trainer to label each imported file before training the model.
