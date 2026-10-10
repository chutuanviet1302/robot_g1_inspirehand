# Fine-tuning SmolVLA on Kaggle

The GTX 1650 (4 GB) cannot fine-tune SmolVLA (~450 M parameters); Kaggle's free T4 (16 GB) can.

1. **Record and export** (on the laptop):

   ```
   homehand collect-vla --name vla_v1 -n 200
   .venv-vla\Scripts\python -m homehand.data.export_lerobot --name vla_v1
   ```

   Then zip the folder `data/lerobot/vla_v1` (right click → Send to → Compressed folder).

2. **Upload**: kaggle.com → Datasets → New Dataset → upload `vla_v1.zip`, name it `homehand-vla-v1`
   (private). Kaggle unzips it.

3. **Notebook**: kaggle.com → Code → New Notebook → File → Import Notebook → `kaggle/smolvla_finetune.ipynb`.
   In the right panel: Accelerator **GPU T4 x2**, Internet **on**, add the dataset `homehand-vla-v1` as input.

4. **Run**: Save Version → **Save & Run All (Commit)**. It runs in the background (≈4 h for 20 k steps at
   batch 16) and the checkpoints end up in the version's Output tab.

5. **Back on the laptop**: download `smolvla_homehand.zip` from the Output tab, unzip it to
   `models/smolvla_homehand/`, then

   ```
   .venv-vla\Scripts\python scripts/eval_smolvla.py --model models/smolvla_homehand -n 20
   ```

If a run stops early (12 h session limit), create a dataset from the version's output, add it as input to
a new run, copy `outputs/` into `/kaggle/working` and run the notebook again: it resumes from `last`.
