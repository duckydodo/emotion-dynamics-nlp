# Context-Aware Emotion Dynamics and Forecasting

An NLP project investigating how conversational context and speaker information affect emotion recognition and next-emotion forecasting in dialogue.

## Dataset Selection

Three candidate datasets were evaluated:

- DailyDialog
- MELD
- EmoWOZ

MELD was selected as the primary dataset based on its emotion diversity, transition density, forecasting difficulty, and explicit multi-speaker structure.

See `docs/gate1_dataset_selection.md` for the dataset-selection analysis.

## Project Structure

- `notebooks/` — exploratory analysis and experiments
- `src/` — reusable preprocessing, modeling, evaluation, and analysis code
- `docs/` — project decisions and methodology
- `outputs/` — generated figures, tables, and model artifacts
- `data/` — local datasets and processed data; excluded from Git
