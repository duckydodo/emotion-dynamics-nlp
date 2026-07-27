# Gate 1 — Dataset Selection

## Decision

**Primary dataset:** MELD  
**Secondary / cross-domain dataset:** EmoWOZ  
**Not selected as primary:** DailyDialog

## Objective

The dataset-selection stage evaluated candidate conversational emotion datasets for a project focused on context-aware emotion recognition, emotion dynamics, and next-emotion forecasting.

The primary dataset should provide:

- dialogue-level context,
- sufficiently frequent emotional transitions,
- meaningful class diversity,
- non-trivial forecasting targets,
- speaker or interaction structure,
- manageable computational requirements.

## Candidate Datasets

### DailyDialog

DailyDialog contains:

- 13,118 dialogues,
- 102,979 utterances,
- seven emotion states including "no emotion".

Key findings:

- 83.10% of utterances are labelled no emotion.
- 47.62% of dialogues contain only no-emotion labels.
- 50.97% of dialogues contain zero emotion transitions.
- Mean transitions per dialogue: 1.23.
- Median transitions per dialogue: 0.
- 17.89% of forecasting targets involve an emotion change.
- Always predicting no emotion gives 82.35% forecasting accuracy.
- Predicting that the next emotion equals the current emotion gives 82.11%.

DailyDialog provides substantial scale, but its extreme label imbalance and low transition density make it poorly suited as the primary dataset for emotion-dynamics and forecasting experiments.

### MELD

The downloaded MELD annotations contain:

- 1,432 observed dialogues,
- 13,708 utterances,
- seven emotion classes,
- explicit speaker identities.

Key findings:

- Neutral represents 46.95% of utterances.
- 7.68% of dialogues are entirely neutral.
- 88.48% of dialogues contain at least one valid emotion change.
- Mean emotion changes per dialogue: 5.08.
- Median emotion changes per dialogue: 4.
- 59.64% of valid adjacent forecasting targets involve an emotion change.
- Always predicting neutral gives 46.70% forecasting accuracy.
- Predicting that the next emotion equals the current emotion gives 40.36%.

Speaker analysis found:

- 77.50% of valid adjacent pairs involve a speaker switch.
- Emotion changes occur in 44.25% of same-speaker adjacent pairs.
- Emotion changes occur in 64.11% of speaker-switch adjacent pairs.
- Mean speakers per dialogue: 2.73.
- Median speakers per dialogue: 2.

Data-quality inspection identified 69 dialogues with internal utterance-ID gaps, corresponding to 99 missing turn positions. Sequential experiments will only construct transitions between genuinely adjacent utterance IDs.

The distributed training CSV contains 1,038 unique dialogue IDs rather than the 1,039 stated in the supplied MELD documentation; Dialogue_ID 60 is absent. The utterance count still matches the documented 9,989 training utterances.

### EmoWOZ

The reconstructed EmoWOZ corpus contains:

- 11,433 unique dialogues,
- 83,617 emotion-labelled user turns,
- seven task-oriented emotion classes.

Key findings:

- Neutral represents 70.15% of user emotion labels.
- Satisfied represents 20.97%.
- Together, neutral and satisfied represent 91.12% of labels.
- 4.81% of dialogues are entirely neutral.
- 95.18% of dialogues contain at least one user-emotion change.
- Mean user-emotion changes per dialogue: 2.28.
- Median user-emotion changes per dialogue: 2.
- 36.18% of successive user-emotion forecasting targets involve a change.
- Always predicting neutral gives 66.22% forecasting accuracy.
- Predicting persistence gives 63.82%.

However, the transition structure is concentrated. Neutral-to-satisfied accounts for 49.60% of all changes, while satisfied-to-neutral accounts for 16.18%. Neutral/satisfied transitions therefore dominate the observed dynamics.

The raw DialMAGE component contains 849 empty textual turns: 848 system turns and one labelled user turn. Primary text-based forecasting experiments should exclude examples whose required current user, system-response, or target-user text is empty.

## Final Selection

MELD is selected as the primary dataset.

Although MELD is substantially smaller than DailyDialog and EmoWOZ, the project targets emotional dynamics rather than corpus scale alone. MELD provides:

1. substantially greater class balance,
2. frequent emotion changes,
3. non-trivial next-emotion forecasting,
4. diverse within-dialogue emotional trajectories,
5. explicit speaker identities,
6. multi-party conversational structure,
7. general conversational emotion categories.

These properties make MELD better aligned with the project's primary research questions.

EmoWOZ is retained as a potential secondary dataset for cross-domain validation. Its larger scale and user-system interaction structure provide a useful contrast to MELD, but its task-specific emotion ontology and concentration around neutral/satisfied interactions make it less suitable as the primary corpus.

DailyDialog is retained as a dataset-selection benchmark but will not be used as the primary modeling corpus.

## Gate 1 Status

**PASS — Primary dataset locked: MELD**

The project may now proceed to preprocessing, task formulation, baseline definition, and modeling.
