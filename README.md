# Week Plan: Merge ABD Branch, Pretrain on FNSPID, and Start Fine-Tuning

# Weekly Objectives

## First Objective: Merge ABD Branch

By the end of this objective, the main branch should no longer rely on one-hot inputs. It should use token IDs, trainable token embeddings, positional embeddings, batching, attention masks, MLM masking, validation metrics, and masked-word prediction.

## Second Objective: Streamline FNSPID Data Processing

By the end of this objective, we should have a pipeline that loads FNSPID or a sample of FNSPID, extracts usable article text or text fields, and converts the result into the text format expected by the pretraining code.

## Third Objective: Run Initial Pretraining

By the end of this objective, we should have trained the model under a few parameter settings and selected the best initial setup based on validation loss.

## Fourth Objective: Start Fine-Tuning

By the end of this objective, we should decide the most suitable fine-tuning task and build a first working fine-tuning prototype.

---

# Day 10: Understand ABD and Begin Merge

## Task 0: Replace One-Hot Inputs with Trainable Token Embeddings

**Owner:** Covered through Task 1 and Task 4  
**Estimated time:** 0 additional hours as a standalone task

### Description

Confirm that the final merged code removes the old one-hot representation and replaces it with trainable token embeddings.

The old main branch manually creates one-hot vectors for each token and sets:

```python
d_model = vocab_size
```

This is not ideal because the embedding size becomes tied to the vocabulary size. In a real transformer, tokens should first become integer IDs, and those IDs should be mapped to dense learnable vectors using:

```python
nn.Embedding(vocab_size, d_model)
```

This allows `d_model` to be a design choice such as 32, 64, 128, or 256, instead of being forced to equal the number of vocabulary tokens.

### Measurable Completion Criteria

```text
- No active training path uses one-hot vectors as transformer input.
- Tokens are represented as integer IDs.
- The model contains nn.Embedding(vocab_size, d_model).
- The model accepts input_ids with shape [batch_size, seq_len].
- d_model is independent of vocab_size.
- CrossEntropyLoss receives class-ID targets, not one-hot vectors.
```

### Reference

```text
PyTorch nn.Embedding:
https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
```

---

## Task 0.5: Add Positional Embeddings Using nn.Embedding

**Owner:** Covered through Task 1 and Task 4  
**Estimated time:** 0 additional hours as a standalone task

### Description

Confirm that the final merged model adds position information to token embeddings.

Self-attention can compare tokens to other tokens, but by itself it does not know the order of tokens. For example, without position information, these sentences are difficult to distinguish:

```text
Company beat expectations.
Expectations beat company.
```

The ABD branch solves this using positional embeddings:

```python
self.position_embedding = nn.Embedding(max_len, d_model)
```

During the forward pass, position IDs are created:

```python
pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
```

Then token embeddings and positional embeddings are added:

```python
x = self.token_embedding(input_ids) + self.position_embedding(pos)
```

### Measurable Completion Criteria

```text
- The model defines self.position_embedding = nn.Embedding(max_len, d_model).
- The forward pass creates position IDs with torch.arange(t).
- The position tensor is shaped correctly for batching.
- Token embeddings and positional embeddings are added before the transformer layers.
- The resulting tensor has shape [batch_size, seq_len, d_model].
```

### Reference

```text
Attention Is All You Need:
https://arxiv.org/abs/1706.03762
```

---

## Task 1: Understand ABD Blocks 13–19 and Write an Explanation Report

**Owner:** Person A  
**Estimated time:** 2 hours

### Description

Study the ABD branch section starting from `TransformerLanguageModel` and continuing through the final masked-word prediction function.

This includes:

```text
- TransformerLanguageModel
- token embeddings
- positional embeddings
- transformer layer loop
- vocabulary classifier head
- make_mlm_batch
- get_metrics
- CrossEntropyLoss(ignore_index=-100)
- AdamW optimizer
- run_epoch
- training history
- validation loop
- predict_masked_words
```

The deliverable should be a short report or mini-presentation that Person B can read to fully understand this part before validating the merge.

The report should explain the workflow:

```text
text
→ tokens
→ token IDs
→ token embeddings + positional embeddings
→ transformer layers
→ vocab logits
→ MLM loss
→ backward pass
→ optimizer update
→ validation metrics
→ masked-word prediction
```

### Measurable Completion Criteria

```text
- Report explains TransformerLanguageModel.__init__.
- Report explains TransformerLanguageModel.forward.
- Report explains why input_ids have shape [batch_size, seq_len].
- Report explains why embeddings have shape [batch_size, seq_len, d_model].
- Report explains why logits have shape [batch_size, seq_len, vocab_size].
- Report explains make_mlm_batch.
- Report explains why labels use -100.
- Report explains CrossEntropyLoss(ignore_index=-100).
- Report explains get_metrics: loss, top-1, top-5, perplexity.
- Report explains run_epoch for both training and validation.
- Report explains predict_masked_words.
- Report identifies which code should be merged into main.
- Report flags any suspicious, unclear, or AI-heavy code.
```

### References

```text
PyTorch nn.Module:
https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html

PyTorch nn.Embedding:
https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html

PyTorch CrossEntropyLoss:
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

BERT paper for Masked Language Modeling:
https://arxiv.org/abs/1810.04805
```

---

## Task 2: Merge ABD Blocks 1–12 into Main Branch

**Owner:** Person B  
**Estimated time:** 2 hours

### Description

Merge the ABD branch code before `TransformerLanguageModel` into the main branch.

This includes:

```text
- imports
- seed setup
- device setup
- configuration variables
- toy train/validation texts
- special tokens
- basic_tokenize
- build_vocab
- encode_text
- TextDataset
- DataLoader
- improved TransformerLayer
```

The purpose is to replace the older one-hot-based preprocessing and early transformer components in the main branch with the cleaner ABD implementation.

This should not be a blind copy-paste. The code should be inserted into the main notebook in a clean order and should preserve the main project narrative.

### Measurable Completion Criteria

```text
- Main branch has clean imports.
- Main branch has seed setup for reproducibility.
- Main branch has a clear configuration/settings section.
- Special tokens [PAD], [UNK], [CLS], [SEP], [MASK] are defined once.
- basic_tokenize works and preserves [MASK].
- build_vocab creates both stoi and itos.
- encode_text returns input_ids and attention_mask.
- TextDataset returns torch.long tensors.
- DataLoader produces batches of shape [batch_size, max_len].
- TransformerLayer accepts x and attention_mask.
- The old one-hot preprocessing path is removed or clearly deprecated.
```

### References

```text
PyTorch Dataset and DataLoader tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

PyTorch tensor documentation:
https://docs.pytorch.org/docs/stable/tensors.html

Hugging Face tokenizer summary:
https://huggingface.co/docs/transformers/en/tokenizer_summary
```

---

# Day 11: Validate First Merge and Merge Model/Training Logic

## Task 3: Validate Merged Blocks 1–12 and Debug

**Owner:** Person A  
**Estimated time:** 1 hour

### Description

After Person B merges blocks 1–12, Person A should independently validate that the preprocessing and transformer-layer foundation works.

This validation must happen before merging blocks 13–19 because bugs in tokenization, vocabulary, encoding, dataset creation, batching, or attention masking can cause later training bugs that are harder to diagnose.

### Measurable Completion Criteria

```text
- Notebook/script runs from imports through DataLoader creation.
- basic_tokenize("[MASK]") preserves the mask token.
- build_vocab returns valid stoi and itos dictionaries.
- encode_text returns exactly max_len IDs.
- encode_text returns exactly max_len attention-mask values.
- batch["input_ids"].shape is [batch_size, max_len].
- batch["attention_mask"].shape is [batch_size, max_len].
- TransformerLayer forward pass works on fake embeddings.
- Attention mask correctly blocks padding tokens.
- Any bugs are fixed or documented with exact error messages.
```

### Suggested Smoke Test

```python
batch = next(iter(train_loader))
print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
```

---

## Task 4: Merge ABD Blocks 13–19

**Owner:** Person A  
**Estimated time:** 1 hour

### Description

Merge the ABD code from `TransformerLanguageModel` through `predict_masked_words`.

This includes:

```text
- TransformerLanguageModel
- token_embedding
- position_embedding
- transformer layer stack
- classifier head
- make_mlm_batch
- get_metrics
- loss function
- optimizer
- run_epoch
- training loop
- validation loop
- history tracking
- predict_masked_words
```

The model should be integrated into the main notebook in a clean order:

```text
1. imports
2. configuration
3. data setup
4. tokenizer/vocabulary
5. dataset/dataloader
6. model definitions
7. MLM utilities
8. training/evaluation
9. prediction
```

### Measurable Completion Criteria

```text
- TransformerLanguageModel is added to main.
- The model defines token_embedding and position_embedding.
- The model uses a ModuleList or equivalent layer stack.
- The model returns logits with shape [batch_size, seq_len, vocab_size].
- make_mlm_batch returns masked_ids and labels.
- Labels use -100 for non-masked positions.
- CrossEntropyLoss(ignore_index=-100) is used.
- run_epoch supports training=True and training=False.
- Training history is stored.
- predict_masked_words returns top-k candidate tokens for [MASK].
```

### Important Merge Rule

```text
Do not keep both the old one-hot Transformer path and the new TransformerLanguageModel path as active competing implementations.

The main runnable path should use TransformerLanguageModel.
```

---

## Task 5: Validate Merged Blocks 13–19 and Debug

**Owner:** Person B  
**Estimated time:** 2 hours

### Description

Person B should validate the code that Person A merged. This should happen after Person B reads Person A’s Task 1 explanation report.

The validation should test the whole toy MLM flow end-to-end:

```text
build toy dataset
→ build vocabulary
→ create DataLoader
→ initialize model
→ create masked batches
→ train for at least one epoch
→ run validation
→ predict masked words
```

This task is intentionally assigned to the person who did not merge the code, because independent validation catches more mistakes.

### Measurable Completion Criteria

```text
- Full notebook/script runs top-to-bottom on toy texts.
- One training epoch completes without runtime errors.
- Validation loss is printed.
- top-1 and top-5 metrics are printed.
- predict_masked_words("Ahmed is a [MASK] math student.", top_k=5) runs.
- No CrossEntropyLoss target shape error occurs.
- No device mismatch error occurs.
- No attention-mask broadcasting error occurs.
- Any bugs are fixed or documented with exact error messages.
```

### References

```text
PyTorch saving/loading models:
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

PyTorch CrossEntropyLoss:
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
```

---

# Day 12: Build FNSPID-to-Text Pipeline

## Task 6: Write Code to Load FNSPID and Extract Links

**Owner:** Person A  
**Estimated time:** 1.25 hours

### Description

Write the first version of the FNSPID loading function.

The function should load a sample of the dataset, inspect the available columns, identify the column containing article URLs/links if present, and return a clean list or dataframe of candidate links.

The code should not assume a column name before inspecting the dataset. It should print the column names and then select the correct column based on what exists.

### Measurable Completion Criteria

```text
- FNSPID sample loads without crashing.
- Code prints the dataset column names.
- Code prints the number of rows loaded.
- Code identifies the likely URL/link column.
- Missing or empty links are dropped.
- Duplicate links are removed.
- Code prints the number of non-empty unique links.
- Code prints the first 5 links.
- Output is either a list of URLs or a dataframe with a URL column.
```

### References

```text
Hugging Face tabular dataset loading:
https://huggingface.co/docs/datasets/tabular_load

FNSPID Hugging Face dataset:
https://huggingface.co/datasets/Zihan1004/FNSPID

Hugging Face datasets loading guide:
https://huggingface.co/docs/datasets/loading
```

---

## Task 7: Write Code to Extract Article Text from a Link

**Owner:** Person B  
**Estimated time:** 1 hour

### Description

Write a function that takes one article URL and attempts to extract readable article text.

The first version does not need to perfectly scrape every website. It only needs to work on some links and fail safely on others.

The function should:

```text
- request the page
- parse HTML
- remove scripts/styles/navigation noise
- extract paragraph text
- return one clean string
```

If the page cannot be loaded or parsed, the function should return `None` and log the failed URL instead of crashing the whole pipeline.

### Measurable Completion Criteria

```text
- Function accepts one URL string.
- Function returns a string of extracted article text or None.
- Function handles timeout errors.
- Function handles HTTP errors.
- Function removes obvious HTML/script/style noise.
- Function logs failed URLs separately.
- At least 10 sample links are tested.
- The number of successful and failed extractions is printed.
```

### Suggested Libraries

```text
requests
beautifulsoup4
trafilatura
newspaper3k
```

### Important Note

Some financial news websites may block scraping, require JavaScript, or restrict automated access. The extraction function should fail gracefully and should not assume every link can be extracted.

---

## Task 8: Convert Extracted Text into the Expected Model Format

**Owner:** Person A  
**Estimated time:** 0.75 hour

### Description

Convert the extracted article texts into the format expected by the current pretraining code.

The current pretraining code expects a Python list of strings:

```python
texts = [
    "Tesla reported strong revenue this quarter.",
    "Apple reported weak guidance despite strong revenue."
]
```

This task should produce a variable such as:

```python
texts_for_pretraining = [...]
```

Each item should be either one full article or one article chunk.

If articles are very long, splitting them into chunks is better than simply truncating everything at `max_len`, because truncation wastes most of the article.

### Measurable Completion Criteria

```text
- Output variable is clearly named, such as texts_for_pretraining.
- Output is a Python list of strings.
- Empty strings are removed.
- Very short texts are removed, for example fewer than 30 words.
- Duplicated texts are removed if possible.
- Very long articles are either chunked or clearly marked for later chunking.
- The output can be passed directly into build_vocab and TextDataset.
```

### Recommended Decision

```text
Use chunking for long articles instead of relying only on truncation.
```

---

## Task 9: Streamline the Full Pretraining Pipeline

**Owner:** Person B  
**Estimated time:** 1 hour

### Description

Connect the FNSPID text output from Tasks 6–8 into the existing pretraining code.

The code should allow switching between toy texts and FNSPID texts using one configuration flag or variable.

The objective is to avoid manually rewriting notebook cells every time the data source changes.

### Measurable Completion Criteria

```text
- One variable controls the data source: "toy" or "fnspid_sample".
- The selected texts are split into train_texts and val_texts.
- build_vocab uses train_texts only.
- TextDataset and DataLoader work with the selected texts.
- One small pretraining run works using the selected data source.
- Main parameters are configurable at the top:
  - max_len
  - batch_size
  - d_model
  - d_internal
  - num_layers
  - lr
  - num_epochs
  - mlm_probability
```

### Recommended Code Structure

```text
1. imports
2. config
3. data loading
4. tokenization/vocab
5. dataset/dataloader
6. model
7. MLM utilities
8. training/evaluation
9. prediction
```

---

# Day 13: Run Initial Pretraining Experiments

## Task 10: Find the Best Initial Parameter Setup and Visualize Loss

**Owner:** Person A and Person B  
**Estimated time:** 4 hours total

### Description

Run a small structured parameter search for pretraining.

The goal is not to find the perfect model. The goal is to identify a reasonable first setup with the lowest validation loss among the tested options.

Because the team only has four total hours on Day 13, keep the FNSPID sample small enough that each experiment finishes.

### Suggested Experiment Grid

```text
Run 1:
d_model = 32
d_internal = 64
num_layers = 2
lr = 1e-4

Run 2:
d_model = 64
d_internal = 128
num_layers = 2
lr = 1e-4

Run 3:
d_model = 64
d_internal = 128
num_layers = 3
lr = 1e-4

Run 4:
d_model = 64
d_internal = 128
num_layers = 2
lr = 3e-4
```

### Person A Subtask

```text
- Prepare experiment configurations.
- Run at least two experiments.
- Record train loss, validation loss, top-1, and top-5.
```

### Person B Subtask

```text
- Run at least two experiments.
- Create loss visualizations.
- Build the comparison table.
```

### Measurable Completion Criteria

```text
- At least 3 parameter setups are tested.
- Each setup records train loss.
- Each setup records validation loss.
- Each setup records top-1 MLM accuracy if available.
- Each setup records top-5 MLM accuracy if available.
- A loss plot is created.
- A comparison table is created.
- The best setup is selected using validation loss, not training loss.
```

### Required Output Table

```text
experiment_id
d_model
d_internal
num_layers
lr
batch_size
max_len
num_epochs
train_loss
val_loss
top1
top5
notes
```

### References

```text
PyTorch AdamW:
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

PyTorch gradient clipping:
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html

Matplotlib pyplot:
https://matplotlib.org/stable/api/pyplot_summary.html
```

---

# Day 14: Choose Fine-Tuning Task and Build First Prototype

## Task 11: Determine the Fine-Tuning Task That Best Fits the Project

**Owner:** Person A  
**Estimated time:** 1.5 hours

### Description

Decide the best first fine-tuning task for the project.

The original project goal is to use financial news for fundamental analysis and stock-related signals. The current architecture is encoder-only, which fits classification and representation-learning tasks better than generation tasks.

Compare the following options:

```text
Option 1: financial sentiment classification
Option 2: stock movement direction prediction
Option 3: narrative stress classification
Option 4: summarization
Option 5: event type classification
```

### Recommended Direction

Start with either:

```text
financial sentiment classification
```

or:

```text
narrative stress classification
```

Do not start with summarization unless the architecture changes, because summarization is usually a generation task and is less aligned with the current encoder-only model.

### Measurable Completion Criteria

```text
- A one-page decision note is written.
- The chosen fine-tuning task is clearly stated.
- The note explains why the task fits the current encoder-only transformer.
- The note explains why the rejected options are less suitable for this week.
- The note defines the labels/classes.
- The note identifies what labeled data is needed.
- The note defines the evaluation metric.
- The note explains how the fine-tuning task supports the larger project goal.
```

### References

```text
BERT paper for encoder classification fine-tuning:
https://arxiv.org/abs/1810.04805

Hugging Face sequence classification guide:
https://huggingface.co/docs/transformers/tasks/sequence_classification

Financial PhraseBank dataset:
https://huggingface.co/datasets/takala/financial_phrasebank
```

---

## Task 12: Complete First Fine-Tuning Prototype

**Owner:** Person B, with review from Person A  
**Estimated time:** 2.5 hours

### Description

Build a minimal supervised fine-tuning pipeline using the pretrained encoder.

This should not aim to be a final research-quality fine-tuned model. The goal is to prove that the pretrained transformer can be adapted to a downstream classification task.

The simplest flow is:

```text
financial text
→ encode text
→ pass through pretrained embedding + transformer encoder
→ take [CLS] representation or mean-pooled representation
→ pass through classification head
→ predict sentiment/narrative class
```

The classification model should use a new head:

```python
self.classifier = nn.Linear(d_model, num_classes)
```

The MLM vocabulary-prediction head should not be used for fine-tuning classification, because the MLM head predicts words while the fine-tuning head predicts task labels.

### Measurable Completion Criteria

```text
- A new FineTuningClassifier class or equivalent is created.
- The pretrained embedding and transformer layers are reused.
- The model uses the [CLS] token representation or mean pooling.
- The output shape is [batch_size, num_classes].
- CrossEntropyLoss is used for classification.
- One tiny labeled dataset is created or loaded.
- Training runs for at least one epoch.
- Validation accuracy or macro-F1 is printed.
- The prototype can make a prediction on one new financial sentence.
```

### Minimum Toy Dataset Acceptable for Prototype

```text
"Company reported strong revenue growth." → positive
"Company warned about weak demand." → negative
"Company held its annual meeting." → neutral
```

### Better Dataset If Time Allows

```text
Financial PhraseBank:
https://huggingface.co/datasets/takala/financial_phrasebank
```

### References

```text
Hugging Face sequence classification guide:
https://huggingface.co/docs/transformers/tasks/sequence_classification

Financial PhraseBank:
https://huggingface.co/datasets/takala/financial_phrasebank

PyTorch CrossEntropyLoss:
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

Scikit-learn classification metrics:
https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
```

---

# Final Weekly Schedule

| Day | Person A | Person B |
|---|---|---|
| Day 10 | Task 1: understand blocks 13–19 and write report | Task 2: merge blocks 1–12 |
| Day 11 | Task 3: validate blocks 1–12; Task 4: merge blocks 13–19 | Task 5: validate blocks 13–19 |
| Day 12 | Task 6: load FNSPID links; Task 8: convert text format | Task 7: extract text from links; Task 9: streamline pipeline |
| Day 13 | Task 10: run experiments and compare | Task 10: run experiments and visualize |
| Day 14 | Task 11: choose fine-tuning task | Task 12: build first fine-tuning prototype |

---

# Total Time Estimate

| Task Group | Estimated Time |
|---|---:|
| Day 10 merge/study | 4 hours |
| Day 11 validation/merge | 4 hours |
| Day 12 FNSPID pipeline | 4 hours |
| Day 13 pretraining experiments | 4 hours |
| Day 14 fine-tuning decision/prototype | 4 hours |
| **Total** | **20 person-hours** |

---

# Final Success Criteria

By the end of Day 14, the week is successful if:

```text
1. Main branch no longer relies on one-hot transformer inputs.
2. Main branch has token embeddings.
3. Main branch has positional embeddings.
4. Main branch can run MLM training end-to-end on toy data.
5. Main branch can run MLM training on at least a small FNSPID-derived text sample.
6. Validation loss and top-k MLM metrics are recorded.
7. At least three pretraining configurations are compared.
8. One fine-tuning task is selected with justification.
9. A minimal classification fine-tuning prototype runs.
```

---
