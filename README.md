# English → French translation: from a character RNN to a Transformer (PyTorch)

Three steps toward a neural translator, each built from scratch in PyTorch and tracked in
Weights & Biases:

1. **Character-level sequence-to-sequence RNN**, as a baseline.
2. **Subword tokenisation** with SentencePiece (BPE, 5,000 tokens per language), trained on a
   10,000-sentence sample of an English–French corpus of Canadian government web pages.
3. **Encoder–decoder Transformer** (`nn.Transformer`, multi-head attention), with greedy and
   top-p (nucleus) sampling decoding.

## Results

Validation, Transformer with 3 encoder and 3 decoder layers, embeddings of 192, 50 epochs
(two runs, see [`wandb_export_*.csv`](wandb_export_2025-01-13T23_45_28.098+01_00.csv)):

| | Loss | Top-1 token accuracy | Top-5 token accuracy |
|---|---:|---:|---:|
| Training | 3.32 | 36.4% | 61.0% |
| Validation | 3.74 | 35.8% | 58.7% |

Training and validation stay close, so the model is not overfitting: it is limited by capacity
and data. It has learned the vocabulary and local statistics of French, but full-sentence
translations are not yet usable. A larger run (6 layers, embeddings of 512) crashed after 49 of
50 epochs at a training loss of 1.40, which points to more capacity as the next step, along with
more than 10,000 training sentences and a BLEU evaluation.

## Files

- `MACHINE_TRANSLATION.ipynb`: the three models, training loops, decoding (notebook in French)
- `spm_en.*`, `spm_fr.*`: trained SentencePiece tokenisers
- `en_sub.txt`, `fr_sub.txt`: the 10,000 aligned sentences used to train the tokenisers
- `compte-rendu.pdf`: report (French); `sujet.pdf`: assignment

```bash
pip install torch sentencepiece wandb pandas matplotlib
jupyter notebook MACHINE_TRANSLATION.ipynb
```

Course project, Bachelor's degree in AI, Université Côte d'Azur (January 2025).
