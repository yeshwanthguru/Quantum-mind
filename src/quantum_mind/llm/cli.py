"""The ``quantum-mind-llm`` command: train a language model from scratch on your own text, sample from it, inspect it.

::

    quantum-mind-llm train --data my_texts/ --out my_model --preset small --steps 20000            # plain text, any folder of .txt/.md/.py
    # text + question-answer pairs
    quantum-mind-llm train --data my_texts/ --qa pairs.jsonl --out my_model --dtype bfloat16 --batch 32 --accum 4
    quantum-mind-llm train --library --out lib_model --preset tiny --steps 3000                     # this library's own docstrings
    quantum-mind-llm train --out my_model --resume --steps 40000 --data my_texts/                  # continue from a checkpoint
    quantum-mind-llm sample --checkpoint my_model --prompt "Once upon a time"
    quantum-mind-llm sample --checkpoint my_model --chat --prompt "What does COBRA do?"
    quantum-mind-llm info --checkpoint my_model
    quantum-mind-chat --backend scratch --checkpoint my_model                                      # talk to it

On a GPU install the CUDA build of JAX first (``pip install "jax[cuda12]"``); the same command then runs there. Question-
answer files are JSON
lines: ``{"question": "...", "answer": "..."}`` or ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus import library_corpus, read_corpus, read_qa
from .tokenizer import BPETokenizer


def parser():
    p = argparse.ArgumentParser(prog='quantum-mind-llm', description='Train and use a language model built from scratch.')
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train', help='train a model on your text (and optional question-answer pairs)')
    t.add_argument('--data', nargs='*', default=[], help='files or folders of text (.txt .md .rst .py .tex .csv)')
    t.add_argument('--qa', help='JSON-lines file of question-answer pairs (trained on the answers only)')
    t.add_argument('--qa-weight', type=float, default=0.3, help='share of batches drawn from the question-answer pairs')
    t.add_argument('--library', action='store_true', help="use this library's own docstrings as the corpus and question-answer pairs")
    t.add_argument('--out', required=True, help='directory for the model (config, tokenizer, weights, checkpoints)')
    t.add_argument('--preset', choices=('tiny', 'small', 'base'), default='tiny')
    t.add_argument('--vocab', type=int, default=2048, help='tokenizer vocabulary size (default 2048; 8192 to 32000 for large corpora)')
    t.add_argument('--block', type=int, default=None, help='context length (default: the preset)')
    t.add_argument('--dropout', type=float, default=0.0)
    t.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32', help='compute dtype; bfloat16 is fast on a GPU')
    t.add_argument('--steps', type=int, default=2000)
    t.add_argument('--batch', type=int, default=32, help='sequences per micro-batch (what fits in memory)')
    t.add_argument('--accum', type=int, default=1, help='micro-batches per optimiser step (effective batch = batch x accum)')
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--warmup', type=int, default=100)
    t.add_argument('--eval-every', type=int, default=250)
    t.add_argument('--log-every', type=int, default=50)
    t.add_argument('--checkpoint-every', type=int, default=500)
    t.add_argument('--resume', action='store_true', help='continue from the checkpoint in --out')
    t.add_argument('--seed', type=int, default=0)
    s = sub.add_parser('sample', help='write text from a trained model')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--prompt', default='')
    s.add_argument('--chat', action='store_true', help='treat the prompt as a question and use the chat format')
    s.add_argument('--tokens', type=int, default=150)
    s.add_argument('--temperature', type=float, default=0.8)
    s.add_argument('--top-k', type=int, default=40)
    s.add_argument('--top-p', type=float, default=0.95)
    s.add_argument('--seed', type=int, default=0)
    i = sub.add_parser('info', help='size and settings of a trained model')
    i.add_argument('--checkpoint', required=True)
    return p


def _train(a, out):
    from .model import LanguageModel
    from .trainer import ChatData, MixedData, TextData, train
    texts, pairs = read_corpus(a.data), []
    if a.library:
        lib_texts, pairs, _ = library_corpus()
        texts += lib_texts
    if a.qa:
        pairs += read_qa(a.qa)
    if not texts and not pairs:
        out.write('error: give --data, --qa or --library\n')
        return 2
    out_dir = Path(a.out)
    tok_path = out_dir / 'tokenizer.json'
    if a.resume and tok_path.exists():
        tok = BPETokenizer.load(tok_path)
    else:
        tok = BPETokenizer.train(texts + [q + '\n' + ans for q, ans in pairs if isinstance(q, str)], a.vocab)
        out.write('tokenizer: %d tokens (%.2f characters per token)\n' %
                  (tok.vocab_size, sum(map(len, texts[:50])) / max(1, sum(len(tok.encode(t)) for t in texts[:50]))))
    over = dict(dropout=a.dropout, compute_dtype=a.dtype)
    if a.block:
        over['block_size'] = a.block
    if a.resume and (out_dir / 'weights.npz').exists():
        model = LanguageModel.load(a.out)
        model.cfg.dropout, model.cfg.compute_dtype = a.dropout, a.dtype
    else:
        model = LanguageModel.new(tok, a.preset, seed=a.seed, **over)
    block = model.cfg.block_size
    sources = []
    if texts:
        sources.append((TextData(texts, tok, block, seed=a.seed), 1.0 - (a.qa_weight if pairs else 0.0)))
    if pairs:
        sources.append((ChatData(pairs, tok, block, seed=a.seed), a.qa_weight if texts else 1.0))
    data = sources[0][0] if len(sources) == 1 else MixedData(sources)
    out_dir.mkdir(parents=True, exist_ok=True)
    train(model, data, steps=a.steps, batch_size=a.batch, accum=a.accum, lr=a.lr, warmup=a.warmup, eval_every=a.eval_every,
    log_every=a.log_every,
          checkpoint_dir=a.out, checkpoint_every=a.checkpoint_every, resume=a.resume, seed=a.seed)
    model.save(a.out)
    out.write('saved %s (%d parameters)\n' % (a.out, model.n_params))
    return 0


def main(argv=None, stdout=None):
    a = parser().parse_args(argv)
    out = stdout or sys.stdout
    try:
        if a.cmd == 'train':
            return _train(a, out)
        from .model import LanguageModel
        model = LanguageModel.load(a.checkpoint)
        if a.cmd == 'info':
            c = model.cfg
            out.write('%d parameters | vocabulary %d | d_model %d | layers %d | heads %d | context %d | trained %d steps\n' %
                      (model.n_params, c.vocab_size, c.d_model, c.n_layers, c.n_heads, c.block_size, model.step))
            return 0
        kw = dict(max_new_tokens=a.tokens, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, seed=a.seed)
        if a.chat:
            out.write(model.chat([{'role': 'user', 'content': a.prompt}], **kw) + '\n')
        else:
            out.write(a.prompt + model.generate(a.prompt, **kw) + '\n')
        return 0
    except (FileNotFoundError, ValueError) as e:
        out.write('error: %s\n' % e)
        return 2


if __name__ == '__main__':
    sys.exit(main())
