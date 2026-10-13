"""Training a language model from scratch: tokenizer, model, trainer, sampler, checkpoints, the CLI and the chat backend."""
import json
import random

import numpy as np
import pytest

from quantum_mind.llm import BPETokenizer, library_corpus, read_corpus, read_qa

ENGLISH = ('the cat sat on the mat. the dog sat on the log. a cat and a dog sat together on the mat. ' * 30)


def test_the_tokenizer_is_lossless_compresses_and_saves(tmp_path):
    tok = BPETokenizer.train([ENGLISH + ' under_score x=3.14'], 300)
    assert tok.vocab_size <= 300 and tok.encode('') == []
    chars = [chr(c) for c in list(range(0, 300)) + [0x4e00, 0x1F642, 0x200b, 0xfeff]]
    rnd = random.Random(0)
    for _ in range(200):
        s = ''.join(rnd.choices(chars, k=rnd.randint(0, 50)))
        # any string, exactly (underscores, controls, emoji, other scripts)
        assert tok.decode(tok.encode(s)) == s
    assert len(ENGLISH) / len(tok.encode(ENGLISH)) > 2.5                   # real compression on the text it learned
    ids = tok.encode('hi', bos=True, eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id and tok.decode(ids) == 'hi' and '<bos>' in tok.decode(ids, skip_special=False)
    tok.save(tmp_path / 't.json')
    again = BPETokenizer.load(tmp_path / 't.json')
    assert again.encode(ENGLISH[:200]) == tok.encode(ENGLISH[:200]) and again.vocab_size == tok.vocab_size
    assert BPETokenizer.train([ENGLISH], 300).merges == BPETokenizer.train([ENGLISH], 300).merges            # deterministic


pytest.importorskip('jax')
pytest.importorskip('optax')
import jax.numpy as jnp  # noqa: E402

from quantum_mind.llm import ChatData, GPTConfig, LanguageModel, MixedData, TextData, train  # noqa: E402
from quantum_mind.llm.model import format_chat, loss_fn  # noqa: E402


def _tiny(tok, **kw):
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=48, n_layers=2, n_heads=4, block_size=32, **kw)
    return LanguageModel(cfg, tok, seed=0)


@pytest.fixture(scope='module')
def tok():
    return BPETokenizer.train([ENGLISH, 'What is the cat? A cat sat. What is the dog? A dog sat.'], 320)


def test_the_model_is_causal_and_saves_exactly(tok, tmp_path):
    m = _tiny(tok)
    a = np.random.default_rng(0).integers(5, tok.vocab_size, (2, 24)).astype(np.int32)
    b = a.copy()
    b[:, 12:] = (b[:, 12:] + 7) % tok.vocab_size
    la, lb = np.asarray(m.logits(a)), np.asarray(m.logits(b))
    assert np.allclose(la[:, :12], lb[:, :12], atol=1e-5) and not np.allclose(la[:, 12:], lb[:, 12:])      # the past never sees the future
    d, V, L = m.cfg.d_model, m.cfg.vocab_size, m.cfg.n_layers
    assert m.n_params == V * d + m.cfg.block_size * d + 2 * d + L * (2 * d + d * 3 * d + 3 * d + d * d + d + 2 * d + d * 4 * d +
    4 * d + 4 * d * d + d)
    m.save(tmp_path / 'm')
    m2 = LanguageModel.load(tmp_path / 'm')
    assert np.allclose(np.asarray(m2.logits(a)), la, atol=1e-6) and m2.cfg == m.cfg
    with pytest.raises(ValueError):
        GPTConfig(d_model=50, n_heads=4)


def test_the_loss_counts_only_masked_positions(tok):
    m = _tiny(tok)
    t = jnp.asarray(np.random.default_rng(1).integers(5, tok.vocab_size, (3, 17)).astype(np.int32))
    mask = np.zeros((3, 17), np.int32)
    mask[1, 9] = 1
    got = float(loss_fn(m.cfg, m.params, t, jnp.asarray(mask)))
    lg = np.asarray(m.logits(np.asarray(t[:, :-1])))[1, 8]
    want = -(lg - lg.max() - np.log(np.exp(lg - lg.max()).sum()))[int(t[1, 9])]
    assert got == pytest.approx(float(want), rel=1e-4)


def test_it_learns_a_text_and_continues_it(tok):
    m = _tiny(tok)
    data = TextData([ENGLISH] * 4, tok, 32)
    hist = train(m, data, steps=150, batch_size=16, lr=3e-3, warmup=10, log_every=50, eval_every=150, quiet=True)
    assert hist[-1]['loss'] < 1.0 and hist[-1]['val_loss'] < 1.0         # from chance (~ln 320) to the pattern
    assert m.perplexity(ENGLISH[:300]) < 3
    out = m.generate('the cat sat on the', temperature=0, max_new_tokens=8)
    assert out.strip().startswith('mat') or 'mat' in out[:12]


def test_it_learns_to_answer_from_question_answer_pairs(tok):
    pairs = [('What is the cat?', 'A cat sat.'), ('What is the dog?', 'A dog sat.')] * 20
    m = _tiny(tok)
    chat = ChatData(pairs, tok, 32)
    # the question is masked out of the loss
    assert all(np.sum(mk[:3]) == 0 and np.sum(mk) > 0 for _, mk in chat.train[:1])
    train(m, MixedData([(TextData([ENGLISH] * 4, tok, 32), 0.3), (chat, 0.7)]), steps=200, batch_size=16, lr=3e-3, warmup=10, quiet=True)
    assert m.chat([{'role': 'user', 'content': 'What is the cat?'}], temperature=0).strip() == 'A cat sat.'
    assert m.chat([{'role': 'user', 'content': 'What is the dog?'}], temperature=0).strip() == 'A dog sat.'
    ids = format_chat(tok, [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Hi'}])
    assert ids[0] == tok.bos_id and ids[-1] == tok.assistant_id and tok.user_id in ids and 'Be brief.' in tok.decode(ids)


def test_sampling_respects_top_k_seed_and_special_tokens(tok):
    m = _tiny(tok)
    prompt = tok.encode('the cat', bos=True)
    greedy = list(m.stream_ids(prompt, 10, temperature=0))
    assert list(m.stream_ids(prompt, 10, temperature=1.0, top_k=1)) == greedy
    assert list(m.stream_ids(prompt, 10, seed=3)) == list(m.stream_ids(prompt, 10, seed=3))
    # never writes <bos> etc.
    assert all(t >= tok.n_special for t in m.stream_ids(prompt, 30, temperature=1.5, top_k=0, top_p=1.0, seed=1))
    assert len(list(m.stream_ids(prompt, 5))) <= 5


def test_gradient_accumulation_and_resumable_checkpoints(tok, tmp_path):
    m = _tiny(tok)
    data = TextData([ENGLISH] * 4, tok, 32)
    h = train(m, data, steps=20, batch_size=8, accum=2, lr=3e-3, warmup=5, log_every=10, checkpoint_dir=tmp_path / 'ck',
    checkpoint_every=10,
              quiet=True)
    assert h[-1]['loss'] < h[0]['loss'] and (tmp_path / 'ck' / 'opt_state.npz').exists()
    assert json.loads((tmp_path / 'ck' / 'train_state.json').read_text())['step'] == 20
    m2 = LanguageModel.load(tmp_path / 'ck')
    h2 = train(m2, data, steps=30, batch_size=8, accum=2, lr=3e-3, warmup=5, log_every=10, checkpoint_dir=tmp_path / 'ck',
    resume=True, quiet=True)
    assert [r['step'] for r in h2] == [30] or h2[0]['step'] > 20                                      # continued at step 21, not from zero
    assert m2.step == 30


def test_bfloat16_compute_runs_and_stays_close(tok):
    m = _tiny(tok)
    t = np.random.default_rng(2).integers(5, tok.vocab_size, (2, 16)).astype(np.int32)
    ref = np.asarray(m.logits(t))
    cfg = GPTConfig(**{**m.cfg.__dict__, 'compute_dtype': 'bfloat16'})
    half = np.asarray(LanguageModel(cfg, tok, params=m.params).logits(t))
    # params stay float32, matmuls in bfloat16
    assert half.dtype == np.float32 and np.abs(half - ref).max() < 0.1


def test_the_command_line_and_the_chat_backend(tmp_path, capsys):
    from quantum_mind.chat import ChatError, ChatSession, ScratchBackend
    from quantum_mind.llm.cli import main
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'a.txt').write_text(ENGLISH)
    (tmp_path / 'qa.jsonl').write_text(json.dumps({'question': 'What is the cat?', 'answer': 'A cat sat.'}) + '\n')
    out = tmp_path / 'model'
    assert main(['train', '--data', str(tmp_path / 'data'), '--qa', str(tmp_path / 'qa.jsonl'), '--out', str(out), '--vocab',
    '300', '--steps', '6',
                 '--batch', '4', '--block', '32', '--log-every', '3', '--checkpoint-every', '3']) == 0
    assert all((out / f).exists() for f in ('config.json', 'tokenizer.json', 'weights.npz'))
    assert main(['info', '--checkpoint', str(out)]) == 0 and 'parameters' in capsys.readouterr().out
    assert main(['sample', '--checkpoint', str(out), '--prompt', 'the cat', '--tokens', '5', '--temperature', '0']) == 0
    assert main(['train', '--out', str(tmp_path / 'x')]) == 2                                         # nothing to train on
    session = ChatSession(ScratchBackend(str(out), temperature=0, max_new_tokens=6))
    assert isinstance(session.answer('What is the cat?'), str)
    with pytest.raises(ChatError, match='quantum-mind-llm train'):
        ScratchBackend(str(tmp_path / 'nothing'))


def test_the_corpus_readers_and_the_library_corpus(tmp_path):
    (tmp_path / 'a.md').write_text('hello')
    (tmp_path / 'b.bin').write_bytes(b'\x00\x01')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'c.py').write_text('print(1)')
    assert sorted(read_corpus(tmp_path)) == ['hello', 'print(1)']
    (tmp_path / 'q.jsonl').write_text(json.dumps({'messages': [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content':
    'b'}]}) + '\n')
    assert read_qa(tmp_path / 'q.jsonl')[0][1]['content'] == 'b'
    texts, train_pairs, held = library_corpus()
    assert len(texts) > 100 and len(train_pairs) > 300
    seen_q = {q.split(' ', 2)[-1].rstrip('?.') for q, _ in train_pairs}
    # held-out names never appear in training questions
    assert all(q.split(' ', 2)[-1].rstrip('?.') not in seen_q for q, _ in held['unseen_names'])
    assert not any('__main__' in t[:80] for t in texts)
