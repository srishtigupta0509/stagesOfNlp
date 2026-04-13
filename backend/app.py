"""
NLP Visualization Lab — Python Backend
=======================================
Libraries used:
  • spaCy  (en_core_web_sm) — tokenisation, POS tagging, dependency parsing,
                              lemmatisation, named-entity recognition (NER)
  • NLTK   (WordNet)        — real word-sense definitions for the Semantic stage
  • Flask                   — lightweight REST API server

Quick start:
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  python app.py

Then open nlp-lab.html in your browser.
"""

import re
import sys

# ── Flask ──────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── spaCy ──────────────────────────────────────────────────────────────────
import spacy

# ── NLTK / WordNet ─────────────────────────────────────────────────────────
import nltk
# Auto-download required NLTK data (first run only)
for pkg in ('wordnet', 'omw-1.4'):
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        print(f"  Downloading NLTK '{pkg}'…")
        nltk.download(pkg, quiet=True)

from nltk.corpus import wordnet

# ══════════════════════════════════════════════════════════════════════════
# INITIALISE spaCy
# ══════════════════════════════════════════════════════════════════════════
print("Loading spaCy model en_core_web_sm …", end=' ', flush=True)
try:
    NLP = spacy.load('en_core_web_sm')
    print("✅")
except OSError:
    print("❌\n\nModel not found! Run:\n  python -m spacy download en_core_web_sm\n")
    NLP = None

# ── Optional: coreferee for neural coreference resolution ──────────────────
#    Install:  pip install coreferee
#              python -m coreferee install en
COREF_ENGINE = None
try:
    NLP.add_pipe('coreferee')
    COREF_ENGINE = 'coreferee'
    print("✅  coreferee loaded — using neural coreference")
except Exception as e:
    print(f"ℹ️   coreferee not available ({e}); using heuristic coreference instead")

# ══════════════════════════════════════════════════════════════════════════
# CONSTANT MAPPINGS
# ══════════════════════════════════════════════════════════════════════════

# spaCy Universal POS  →  frontend display label
UPOS_MAP = {
    'NOUN':  'NOUN',
    'VERB':  'VERB',
    'AUX':   'AUX',
    'ADJ':   'ADJ',
    'ADV':   'ADV',
    'PRON':  'PRON',
    'DET':   'DET',
    'ADP':   'PREP',   # adposition (preposition / postposition)
    'CCONJ': 'CONJ',
    'SCONJ': 'CONJ',
    'PROPN': 'PROP',   # proper noun
    'NUM':   'NUM',
    'PUNCT': 'PUNCT',
    'PART':  'PART',
    'INTJ':  'INTJ',
    'SYM':   'SYM',
    'X':     'WORD',
    'SPACE': None,
}

# spaCy Penn Treebank fine-grained tag overrides
FINE_TAG_MAP = {
    'MD': 'MODAL',  # modal auxiliary: can, could, will, would …
    'WP': 'Q-WH',   # wh-pronoun: who, what
    'WRB': 'Q-WH',  # wh-adverb: where, when, why, how
    'WDT': 'Q-WH',  # wh-determiner: which, what
}

# Dependency label  →  human-readable role name
DEP_ROLE = {
    'nsubj':     'Subject',
    'nsubjpass': 'Subject (passive)',
    'ROOT':      'Main Verb',
    'dobj':      'Object',
    'obj':       'Object',
    'iobj':      'Indirect Object',
    'attr':      'Attribute',
    'aux':       'Aux / Modal',
    'auxpass':   'Aux (passive)',
    'neg':       'Negation',
    'amod':      'Adj Modifier',
    'advmod':    'Adv Modifier',
    'prep':      'Prep Phrase',
    'det':       'Article',
    'pobj':      'Prep Object',
    'cc':        'Coordinator',
    'conj':      'Conjunction',
    'compound':  'Compound',
    'poss':      'Possessive',
    'relcl':     'Relative Clause',
    'ccomp':     'Clause',
    'xcomp':     'Open Clause',
    'mark':      'Subordinator',
    'expl':      'Expletive',
}

# Roles we want to surface in the Syntactic visualisation
DISPLAY_DEPS = frozenset({
    'nsubj', 'nsubjpass', 'ROOT',
    'dobj', 'obj', 'iobj', 'attr',
    'aux', 'neg', 'amod', 'advmod',
})

# Pronoun set for heuristic coreference
PRONOUNS = frozenset({
    'it', 'he', 'she', 'they',
    'this', 'that', 'these', 'those',
    'him', 'her', 'them',
})

# spaCy UPOS  →  WordNet POS constant
UPOS_TO_WN = {
    'NOUN':  wordnet.NOUN,
    'PROPN': wordnet.NOUN,
    'VERB':  wordnet.VERB,
    'AUX':   wordnet.VERB,
    'ADJ':   wordnet.ADJ,
    'ADV':   wordnet.ADV,
}


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def display_pos(token):
    """Map a spaCy token to our frontend POS label (MODAL, PROP, NOUN …)."""
    if token.tag_ in FINE_TAG_MAP:
        return FINE_TAG_MAP[token.tag_]
    return UPOS_MAP.get(token.pos_, 'WORD') or 'WORD'


def detect_sentence_type(sent):
    """
    Determine sentence type using spaCy's dependency parse — much more
    reliable than surface-form pattern matching.
    """
    text = sent.text.strip()
    if text.endswith('?'):
        return 'Question'
    if text.endswith('!'):
        return 'Exclamation'

    non_space = [t for t in sent if not t.is_space]
    if not non_space:
        return 'Statement'

    first = non_space[0]

    # Modal-initial inversion → question or polite request
    if first.tag_ == 'MD':
        return 'Question / Request'

    # Auxiliary-initial inversion → question ("Is she …?", "Are you …?")
    if first.pos_ == 'AUX':
        return 'Question'

    # Imperative: ROOT is a base-form verb with no explicit subject
    roots = [t for t in sent if t.dep_ == 'ROOT']
    if roots:
        root = roots[0]
        has_subject = any(t.dep_ in ('nsubj', 'nsubjpass') for t in sent)
        if root.tag_ == 'VB' and not has_subject:
            return 'Command / Imperative'

    return 'Statement'


def wordnet_meanings(lemma, spacy_pos):
    """
    Query NLTK WordNet for up to 3 sense definitions.
    Returns [] if the word has no WordNet entry.
    """
    wn_pos = UPOS_TO_WN.get(spacy_pos)
    synsets = wordnet.synsets(lemma, pos=wn_pos) if wn_pos else wordnet.synsets(lemma)
    if not synsets:
        return []

    results = []
    for i, ss in enumerate(synsets[:3]):
        # Make a readable sense label from the synset name, e.g. "window.n.01" → "Noun #1"
        parts = ss.name().split('.')
        pos_letter = parts[1] if len(parts) > 1 else '?'
        label_map = {'n': 'Noun', 'v': 'Verb', 'a': 'Adj', 's': 'Adj', 'r': 'Adv'}
        label = label_map.get(pos_letter, pos_letter.upper()) + f' #{i + 1}'
        results.append({'sense': label, 'def': ss.definition()})
    return results


def token_sent_word_idx(token, sent_boundaries):
    """
    Given a flat token index (token.i in doc) and a list of
    (start_idx, end_idx) per sentence, return (si, wi_in_sentence).
    """
    for si, (start, end) in enumerate(sent_boundaries):
        if start <= token.i < end:
            wi = token.i - start
            return si, wi
    return 0, 0


# ══════════════════════════════════════════════════════════════════════════
# COREFERENCE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════

def resolve_coref_neural(doc, sent_boundaries):
    """
    Use coreferee (neural) to resolve coreference chains.
    Returns list of {pronoun: {word, si, wi}, antecedent: {word, si, wi}}.
    """
    chains = []
    try:
        for chain in doc._.coref_chains:
            mentions = list(chain)
            if len(mentions) < 2:
                continue
            # First mention = antecedent, subsequent = pronouns/references
            ant_idx = mentions[0][0]
            ant_tok = doc[ant_idx]
            ant_si, ant_wi = token_sent_word_idx(ant_tok, sent_boundaries)

            for mention in mentions[1:]:
                pro_idx = mention[0]
                pro_tok = doc[pro_idx]
                pro_si, pro_wi = token_sent_word_idx(pro_tok, sent_boundaries)
                chains.append({
                    'pronoun':    {'word': pro_tok.text, 'si': pro_si, 'wi': pro_wi},
                    'antecedent': {'word': ant_tok.text, 'si': ant_si, 'wi': ant_wi},
                })
    except Exception as e:
        print(f"  [coreferee] {e}")
    return chains


def resolve_coref_heuristic(sentences_data):
    """
    Heuristic coreference: map singular 'it/this/that' to the most recent
    singular noun from an earlier sentence; plural to most recent noun.
    Not perfect — but deterministic and transparent.
    """
    chains = []
    nouns = []  # accumulated {word, si, wi, plural}

    for si, sent in enumerate(sentences_data):
        for wi, tok in enumerate(sent['tokens']):
            pos  = tok['pos']
            norm = tok['lemma']

            if pos in ('NOUN', 'PROP'):
                plural = (tok.get('morph_number') == 'Plur')
                nouns.append({'word': tok['text'], 'si': si, 'wi': wi, 'plural': plural})

            if pos == 'PRON' and norm in PRONOUNS:
                candidate = None
                if norm in ('it', 'this', 'that'):
                    # most recent singular noun from a previous sentence
                    candidate = next(
                        (n for n in reversed(nouns) if n['si'] < si and not n['plural']),
                        None
                    )
                elif norm in ('they', 'them', 'these', 'those'):
                    candidate = next((n for n in reversed(nouns) if n['si'] < si), None)
                elif norm in ('he', 'him', 'his', 'she', 'her', 'hers'):
                    candidate = next((n for n in reversed(nouns) if n['si'] < si), None)

                if candidate:
                    chains.append({
                        'pronoun':    {'word': tok['text'],        'si': si,               'wi': wi},
                        'antecedent': {'word': candidate['word'],  'si': candidate['si'],  'wi': candidate['wi']},
                    })
    return chains


# ══════════════════════════════════════════════════════════════════════════
# PRAGMATIC ANALYSIS  (using spaCy parse, not raw regex)
# ══════════════════════════════════════════════════════════════════════════

def analyze_pragmatics(doc, text):
    """
    Derive speech act and intent using the spaCy dependency parse
    plus a small set of interpretive rules.
    """
    lo = text.lower().strip()

    # ── Detect initial modal via POS ──────────────────────────────────────
    non_space = [t for t in doc if not t.is_space and not t.is_punct]
    first_tok  = non_space[0] if non_space else None
    is_modal_initial = first_tok and first_tok.tag_ == 'MD'
    modal_word = first_tok.text.lower() if is_modal_initial else ''

    # Polite request: "Can/Could/Would/Will you …"
    polite_modals = {'can', 'could', 'would', 'will', 'shall'}
    is_polite = (is_modal_initial
                 and modal_word in polite_modals
                 and re.match(r'^(can|could|would|will|shall)\s+you\b', lo))

    # Environmental complaint: describes temperature / comfort
    env_words = {'hot', 'cold', 'warm', 'freezing', 'boiling',
                 'stuffy', 'humid', 'noisy', 'dark', 'bright'}
    is_env = any(t.lemma_ in env_words for t in doc) and not text.strip().endswith('?')

    # Rhetorical: "right?", "isn't it?", "don't you think?"
    is_rhetorical = bool(re.search(r"don't you think|isn't it|right\?$|no\?$", lo))

    # Warning / directive: uses negation with imperative, or safety vocab
    warning_lemmas = {'careful', 'beware', 'stop', 'never'}
    has_neg = any(t.dep_ == 'neg' for t in doc)
    is_warning = has_neg or any(t.lemma_ in warning_lemmas for t in doc)

    # Exclamation
    is_excl = text.strip().endswith('!')

    # Past-tense narrative: has a VBD/VBN verb
    has_past = any(t.tag_ in ('VBD', 'VBN') for t in doc)
    is_question = text.strip().endswith('?') or is_modal_initial

    is_narrative = has_past and not is_question

    # ── Select speech act ─────────────────────────────────────────────────
    if is_polite:
        return {
            'speechAct': 'Indirect Request', 'icon': '🙏',
            'literal':  f'Asking if the listener is able or willing to act (using "{modal_word}").',
            'intended':  'A face-saving, polite request — the speaker wants the action done, '
                         'not a literal answer about the listener\'s ability.',
            'confidence': 93,
        }
    if is_env:
        return {
            'speechAct': 'Indirect Complaint / Request', 'icon': '🌡️',
            'literal':   'Describing a physical or environmental condition as a plain fact.',
            'intended':  'Implying dissatisfaction and expecting a corrective action '
                         '(e.g., open a window, adjust the thermostat).',
            'confidence': 85,
        }
    if is_rhetorical:
        return {
            'speechAct': 'Rhetorical Question', 'icon': '🎭',
            'literal':   'Grammatically a question, but no direct answer is expected.',
            'intended':  'Used to assert a point strongly, express emotion, or persuade — '
                         'the answer is already implied by context.',
            'confidence': 82,
        }
    if is_warning:
        return {
            'speechAct': 'Warning / Directive', 'icon': '⚠️',
            'literal':   'Explicitly stating a prohibition or condition.',
            'intended':  'Cautioning the listener; urging them to take or avoid a specific action.',
            'confidence': 88,
        }
    if is_excl:
        return {
            'speechAct': 'Exclamation', 'icon': '😲',
            'literal':   'An emphatic statement conveying strong feeling.',
            'intended':  'Sharing heightened emotion — surprise, excitement, frustration or urgency.',
            'confidence': 84,
        }
    if is_question:
        return {
            'speechAct': 'Direct Question', 'icon': '❓',
            'literal':   'Requesting specific information, confirmation or clarification.',
            'intended':  'Genuine inquiry — the listener is expected to provide a direct response.',
            'confidence': 90,
        }
    if is_narrative:
        return {
            'speechAct': 'Narrative / Report', 'icon': '📖',
            'literal':   'Recounting a sequence of past events in order.',
            'intended':  'Informing the listener; may implicitly seek empathy or a follow-up reaction.',
            'confidence': 80,
        }
    return {
        'speechAct': 'Declarative Statement', 'icon': '💬',
        'literal':   'Asserting a fact, opinion or belief as true.',
        'intended':  'Direct communication — literal and intended meanings closely align.',
        'confidence': 76,
    }


# ══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)   # allow the HTML file (file:// or any origin) to call the API


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'spacy':    NLP is not None,
        'model':   'en_core_web_sm',
        'coref':    COREF_ENGINE,
        'nltk_wn':  True,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    POST body:  { "text": "your sentence here" }
    Returns:    full NLP analysis for all 5 stages
    """
    if NLP is None:
        return jsonify({'error': 'spaCy model not loaded. '
                                 'Run: python -m spacy download en_core_web_sm'}), 503

    body = request.get_json(force=True, silent=True) or {}
    text = (body.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # ── spaCy parse ───────────────────────────────────────────────────────
    doc = NLP(text)

    # Pre-compute sentence boundaries as flat token index ranges
    sent_boundaries = []
    for sent in doc.sents:
        sent_boundaries.append((sent.start, sent.end))

    # ── Build per-sentence token lists ────────────────────────────────────
    sentences_data = []
    for sent in doc.sents:
        tokens = []
        for tok in sent:
            if tok.is_space:
                continue
            morph   = tok.morph.to_dict()
            pos_lbl = display_pos(tok)
            tokens.append({
                'text':         tok.text,
                'lemma':        tok.lemma_.lower(),
                'pos':          pos_lbl,
                'pos_fine':     tok.tag_,       # Penn Treebank tag (MD, NN, VBD …)
                'dep':          tok.dep_,
                'dep_role':     DEP_ROLE.get(tok.dep_, tok.dep_),
                'head':         tok.head.text,
                'is_stop':      tok.is_stop,
                'is_punct':     tok.is_punct,
                'ent_type':     tok.ent_type_ or None,  # NER label
                'morph_number': morph.get('Number', ''),
                'morph_tense':  morph.get('Tense', ''),
                'morph_mood':   morph.get('Mood', ''),
            })

        sentences_data.append({
            'text':   sent.text.strip(),
            'type':   detect_sentence_type(sent),
            'tokens': tokens,
        })

    # ── Syntactic roles for first sentence ────────────────────────────────
    first_tokens = sentences_data[0]['tokens'] if sentences_data else []
    roles = []
    for tok in first_tokens:
        if tok['is_punct'] or tok['dep'] not in DISPLAY_DEPS:
            continue
        roles.append({
            'word':     tok['text'],
            'role':     DEP_ROLE.get(tok['dep'], tok['dep']),
            'pos':      tok['pos'],
            'dep':      tok['dep'],
        })
    # Fallback: if parse gave nothing useful, surface the first few tokens
    if not roles:
        roles = [
            {'word': t['text'], 'role': t['dep_role'], 'pos': t['pos'], 'dep': t['dep']}
            for t in first_tokens[:6] if not t['is_punct']
        ]

    # ── Named entities (NER) ──────────────────────────────────────────────
    entities = [
        {
            'text':        ent.text,
            'label':       ent.label_,
            'description': spacy.explain(ent.label_) or ent.label_,
        }
        for ent in doc.ents
    ]

    # ── WordNet word-sense meanings ────────────────────────────────────────
    # Collect meanings for every unique non-stop noun / verb / adjective lemma
    word_meanings = {}
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        if tok.pos_ not in ('NOUN', 'PROPN', 'VERB', 'AUX', 'ADJ'):
            continue
        lemma = tok.lemma_.lower()
        if lemma in word_meanings:
            continue   # already computed
        defs = wordnet_meanings(lemma, tok.pos_)
        if defs:
            word_meanings[lemma] = defs

    # ── Coreference resolution ─────────────────────────────────────────────
    if COREF_ENGINE == 'coreferee':
        coref_chains = resolve_coref_neural(doc, sent_boundaries)
    else:
        coref_chains = resolve_coref_heuristic(sentences_data)

    # ── Pragmatic analysis ─────────────────────────────────────────────────
    pragmatic = analyze_pragmatics(doc, text)

    # ── Build and return response ──────────────────────────────────────────
    return jsonify({
        'sentences':    sentences_data,
        'roles':        roles,
        'entities':     entities,
        'word_meanings': word_meanings,
        'coref_chains': coref_chains,
        'sent_type':    sentences_data[0]['type'] if sentences_data else 'Statement',
        'pragmatic':    pragmatic,
        'model_info': {
            'library': 'spaCy + NLTK WordNet',
            'model':   'en_core_web_sm',
            'coref':   COREF_ENGINE or 'heuristic',
        },
    })


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n🚀  NLP Lab backend running at http://localhost:5000")
    print("   Open nlp-lab.html in your browser (backend must stay running)\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
