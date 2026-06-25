"""
NLP Visualization Lab — Python Backend
=======================================
Libraries used:
  • spaCy  (en_core_web_trf / lg / sm) — tokenisation, POS tagging, dependency
                              parsing, lemmatisation, named-entity recognition
  • NLTK   (WordNet)        — real word-sense definitions for the Semantic stage
  • Flask                   — lightweight REST API server

Quick start (choose the best model available):
  pip install -r requirements.txt

  # Best accuracy (transformer, ~500 MB, requires GPU or is slow on CPU):
  python -m spacy download en_core_web_trf

  # Good accuracy (~750 MB, fast on CPU):
  python -m spacy download en_core_web_lg

  # Minimum / fallback (~12 MB):
  python -m spacy download en_core_web_sm

  # Optional — neural coreference resolution:
  pip install coreferee
  python -m coreferee install en

  python app.py

Then open index.html in your browser (backend must stay running).
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
# INITIALISE spaCy  —  try best model first, fall back gracefully
# ══════════════════════════════════════════════════════════════════════════
_MODELS = ['en_core_web_trf', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
NLP        = None
MODEL_NAME = None

for _m in _MODELS:
    try:
        print(f"Trying spaCy model {_m} …", end=' ', flush=True)
        NLP        = spacy.load(_m)
        MODEL_NAME = _m
        print("✅")
        break
    except OSError:
        print("not installed, trying next…")

if NLP is None:
    print(
        "\n❌  No spaCy model found!\n"
        "    Install at least the small model:\n"
        "      python -m spacy download en_core_web_sm\n"
        "    For better accuracy:\n"
        "      python -m spacy download en_core_web_lg\n"
    )

# ── Optional: coreferee for neural coreference resolution ──────────────────
#    Install:  pip install coreferee
#              python -m coreferee install en
COREF_ENGINE = None
if NLP is not None:
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

# Pronoun sets for heuristic coreference — split by gender/number hints
PRONOUNS_NEUTER   = frozenset({'it', 'this', 'that'})
PRONOUNS_PLURAL   = frozenset({'they', 'them', 'these', 'those'})
PRONOUNS_MASC     = frozenset({'he', 'him', 'his'})
PRONOUNS_FEM      = frozenset({'she', 'her', 'hers'})
PRONOUNS_ALL      = PRONOUNS_NEUTER | PRONOUNS_PLURAL | PRONOUNS_MASC | PRONOUNS_FEM

# Typical male/female first-name signals (small heuristic set)
_MASC_NAMES = frozenset({
    'james','john','robert','michael','william','david','richard','joseph',
    'thomas','charles','christopher','daniel','matthew','anthony','mark',
    'donald','steven','paul','andrew','kenneth','george','josh','jack',
    'harry','henry','edward','alex','adam','ryan','jake','tom','peter',
    'raju','raj','amit','arjun','rohan','aman','vikram','nikhil','arun',
})
_FEM_NAMES  = frozenset({
    'mary','patricia','jennifer','linda','barbara','elizabeth','susan',
    'jessica','sarah','karen','lisa','nancy','betty','margaret','sandra',
    'ashley','emily','donna','michelle','carol','amanda','melissa','deborah',
    'stephanie','rebecca','laura','helen','sharon','cynthia','amy','anna',
    'priya','riya','neha','pooja','ananya','divya','kavya','shreya','sita',
    'meera','aisha','fatima','sara',
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
    Improved heuristic coreference resolution.

    Improvements over the original:
    • Gender-aware: 'he/him/his' prefers masculine names; 'she/her/hers'
      prefers feminine names; 'it/this/that' prefers inanimate nouns.
    • Also resolves within-sentence forward references (pronoun in same
      sentence as antecedent), not just cross-sentence.
    • Tracks a small window of recent nouns (up to 8) instead of the full
      history, so stale antecedents from many sentences back are ignored.
    • Avoids mapping a pronoun to itself.
    """
    chains  = []
    # Recent noun candidates — capped to a sliding window
    WINDOW  = 8
    nouns   = []   # {word, si, wi, plural, gender}  gender ∈ {m, f, n, ?}

    def _gender(tok_text, pos):
        """Guess grammatical gender from token text."""
        lo = tok_text.lower()
        if pos == 'PROP':
            if lo in _MASC_NAMES: return 'm'
            if lo in _FEM_NAMES:  return 'f'
        return 'n' if pos == 'NOUN' else '?'

    for si, sent in enumerate(sentences_data):
        for wi, tok in enumerate(sent['tokens']):
            pos  = tok['pos']
            norm = tok['lemma']

            # Accumulate nouns/proper-nouns as potential antecedents
            if pos in ('NOUN', 'PROP'):
                plural = (tok.get('morph_number') == 'Plur')
                nouns.append({
                    'word': tok['text'], 'si': si, 'wi': wi,
                    'plural': plural, 'gender': _gender(tok['text'], pos),
                })
                if len(nouns) > WINDOW:
                    nouns.pop(0)

            # Try to resolve pronouns
            if pos == 'PRON' and norm in PRONOUNS_ALL:
                candidate = None
                pool = [n for n in reversed(nouns)
                        if not (n['si'] == si and n['wi'] == wi)]  # exclude self

                if norm in PRONOUNS_NEUTER:
                    # prefer singular inanimate nouns
                    candidate = next(
                        (n for n in pool if not n['plural'] and n['gender'] in ('n', '?')),
                        next((n for n in pool if not n['plural']), None)
                    )
                elif norm in PRONOUNS_PLURAL:
                    # prefer plural nouns; fall back to any noun
                    candidate = next(
                        (n for n in pool if n['plural']),
                        next(iter(pool), None)
                    )
                elif norm in PRONOUNS_MASC:
                    # prefer masculine proper nouns, then any singular noun
                    candidate = next(
                        (n for n in pool if n['gender'] == 'm'),
                        next((n for n in pool if not n['plural']), None)
                    )
                elif norm in PRONOUNS_FEM:
                    candidate = next(
                        (n for n in pool if n['gender'] == 'f'),
                        next((n for n in pool if not n['plural']), None)
                    )

                if candidate:
                    chains.append({
                        'pronoun':    {'word': tok['text'],        'si': si,               'wi': wi},
                        'antecedent': {'word': candidate['word'],  'si': candidate['si'],  'wi': candidate['wi']},
                    })
    return chains


# ══════════════════════════════════════════════════════════════════════════
# PRAGMATIC ANALYSIS  (using spaCy parse + expanded rule set)
# ══════════════════════════════════════════════════════════════════════════

def analyze_pragmatics(doc, text):
    """
    Derive speech act and intent using the spaCy dependency parse plus an
    expanded, priority-ordered set of interpretive rules.

    Improvements over v1:
    - Greeting, farewell, gratitude, apology, offer speech acts
    - Sarcasm / irony detection (positive words + negative sentiment markers)
    - General complaint detection (not just environmental)
    - Conditional / hypothetical detection
    - Opinion / belief detection
    - Stronger confidence signals from parse features
    - Each check is a named Boolean — easy to extend
    """
    lo       = text.lower().strip()
    stripped = text.strip()

    # ── Structural parse signals ──────────────────────────────────────────
    non_space = [t for t in doc if not t.is_space and not t.is_punct]
    first_tok = non_space[0] if non_space else None
    lemmas    = {t.lemma_.lower() for t in doc}
    pos_tags  = {t.tag_ for t in doc}
    dep_set   = {t.dep_ for t in doc}

    is_modal_initial = bool(first_tok and first_tok.tag_ == 'MD')
    modal_word       = first_tok.text.lower() if is_modal_initial else ''
    has_neg          = 'neg' in dep_set
    has_past         = bool(pos_tags & {'VBD', 'VBN'})
    is_question      = stripped.endswith('?') or is_modal_initial
    is_excl          = stripped.endswith('!')

    # ── Greeting / Farewell ───────────────────────────────────────────────
    _greet    = {'hello', 'hi', 'hey', 'howdy', 'greetings',
                 'good morning', 'good afternoon', 'good evening', 'good day'}
    _farewell = {'bye', 'goodbye', 'farewell', 'see you', 'take care',
                 'good night', 'later', 'ciao', 'adios'}
    is_greeting = any(lo.startswith(g) for g in _greet) or lo in _greet
    is_farewell = any(lo.startswith(f) for f in _farewell) or lo in _farewell

    # ── Gratitude ─────────────────────────────────────────────────────────
    _thanks = {'thank', 'thanks', 'grateful', 'appreciate', 'gratitude', 'cheers'}
    is_gratitude = bool(lemmas & _thanks)

    # ── Apology ───────────────────────────────────────────────────────────
    _sorry = {'sorry', 'apologise', 'apologize', 'apology', 'forgive',
              'pardon', 'excuse', 'regret'}
    is_apology = bool(lemmas & _sorry)

    # ── Offer ─────────────────────────────────────────────────────────────
    _offer_pats = [r'\bshall i\b', r'\bwould you like\b',
                   r'\bcan i (get|bring|help)\b', r'\bdo you want\b', r'\blet me\b']
    is_offer = any(re.search(p, lo) for p in _offer_pats)

    # ── Polite indirect request ───────────────────────────────────────────
    polite_modals = {'can', 'could', 'would', 'will', 'shall'}
    is_polite = (
        is_modal_initial
        and modal_word in polite_modals
        and bool(re.match(r'^(can|could|would|will|shall)\s+you\b', lo))
    )

    # ── Sarcasm / irony ───────────────────────────────────────────────────
    _sarcasm_positive = {'obviously', 'clearly', 'sure', 'totally', 'great',
                         'wonderful', 'fantastic', 'brilliant', 'genius',
                         'right', 'perfect'}
    _negative_ctx     = {'not', 'never', 'no', "n't", 'hate', 'terrible',
                         'awful', 'worst', 'horrible', 'fail', 'wrong', 'bad'}
    is_sarcasm = bool((lemmas & _sarcasm_positive) and (lemmas & _negative_ctx))

    # ── Environmental / comfort complaint ─────────────────────────────────
    _env_words = {'hot', 'cold', 'warm', 'freezing', 'boiling', 'stuffy',
                  'humid', 'noisy', 'loud', 'quiet', 'dark', 'bright', 'stale'}
    is_env = bool(lemmas & _env_words) and not is_question

    # ── General complaint ─────────────────────────────────────────────────
    _complaint_words = {'annoying', 'frustrate', 'bother', 'upset', 'disappoint',
                        'unacceptable', 'terrible', 'awful', 'horrible', 'hate',
                        'sick', 'tired', 'ridiculous', 'absurd', 'outrageous'}
    is_complaint = bool(lemmas & _complaint_words) and not is_question

    # ── Rhetorical question ───────────────────────────────────────────────
    _rhetorical_pats = [
        r"don'?t you think", r"isn'?t it", r"right\?$", r"\bno\?$",
        r"who (doesn'?t|wouldn'?t|couldn'?t)\b", r"what'?s the point",
        r"why would (anyone|you)\b", r"does it (really|even) matter",
    ]
    is_rhetorical = bool(is_question and any(re.search(p, lo) for p in _rhetorical_pats))

    # ── Warning / directive ───────────────────────────────────────────────
    _warning_lemmas = {'careful', 'beware', 'stop', 'never', 'watch',
                       'danger', 'avoid', 'warn', 'caution', 'alert'}
    is_warning = bool(
        (has_neg and first_tok and first_tok.tag_ == 'VB')
        or (lemmas & _warning_lemmas)
    )

    # ── Conditional / hypothetical ────────────────────────────────────────
    _cond_pats = [r'\bif\b', r'\bunless\b', r'\bsuppose\b',
                  r'\bassume\b', r'\bimagine\b', r'\bwhat if\b']
    is_conditional = any(re.search(p, lo) for p in _cond_pats)

    # ── Opinion / belief ─────────────────────────────────────────────────
    _opinion_lemmas = {'think', 'believe', 'feel', 'reckon', 'suppose',
                       'guess', 'seem', 'appear', 'consider', 'find'}
    is_opinion = bool(lemmas & _opinion_lemmas) and not is_question

    # ── Past narrative ────────────────────────────────────────────────────
    is_narrative = has_past and not is_question

    # ══ Priority-ordered speech act selection ═════════════════════════════
    if is_greeting:
        return {
            'speechAct': 'Greeting', 'icon': '👋',
            'literal':   'An opening social phrase used to acknowledge the listener.',
            'intended':  'Establishing or maintaining a friendly relationship; '
                         'no specific informational content is expected in return.',
            'confidence': 96,
        }
    if is_farewell:
        return {
            'speechAct': 'Farewell', 'icon': '🤝',
            'literal':   'A closing social phrase signalling the end of an interaction.',
            'intended':  'Politely ending the conversation; may carry warmth or goodwill.',
            'confidence': 95,
        }
    if is_gratitude:
        return {
            'speechAct': 'Expression of Gratitude', 'icon': '🙌',
            'literal':   'Acknowledging a benefit received from the listener.',
            'intended':  'Strengthening social bonds; acknowledgement ("you\'re welcome") '
                         'is the typical expected response.',
            'confidence': 94,
        }
    if is_apology:
        return {
            'speechAct': 'Apology', 'icon': '🙇',
            'literal':   'Admitting fault or expressing regret for a past action.',
            'intended':  'Repairing a social breach and seeking forgiveness or understanding.',
            'confidence': 93,
        }
    if is_sarcasm:
        return {
            'speechAct': 'Sarcasm / Irony', 'icon': '😏',
            'literal':   'Uses positive or affirming language on the surface.',
            'intended':  'The true meaning is the opposite — expressing criticism, '
                         'frustration, or mockery through deliberate overstatement.',
            'confidence': 72,
        }
    if is_rhetorical:
        return {
            'speechAct': 'Rhetorical Question', 'icon': '🎭',
            'literal':   'Grammatically a question, but no direct answer is expected.',
            'intended':  'Used to assert a point strongly, express emotion, or persuade — '
                         'the answer is already implied by context.',
            'confidence': 84,
        }
    if is_offer:
        return {
            'speechAct': 'Offer / Proposal', 'icon': '🤲',
            'literal':   'Proposing to do something for the benefit of the listener.',
            'intended':  'Signalling helpfulness or generosity; listener may accept or decline.',
            'confidence': 88,
        }
    if is_polite:
        return {
            'speechAct': 'Indirect Request', 'icon': '🙏',
            'literal':   f'Asking if the listener is able or willing to act (using "{modal_word}").',
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
    if is_complaint:
        return {
            'speechAct': 'Complaint', 'icon': '😤',
            'literal':   'Expressing dissatisfaction or displeasure about something.',
            'intended':  'Seeking acknowledgement, sympathy, or corrective action from '
                         'the listener.',
            'confidence': 81,
        }
    if is_warning:
        return {
            'speechAct': 'Warning / Directive', 'icon': '⚠️',
            'literal':   'Explicitly stating a prohibition, risk, or cautionary condition.',
            'intended':  'Cautioning the listener; urging them to take or avoid a specific '
                         'action for their own safety or benefit.',
            'confidence': 88,
        }
    if is_conditional:
        return {
            'speechAct': 'Conditional / Hypothetical', 'icon': '🔀',
            'literal':   'Presenting a scenario dependent on a condition being true.',
            'intended':  'Exploring possibilities, negotiating, or setting expectations — '
                         'the outcome is not stated as fact.',
            'confidence': 83,
        }
    if is_excl:
        return {
            'speechAct': 'Exclamation', 'icon': '😲',
            'literal':   'An emphatic statement conveying strong feeling.',
            'intended':  'Sharing heightened emotion — surprise, excitement, frustration '
                         'or urgency.',
            'confidence': 84,
        }
    if is_question:
        return {
            'speechAct': 'Direct Question', 'icon': '❓',
            'literal':   'Requesting specific information, confirmation or clarification.',
            'intended':  'Genuine inquiry — the listener is expected to provide a direct '
                         'response.',
            'confidence': 90,
        }
    if is_opinion:
        return {
            'speechAct': 'Opinion / Belief', 'icon': '💭',
            'literal':   'Sharing a personal view or judgement rather than an objective fact.',
            'intended':  'Inviting agreement, discussion, or respectful disagreement from '
                         'the listener.',
            'confidence': 79,
        }
    if is_narrative:
        return {
            'speechAct': 'Narrative / Report', 'icon': '📖',
            'literal':   'Recounting a sequence of past events in order.',
            'intended':  'Informing the listener; may implicitly seek empathy or a '
                         'follow-up reaction.',
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
        'model':    MODEL_NAME or 'none',
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
            'model':    MODEL_NAME or 'unknown',
            'coref':    COREF_ENGINE or 'heuristic',
        },
    })


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n🚀  NLP Lab backend running at http://localhost:5000")
    print("   Open nlp-lab.html in your browser (backend must stay running)\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
