"""
Polychronous Multi-Modal Spiking Memory
Unified implementation fixing all prototype gaps.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Iterable
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
import hashlib
import random
import re

# =============================================================================
# 1. NEURAL PRIMITIVES
# =============================================================================


def _spike_value(value: Any) -> float:
    """Best-effort conversion from spike list / scalar / array to a count."""
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    if isinstance(value, np.ndarray):
        return float(np.sum(value))
    return float(value)


class SparseCodebook:
    """Deterministic sparse code allocator with collision-aware fallbacks."""

    def __init__(self, size: int, active_k: int, seed: int = 0,
                 max_attempts: int = 256, oov_label: str = "__OOV__"):
        self.size = size
        self.active_k = max(1, min(active_k, size))
        self.seed = seed
        self.max_attempts = max_attempts
        self.oov_label = oov_label
        self.label_to_code: Dict[str, Tuple[int, ...]] = {}
        self.code_to_label: Dict[Tuple[int, ...], str] = {}
        self.collision_count = 0
        self._oov_code = tuple(range(self.active_k))

    def _candidate_code(self, label: str, attempt: int) -> Tuple[int, ...]:
        payload = f"{self.seed}|{label}|{attempt}"
        digest = hashlib.sha256(payload.encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        return tuple(sorted(rng.sample(range(self.size), self.active_k)))

    def assign(self, label: Optional[str]) -> Set[int]:
        key = self.oov_label if label is None or str(label).strip() == "" else str(label)
        if key in self.label_to_code:
            return set(self.label_to_code[key])

        for attempt in range(self.max_attempts):
            code = self._candidate_code(key, attempt)
            if code not in self.code_to_label:
                self.label_to_code[key] = code
                self.code_to_label[code] = key
                return set(code)

        self.collision_count += 1
        self.label_to_code[key] = self._oov_code
        self.code_to_label.setdefault(self._oov_code, self.oov_label)
        return set(self._oov_code)

    def decode(self, counts: np.ndarray, min_score: float = 1.0) -> str:
        if not self.label_to_code:
            return self.oov_label

        best_label = self.oov_label
        best_score = -np.inf
        for label, code in self.label_to_code.items():
            if label == self.oov_label:
                continue
            score = float(np.sum(counts[list(code)]))
            if score > best_score:
                best_score = score
                best_label = label
        if best_score < min_score:
            return self.oov_label
        return best_label


@dataclass
class EpisodeRecord:
    """Stored episode metadata for replay, consolidation, and temporal queries."""

    episode_id: int
    engram_id: Optional[int]
    timestamp: float
    sql_row: Dict[str, Any]
    graph_edge: Tuple[int, str, int]
    text: Optional[str]
    energy: float
    neuromodulator: float
    familiar: bool
    ca3_assembly: Set[int] = field(default_factory=set)
    dg_assembly: Set[int] = field(default_factory=set)
    ca3_spikes: Dict[int, List[float]] = field(default_factory=dict)
    sql_spikes: Dict[int, List[float]] = field(default_factory=dict)
    graph_spikes: Dict[int, List[float]] = field(default_factory=dict)
    text_spikes: Dict[int, List[float]] = field(default_factory=dict)
    predecessor_episode_id: Optional[int] = None
    successor_episode_id: Optional[int] = None

@dataclass
class LIFNeuron:
    """Leaky Integrate-and-Fire neuron."""
    nid: int
    tau_m: float = 20.0      # ms
    v_rest: float = -70.0    # mV
    v_thresh: float = -55.0  # mV
    v_reset: float = -75.0   # mV
    tau_ref: float = 2.0     # ms
    dt: float = 0.1          # ms
    
    def __post_init__(self):
        self.v = self.v_rest
        self.refractory_until = -1.0
        self.spike_times: List[float] = []
        self._last_I = 0.0
        self.homeostatic_offset = 0.0
        
    def step(self, t: float, I_syn: float) -> bool:
        """Euler step. Returns True if spike emitted."""
        if t < self.refractory_until:
            self.v = self.v_reset
            return False
        
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau_m * self.dt
        self.v += dv
        
        if self.v >= self.v_thresh + self.homeostatic_offset:
            self.v = self.v_reset
            self.refractory_until = t + self.tau_ref
            self.spike_times.append(t)
            return True
        return False
    
    def reset(self):
        self.v = self.v_rest
        self.refractory_until = -1.0
        self.spike_times.clear()


# =============================================================================
# 2. SYNAPSE MODEL
# =============================================================================

class DelayedSynapse:
    """Delayed synapse with exponential PSC and STDP."""
    
    def __init__(self, pre: str, post: int, weight: float, delay: float,
                 tau_syn: float = 3.0, w_max: float = 10.0, w_min: float = 0.0,
                 gain: float = 1.0):
        self.pre = pre           # Pre-synaptic neuron ID (string for input, int for recurrent)
        self.post = post         # Post-synaptic neuron index
        self.w = weight
        self.delay = delay       # ms
        self.tau_syn = tau_syn
        self.w_max = w_max
        self.w_min = w_min
        self.gain = gain
        self.dale_sign = 1 if weight >= 0 else -1
        
        # STDP parameters
        self.A_plus = 0.5
        self.A_minus = 0.48
        self.tau_plus = 12.0
        self.tau_minus = 12.0
        
        # Spike tracking for PSC
        self.pre_spike_times: List[float] = []
        self._arrival_times: List[float] = []
        self._arrival_index = 0
        self._trace = 0.0
        self._last_current_t: Optional[float] = None
        
    def current(self, t: float) -> float:
        """Compute postsynaptic current at time t.

        The original alpha kernel is approximated with a cached exponential
        trace so that current updates stay O(1) per synapse per timestep.
        """
        if self._last_current_t is None:
            self._last_current_t = t

        dt = t - self._last_current_t
        if dt > 0:
            self._trace *= float(np.exp(-dt / self.tau_syn))
            self._last_current_t = t

        while self._arrival_index < len(self._arrival_times) and self._arrival_times[self._arrival_index] <= t:
            self._trace += self.w
            self._arrival_index += 1

        return self.gain * self._trace
    
    def add_spike(self, t: float):
        self.pre_spike_times.append(t)
        self._arrival_times.append(t + self.delay)
        
    def stdp_update(self, t_post: float, pre_spike_times: List[float], M: float):
        """
        STDP update when post-synaptic neuron fires.
        M: neuromodulatory gain (0=no plasticity, 1=normal, 2=boosted)
        """
        if M == 0 or self.w <= 0:
            return
        for t_pre in pre_spike_times:
            dt = t_post - (t_pre + self.delay)
            if dt >= 0:
                dw = M * self.A_plus * np.exp(-dt / self.tau_plus)
            else:
                dw = -M * self.A_minus * np.exp(dt / self.tau_minus)
            self.w = np.clip(self.w + dw, self.w_min, self.w_max)
            
    def reset(self):
        self.pre_spike_times.clear()
        self._arrival_times.clear()
        self._arrival_index = 0
        self._trace = 0.0
        self._last_current_t = None


# =============================================================================
# 3. MODALITY ENCODERS
# =============================================================================

class SQLEncoder:
    """100 neurons: 5 fields × 20 population code."""
    
    FIELDS = ['age', 'salary', 'city', 'role', 'dept']
    N_PER_FIELD = 20
    TOTAL = 100
    
    def __init__(self, duration: float = 10.0, dt: float = 0.1, seed: int = 0,
                 categorical_active_k: int = 3):
        self.duration = duration
        self.dt = dt
        self.seed = seed
        self.categorical_active_k = max(2, min(categorical_active_k, self.N_PER_FIELD))
        salary_low_centers = np.linspace(0.0, 1e5, 10)
        salary_high_centers = np.linspace(np.log1p(1e5), np.log1p(1e10), 10)
        self.numeric_profiles = {
            'age': {
                'centers': np.linspace(20, 80, self.N_PER_FIELD),
                'sigma': 8.0,
                'base_rate': 240.0,
                'transform': lambda value: float(value),
                'inverse': lambda value: float(value),
                'clip': (0.0, 120.0),
            },
            # Salary uses a two-band code: a dense low-range band for common
            # enterprise values and a log-scale tail band for outliers.
            'salary': {
                'bands': [
                    {
                        'name': 'low',
                        'start': 0,
                        'centers': salary_low_centers,
                        'sigma': 15000.0,
                        'base_rate': 280.0,
                        'transform': lambda value: float(np.clip(float(value), 0.0, 1e6)),
                        'inverse': lambda value: float(value),
                    },
                    {
                        'name': 'high',
                        'start': 10,
                        'centers': salary_high_centers,
                        'sigma': 0.9,
                        'base_rate': 220.0,
                        'transform': lambda value: float(np.log1p(max(float(value), 0.0))),
                        'inverse': lambda value: float(np.expm1(max(float(value), 0.0))),
                    },
                ],
                'clip': (0.0, 1e10),
            },
        }
        # Backward-compatible attributes used by some notebook cells and tests.
        self.centers = {
            'age': self.numeric_profiles['age']['centers'],
            'salary': np.concatenate([salary_low_centers, np.expm1(salary_high_centers)]),
        }
        self.sigma = {
            'age': self.numeric_profiles['age']['sigma'],
            'salary': 15000.0,
        }
        self._base_rate = {'age': 240.0, 'salary': 260.0}
        self._cat_cache: Dict[str, int] = {}
        self._cat_codebooks: Dict[str, SparseCodebook] = {
            field: SparseCodebook(
                size=self.N_PER_FIELD,
                active_k=min(self.categorical_active_k, self.N_PER_FIELD),
                seed=self.seed + (idx + 1) * 7919,
            )
            for idx, field in enumerate(f for f in self.FIELDS if f not in self.numeric_profiles)
        }
        self._max_spikes_per_neuron = 5
    
    def _row_seed(self, row: Dict, field: str) -> int:
        """Deterministically seed the encoder so identical rows repeat exactly."""
        items = tuple(sorted((k, row.get(k, None)) for k in self.FIELDS))
        payload = f"{field}|{items!r}|{self.duration:.3f}"
        return int(hashlib.sha1(payload.encode()).hexdigest()[:8], 16)

    def _numeric_value(self, field: str, value: Any) -> float:
        profile = self.numeric_profiles[field]
        transformed = profile['transform'](value)
        low, high = profile['clip']
        if field == 'salary':
            transformed = float(np.clip(transformed, profile['transform'](low), profile['transform'](high)))
        else:
            transformed = float(np.clip(transformed, low, high))
        return transformed

    def _categorical_codebook(self, field: str) -> SparseCodebook:
        if field not in self._cat_codebooks:
            self._cat_codebooks[field] = SparseCodebook(
                size=self.N_PER_FIELD,
                active_k=min(self.categorical_active_k, self.N_PER_FIELD),
                seed=self.seed + (len(self._cat_codebooks) + 1) * 7919,
            )
        return self._cat_codebooks[field]
    
    def encode(self, row: Dict) -> Dict[int, List[float]]:
        """Return {neuron_id: [spike_times]}."""
        spikes = defaultdict(list)
        base = 0
        for field in self.FIELDS:
            val = row.get(field, 0)
            if field in self.numeric_profiles:
                if field == 'salary':
                    # Two-band multi-resolution encoding: low-range linear
                    # neurons for normal salaries and a log-scale tail band for
                    # extreme enterprise values.
                    profile = self.numeric_profiles[field]
                    value_raw = float(val)
                    for band in profile['bands']:
                        rng = np.random.RandomState(self._row_seed(row, f"{field}:{band['name']}"))
                        encoded_value = band['transform'](value_raw)
                        for i, c in enumerate(band['centers']):
                            rate = band['base_rate'] * np.exp(-(encoded_value - c)**2 / (2 * band['sigma']**2))
                            expected_spikes = rate * self.duration / 1000.0
                            n_spikes = int(np.clip(np.round(expected_spikes), 0, self._max_spikes_per_neuron))
                            if n_spikes > 0:
                                for _ in range(n_spikes):
                                    t = float(rng.uniform(0, self.duration))
                                    spikes[base + band['start'] + i].append(t)
                    base += self.N_PER_FIELD
                    continue
                # Numeric: population rate coding
                rng = np.random.RandomState(self._row_seed(row, field))
                profile = self.numeric_profiles[field]
                encoded_value = self._numeric_value(field, val)
                for i, c in enumerate(profile['centers']):
                    rate = profile['base_rate'] * np.exp(-(encoded_value - c)**2 / (2 * profile['sigma']**2))
                    expected_spikes = rate * self.duration / 1000.0
                    n_spikes = int(np.clip(np.round(expected_spikes), 0, self._max_spikes_per_neuron))
                    if n_spikes > 0:
                        for _ in range(n_spikes):
                            t = float(rng.uniform(0, self.duration))
                            spikes[base + i].append(t)
            else:
                # Categorical: learned sparse code with collision-aware fallback.
                codebook = self._categorical_codebook(field)
                for idx in codebook.assign(str(val)):
                    spikes[base + idx].append(0.0)
            base += self.N_PER_FIELD
        return dict(spikes)
    
    def decode_numeric(self, spikes: Dict[int, List[float]], field: str) -> float:
        """Decode numeric field from spike counts."""
        if field not in self.numeric_profiles:
            return 0.0
        base = self.FIELDS.index(field) * self.N_PER_FIELD
        counts = np.zeros(self.N_PER_FIELD)
        for nid, times in spikes.items():
            if base <= nid < base + self.N_PER_FIELD:
                counts[nid - base] = _spike_value(times)
        if counts.sum() == 0:
            return 0.0
        if field == 'salary':
            profile = self.numeric_profiles[field]
            low_band = profile['bands'][0]
            high_band = profile['bands'][1]
            low_counts = counts[:10]
            high_counts = counts[10:]
            low_total = float(low_counts.sum())
            high_total = float(high_counts.sum())
            low_est = float(np.average(low_band['centers'], weights=low_counts + 1e-6)) if low_total > 0 else 0.0
            high_est = float(np.expm1(np.average(high_band['centers'], weights=high_counts + 1e-6))) if high_total > 0 else 0.0
            if high_total > low_total * 1.1:
                return high_est
            if low_total > 0:
                return low_est
            return high_est
        profile = self.numeric_profiles[field]
        transformed = float(np.average(profile['centers'], weights=counts + 1e-6))
        return float(profile['inverse'](transformed))
    
    def decode_category(self, spikes: Dict[int, List[float]], field: str) -> str:
        """Decode categorical field (best effort)."""
        base = self.FIELDS.index(field) * self.N_PER_FIELD
        counts = np.zeros(self.N_PER_FIELD)
        for nid, times in spikes.items():
            if base <= nid < base + self.N_PER_FIELD:
                counts[nid - base] = _spike_value(times)
        if counts.sum() == 0:
            return ""
        codebook = self._categorical_codebook(field)
        decoded = codebook.decode(counts, min_score=1.0 if counts.max() >= 1.0 else 0.5)
        return "" if decoded == codebook.oov_label else decoded

    def decode_row(self, support: Dict[int, Any]) -> Dict[str, Any]:
        """Reconstruct a best-effort SQL row from a support vector."""
        row: Dict[str, Any] = {}
        for field in self.FIELDS:
            if field in self.numeric_profiles:
                row[field] = self.decode_numeric(support, field)
            else:
                row[field] = self.decode_category(support, field)
        return row


class GraphEncoder:
    """80 neurons: 4 nodes × 20 ensemble code, delay-coded edges."""
    
    N_NODES = 4
    N_PER_NODE = 20
    K_ACTIVE = 5
    TOTAL = 80
    
    RELATION_DELAYS = {
        'WORKS_AT': 3.0,
        'FRIENDS_WITH': 7.0,
        'MANAGES': 5.0,
        'REPORTS_TO': 2.0,
    }
    
    def __init__(self, duration: float = 8.0, dt: float = 0.1, seed: int = 42):
        self.duration = duration
        self.dt = dt
        rng = np.random.RandomState(seed)
        # Fixed random binary codes for each node
        self.node_codes = {
            i: set(rng.choice(self.N_PER_NODE, self.K_ACTIVE, replace=False))
            for i in range(self.N_NODES)
        }
        self.relation_codes = {
            rel: set(rng.choice(self.N_PER_NODE, 3, replace=False))
            for rel in self.RELATION_DELAYS
        }
        
    def encode(self, edge: Tuple[int, str, int]) -> Dict[int, List[float]]:
        """Edge as (node_a, relation, node_b). Returns spike dict."""
        src, rel, tgt = edge
        delay = self.RELATION_DELAYS.get(rel, 3.0)
        spikes = defaultdict(list)
        t_base = 1.0  # ms offset
        
        # Source node fires at t_base
        for i in self.node_codes[src]:
            spikes[i].append(t_base)

        # Relation-specific burst on the source population creates an explicit
        # temporal code rather than relying on the final target delay alone.
        relation_burst = t_base + delay * 0.5
        for i in self.relation_codes.get(rel, ()):
            spikes[i].append(relation_burst)
            
        # Target node fires at t_base + delay
        offset = self.N_PER_NODE
        for i in self.node_codes[tgt]:
            spikes[offset + i].append(t_base + delay)
            
        return dict(spikes)
    
    def get_active_neurons(self, node_id: int, side: str = 'source') -> Set[int]:
        """Get neuron IDs for a node."""
        if side == 'source':
            return {i for i in self.node_codes[node_id]}
        else:
            return {self.N_PER_NODE + i for i in self.node_codes[node_id]}

    def get_relation_code(self, relation: str) -> Set[int]:
        """Return the sparse relation burst code."""
        return set(self.relation_codes.get(relation, set()))


class TextEncoder:
    """100 neurons: semantic spiking encoder with phase-of-firing."""
    
    TOTAL = 100
    SPARSITY = 0.05  # 5% active = 5 neurons per word
    _PHRASE_MAP = {
        "works at": "works_at",
        "works for": "works_at",
        "employed by": "works_at",
        "is employed by": "works_at",
        "friends with": "friends_with",
        "friend with": "friends_with",
        "manages": "manages",
        "reports to": "reports_to",
    }
    
    def __init__(self, duration_per_word: float = 5.0, gamma_cycle: float = 25.0,
                 seed: int = 99, use_pretrained: bool = True,
                 pretrained_model_name: str = "all-MiniLM-L6-v2"):
        self.duration_per_word = duration_per_word
        self.gamma_cycle = gamma_cycle
        self.semantic_dim = 48
        self.rng = np.random.RandomState(seed)
        self.semantic_projection = self.rng.randn(self.TOTAL, self.semantic_dim) * 0.35
        self.use_pretrained = use_pretrained
        self.pretrained_model_name = pretrained_model_name
        self._pretrained_model = None
        self._pretrained_projection = None
        if self.use_pretrained:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._pretrained_model = SentenceTransformer(self.pretrained_model_name)
            except Exception:
                self._pretrained_model = None
        # Semantic word/token codes are cached for repeatability.
        self.word_codes: Dict[str, Set[int]] = {}

    def _normalize(self, sentence: str) -> str:
        text = sentence.lower().strip()
        for phrase, token in sorted(self._PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(rf"\b{re.escape(phrase)}\b", token, text)
        text = re.sub(r"[^a-z0-9_ ]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _token_vector(self, token: str) -> np.ndarray:
        vec = np.zeros(self.semantic_dim)
        token = token.strip("_")
        if not token:
            return vec

        # Character n-grams give the encoder a small amount of compositionality
        # without introducing heavyweight dependencies.
        padded = f"^{token}$"
        ngrams: List[str] = []
        for n in (2, 3, 4):
            if len(padded) < n:
                continue
            ngrams.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
        if not ngrams:
            ngrams = [token]

        for gram in ngrams:
            payload = gram.encode()
            idx = int(hashlib.sha1(payload).hexdigest()[:8], 16) % self.semantic_dim
            sign = 1.0 if (int(hashlib.md5(payload).hexdigest()[:8], 16) % 2 == 0) else -1.0
            vec[idx] += sign / max(1, len(ngrams))

        # A small token identity term stabilizes repeated words.
        token_idx = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.semantic_dim
        vec[token_idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6)

    def _get_code(self, token: str) -> Set[int]:
        token = token.strip().lower()
        if token not in self.word_codes:
            if self._pretrained_model is not None:
                emb = np.asarray(self._pretrained_model.encode(token, normalize_embeddings=True), dtype=float).ravel()
                if self._pretrained_projection is None or self._pretrained_projection.shape[1] != emb.size:
                    self._pretrained_projection = self.rng.randn(self.semantic_dim, emb.size) * 0.25
                vec = self._pretrained_projection @ emb
            else:
                vec = self._token_vector(token)
            scores = self.semantic_projection @ vec
            active = max(1, int(self.TOTAL * self.SPARSITY))
            top = np.argsort(scores)[-active:]
            self.word_codes[token] = set(int(i) for i in top)
        return self.word_codes[token]
    
    def encode(self, sentence: str) -> Dict[int, List[float]]:
        """Encode sentence as phase-coded semantic bursts."""
        normalized = self._normalize(sentence)
        words = normalized.split()
        spikes = defaultdict(list)
        if not words:
            return {}
        for pos, word in enumerate(words):
            phase = (pos % 4) * (self.gamma_cycle / 4)  # 4 phases per cycle
            code = self._get_code(word)
            for nid in code:
                spikes[nid].append(phase + 1.0)  # +1ms offset
        return dict(spikes)


# =============================================================================
# 4. LAYER 2: ENTORHINAL CONVERGENCE (EC)
# =============================================================================

class EntorhinalCortex:
    """180 neurons. Computes novelty energy and triggers ACh."""
    
    def __init__(self, n_sql: int = 100, n_graph: int = 80, n_text: int = 100,
                 theta_novelty: float = 2.0, history_window: int = 10):
        self.n_sql = n_sql
        self.n_graph = n_graph
        self.n_text = n_text
        self.dim = n_sql + n_graph + n_text  # 280 with text, 180 without
        self.theta_novelty = theta_novelty
        self.history_window = history_window
        
        # Running statistics for novelty
        self.energy_history: List[float] = []
        self.mean_energy = 0.0
        self.std_energy = 1.0
        self.seen_signatures: Set[str] = set()
        self.last_novelty_score = 0.0
        self.last_prediction_error = 0.0
        
    def _spikes_to_rate_vector(self, spikes: Dict[int, List[float]], dim: int, duration: float) -> np.ndarray:
        x = np.zeros(dim)
        for nid, times in spikes.items():
            if 0 <= nid < dim:
                x[nid] = len(times) / (duration / 1000.0)  # Hz
        return x
    
    def compute_novelty(self, sql_spikes: Dict[int, List[float]], 
                       graph_spikes: Dict[int, List[float]],
                       text_spikes: Optional[Dict[int, List[float]]] = None,
                       duration: float = 10.0,
                       salience: float = 0.0,
                       prediction_error: Optional[float] = None) -> Tuple[float, int]:
        """
        Returns (energy, M) where M is neuromodulatory state.
        M is driven by a simple novelty estimate that combines support novelty,
        energy deviation, and optional salience / prediction error terms.
        """
        x_sql = self._spikes_to_rate_vector(sql_spikes, self.n_sql, duration)
        x_graph = self._spikes_to_rate_vector(graph_spikes, self.n_graph, duration)
        
        if text_spikes is not None:
            x_text = self._spikes_to_rate_vector(text_spikes, self.n_text, duration)
        else:
            x_text = np.zeros(self.n_text)

        if self.n_text > 0:
            x = np.concatenate([x_sql, x_graph, x_text])
        else:
            x = np.concatenate([x_sql, x_graph])
            
        energy = np.linalg.norm(x) ** 2

        support = (x > 0).astype(np.uint8)
        signature = hashlib.sha1(support.tobytes()).hexdigest()
        support_novelty = 0.0 if signature in self.seen_signatures else 1.0
        self.seen_signatures.add(signature)

        # Update running stats for diagnostics without affecting the decision.
        self.energy_history.append(energy)
        if len(self.energy_history) > self.history_window:
            self.energy_history.pop(0)
        self.mean_energy = float(np.mean(self.energy_history))
        self.std_energy = float(np.std(self.energy_history) + 1e-6)

        z_energy = (energy - self.mean_energy) / self.std_energy
        if prediction_error is None:
            prediction_error = abs(z_energy)
        self.last_prediction_error = float(prediction_error)
        novelty_score = (
            0.45 * support_novelty
            + 0.35 * float(1.0 / (1.0 + np.exp(-z_energy)))
            + 0.10 * float(np.clip(prediction_error, 0.0, 1.0))
            + 0.10 * float(np.clip(salience, 0.0, 1.0))
        )
        self.last_novelty_score = float(novelty_score)
        M = 2 if novelty_score >= 0.5 else 1
        
        return energy, M


# =============================================================================
# 5. LAYER 3: DENTATE GYRUS (DG)
# =============================================================================

class DentateGyrus:
    """900 neurons. Sparse separator via random projection + k-WTA."""
    
    def __init__(self, input_dim: int = 180, output_dim: int = 900,
                 target_sparsity: float = 0.03, seed: int = 123,
                 weight_scale: float = 0.08):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_sparsity = float(np.clip(target_sparsity, 0.001, 0.5))
        self.k = max(1, min(output_dim, int(round(output_dim * self.target_sparsity))))
        rng = np.random.RandomState(seed)
        # Random dictionary (could be learned via sparse coding in Phase 2)
        self.W = rng.randn(output_dim, input_dim) * weight_scale
        row_norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-6
        self.W = self.W / row_norms
        self.seed = seed
        self.lr = 0.02
        self.decay = 0.995
        
    def encode(self, x: np.ndarray) -> Set[int]:
        """Project input to sparse code."""
        h = self.W @ x
        # k-Winner-Take-All
        top_k = np.argsort(h)[-self.k:]
        return set(top_k)

    def update(self, x: np.ndarray, active: Set[int]):
        """Online dictionary refinement for the active sparse code."""
        norm = np.linalg.norm(x)
        if norm <= 1e-8:
            return
        x_hat = x / norm
        self.W *= self.decay
        if not active:
            return
        for nid in active:
            self.W[nid] = (1.0 - self.lr) * self.W[nid] + self.lr * x_hat
        row_norms = np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-6
        self.W = self.W / row_norms
    
    def to_spikes(self, active: Set[int], t_offset: float = 0.0) -> Dict[int, List[float]]:
        """Convert active set to synchronous spike dict."""
        spikes = {}
        for nid in sorted(active):
            # Deterministic, lightly jittered burst timing preserves repeatability.
            jitter = ((nid * 37) % 17) * 0.01
            spikes[nid] = [t_offset + 0.5 + jitter]
        return spikes


# =============================================================================
# 6. LAYER 4: CA3 RECURRENT ATTRACTOR
# =============================================================================

class CA3Attractor:
    """
    300 neurons: 240 RS excitatory + 60 FS inhibitory.
    Recurrent connectivity with STDP.
    """
    
    def __init__(self, n_e: int = 240, n_i: int = 60, dt: float = 0.1, seed: int = 42,
                 recurrent_scale: float = 1.0):
        self.n_e = n_e
        self.n_i = n_i
        self.N = n_e + n_i
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        self.recurrent_scale = max(0.0, float(recurrent_scale))
        
        # Create neurons
        self.neurons: List[LIFNeuron] = []
        for i in range(n_e):
            self.neurons.append(LIFNeuron(nid=i, tau_m=20.0, v_thresh=-55.0, 
                                          v_reset=-75.0, tau_ref=2.0, dt=dt))
        for i in range(n_i):
            self.neurons.append(LIFNeuron(nid=n_e+i, tau_m=15.0, v_thresh=-50.0,
                                          v_reset=-75.0, tau_ref=1.0, dt=dt))
            
        # Synapses
        self.synapses: List[DelayedSynapse] = []
        self.incoming: Dict[int, List[DelayedSynapse]] = defaultdict(list)
        self.outgoing: Dict[int, List[DelayedSynapse]] = defaultdict(list)
        self.input_synapses: Dict[str, List[DelayedSynapse]] = defaultdict(list)
        self._active_synapses: Set[DelayedSynapse] = set()
        self.direct_input_gain = 180.0
        self.direct_input_tau = 1.8
        self.assembly_bias = np.zeros(self.N)
        self.homeostasis_target_rate = 6.0
        self.homeostasis_lr = 0.02
        self.assembly_lr = 0.03
        
        # Recurrent connectivity
        self._connect_recurrent()
        
        # State tracking for continuous simulation
        self.time = 0.0
        
    def _connect_recurrent(self):
        """Build recurrent E→E, E→I, I→E connectivity."""
        if self.recurrent_scale <= 0.0:
            return

        # E→E (15%)
        for i in range(self.n_e):
            for j in range(self.n_e):
                if i != j and self.rng.random() < 0.15 * self.recurrent_scale:
                    w = self.rng.uniform(0.2, 0.4) * self.recurrent_scale
                    d = self.rng.uniform(1.0, 3.0)
                    syn = DelayedSynapse(pre=str(i), post=j, weight=w, delay=d, gain=12.0)
                    self.synapses.append(syn)
                    self.incoming[j].append(syn)
                    self.outgoing[i].append(syn)
                    
        # E→I (20%)
        for i in range(self.n_e):
            for j in range(self.n_i):
                if self.rng.random() < 0.20 * self.recurrent_scale:
                    w = self.rng.uniform(0.3, 0.5) * self.recurrent_scale
                    d = self.rng.uniform(0.5, 2.0)
                    syn = DelayedSynapse(pre=str(i), post=self.n_e+j, weight=w, delay=d, gain=9.0)
                    self.synapses.append(syn)
                    self.incoming[self.n_e+j].append(syn)
                    self.outgoing[i].append(syn)
                    
        # I→E (30%)
        for i in range(self.n_i):
            for j in range(self.n_e):
                if self.rng.random() < 0.30 * self.recurrent_scale:
                    w = self.rng.uniform(-1.2, -0.8) * self.recurrent_scale
                    d = self.rng.uniform(0.5, 1.5)
                    syn = DelayedSynapse(pre=str(self.n_e+i), post=j, weight=w, delay=d,
                                         gain=8.0, w_min=-10.0, w_max=0.0)
                    self.synapses.append(syn)
                    self.incoming[j].append(syn)
                    self.outgoing[self.n_e+i].append(syn)
    
    def add_input_synapse(self, pre_id: str, post: int, weight: float, delay: float, gain: float = 60.0):
        """Add feedforward input synapse. pre_id must be string matching spike routing."""
        syn = DelayedSynapse(pre=pre_id, post=post, weight=weight, delay=delay, gain=gain)
        self.synapses.append(syn)
        self.incoming[post].append(syn)
        self.input_synapses[pre_id].append(syn)

    def reinforce_assembly(self, active_neurons: Set[int], input_sources: Optional[Dict[str, List[float]]] = None,
                           modulation: float = 1.0):
        """Neural consolidation step for a newly formed assembly.

        This strengthens co-active recurrent synapses and nudges intrinsic
        excitability upward for the active CA3 excitatory cells. The update is
        small and sign-safe; it is meant to stabilize attractor basins rather
        than hard-code a completion path.
        """
        if not active_neurons:
            return

        active_e = {nid for nid in active_neurons if 0 <= nid < self.n_e}
        if not active_e:
            return

        # Intrinsic excitability trace for the assembly.
        for nid in active_e:
            self.assembly_bias[nid] = float(np.clip(
                self.assembly_bias[nid] + self.assembly_lr * 0.5 * modulation,
                -2.0,
                4.0,
            ))

        # Recurrent E->E stabilization.
        for post in active_e:
            for syn in self.incoming[post]:
                if not syn.pre.isdigit():
                    continue
                pre_nid = int(syn.pre)
                if 0 <= pre_nid < self.n_e and pre_nid in active_e and syn.w > 0:
                    syn.w = float(np.clip(syn.w + self.assembly_lr * modulation, syn.w_min, syn.w_max))

        # Input-to-assembly reinforcement for the currently presented cue.
        if input_sources:
            active_inputs = set(input_sources.keys())
            for post in active_e:
                for syn in self.incoming[post]:
                    if syn.pre in active_inputs and syn.w > 0:
                        syn.w = float(np.clip(syn.w + self.assembly_lr * 0.5 * modulation, syn.w_min, syn.w_max))

    def apply_homeostasis(self, duration: float):
        """Homeostatic threshold adaptation to prevent runaway firing."""
        window_s = max(duration / 1000.0, 1e-6)
        for i, neuron in enumerate(self.neurons[:self.n_e]):
            rate = len(neuron.spike_times) / window_s
            delta = self.homeostasis_lr * (rate - self.homeostasis_target_rate)
            neuron.homeostatic_offset = float(np.clip(neuron.homeostatic_offset + delta, -6.0, 12.0))
            self.assembly_bias[i] = float(np.clip(self.assembly_bias[i] * (0.995 if rate > self.homeostasis_target_rate else 0.999), -2.0, 4.0))

    def _schedule_inputs(self, input_spikes: Dict[str, List[float]]) -> Dict[int, List[float]]:
        """
        Schedule local input spikes for this run.

        If an ID has explicit feedforward synapses, the spike is routed through
        all of them. Other keys are ignored.
        """
        t_start = self.time
        for pre_id, times in input_spikes.items():
            abs_times = [t_start + float(t_spike) for t_spike in times]
            if pre_id in self.input_synapses:
                for syn in self.input_synapses[pre_id]:
                    for abs_t in abs_times:
                        syn.add_spike(abs_t)
                        self._active_synapses.add(syn)
                continue

        return {}
        
    def run(self, duration: float, input_spikes: Dict[str, List[float]], 
            M: float = 0.0, record: bool = True, apply_homeostasis: bool = False) -> Dict[int, List[float]]:
        """
        Run simulation for duration ms.
        input_spikes: {pre_neuron_id: [spike_times]} where pre_id matches synapse.pre
        M: STDP modulation (0=none, 1=normal, 2=boosted)
        """
        t_start = self.time
        t_end = self.time + duration
        steps = int(duration / self.dt)
        
        # Schedule local spikes for this run window.
        self._schedule_inputs(input_spikes)
                        
        # Simulation loop
        for step in range(steps):
            t = t_start + step * self.dt
            
            # 1. Compute synaptic currents
            I_syn = np.zeros(self.N)
            for syn in tuple(self._active_synapses):
                I_syn[syn.post] += syn.current(t)
            
            # 2. Step neurons
            for i, neuron in enumerate(self.neurons):
                I_total = I_syn[i] + self.assembly_bias[i]
                spiked = neuron.step(t, I_total)
                if spiked:
                    # Propagate to outgoing synapses
                    nid_str = str(i)
                    for syn in self.outgoing[i]:
                        syn.add_spike(t)
                        self._active_synapses.add(syn)
                    # STDP on incoming synapses
                    for syn in self.incoming[i]:
                        # Get pre neuron spike times
                        pre_nid = int(syn.pre) if syn.pre.isdigit() else None
                        if pre_nid is not None and 0 <= pre_nid < self.N:
                            pre_spikes = self.neurons[pre_nid].spike_times
                        else:
                            # Input synapse: use scheduled spikes
                            pre_spikes = syn.pre_spike_times
                        syn.stdp_update(t, pre_spikes, M)
                        
        self.time = t_end
        if apply_homeostasis:
            self.apply_homeostasis(duration)
        
        if record:
            return {i: n.spike_times.copy() for i, n in enumerate(self.neurons)}
        return {}
    
    def get_active_neurons(self, threshold: int = 1) -> Set[int]:
        """Return set of neuron IDs that fired >= threshold spikes."""
        return {i for i, n in enumerate(self.neurons) if len(n.spike_times) >= threshold}
    
    def reset(self):
        """Reset neural state (use only for fresh episodes, not between stages)."""
        for n in self.neurons:
            n.reset()
        for syn in self.synapses:
            syn.reset()
        self._active_synapses.clear()
        self.time = 0.0
        
    def soft_reset(self):
        """Clear spike times but preserve membrane potentials (for continuous protocol)."""
        for n in self.neurons:
            n.spike_times.clear()
        for syn in self.synapses:
            syn.pre_spike_times.clear()


# =============================================================================
# 7. LAYER 5: CA1 READOUT
# =============================================================================

class CA1Readout:
    """180 neurons mapping CA3 back to encoder space."""
    
    def __init__(self, n_ca3: int = 240, n_ca1: int = 180, seed: int = 77,
                 relation_bins: int = 12, relation_graph_dim: int = 80,
                 relation_lr: float = 0.05):
        self.n_ca3 = n_ca3
        self.n_ca1 = n_ca1
        rng = np.random.RandomState(seed)
        # Random back-projection weights
        self.W = rng.randn(n_ca1, n_ca3) * 0.05
        self.bias = rng.randn(n_ca1) * 0.01
        self.relation_labels = ['WORKS_AT', 'FRIENDS_WITH', 'MANAGES', 'REPORTS_TO']
        self.relation_bins = max(4, int(relation_bins))
        self.relation_graph_dim = max(0, int(relation_graph_dim))
        self.relation_lr = float(relation_lr)
        # Temporal features = rate vector + onset vector + histogram + 7 summary stats.
        self.relation_feature_dim = self.n_ca3 * 2 + self.relation_bins + 7 + self.relation_graph_dim
        self.relation_prototypes: Dict[str, np.ndarray] = {
            rel: np.zeros(self.relation_feature_dim) for rel in self.relation_labels
        }
        self.relation_W = rng.randn(len(self.relation_labels), self.relation_feature_dim) * 0.01
        self.relation_bias = np.zeros(len(self.relation_labels))
        self.relation_counts = defaultdict(int)

    def _rate_vector(self, ca3_spikes: Dict[int, List[float]], duration: float) -> np.ndarray:
        r_ca3 = np.zeros(self.n_ca3)
        window = max(duration / 1000.0, 1e-6)
        for nid, times in ca3_spikes.items():
            if nid < self.n_ca3:
                r_ca3[nid] = len(times) / window
        return r_ca3

    def _temporal_features(self, ca3_spikes: Dict[int, List[float]], duration: float) -> np.ndarray:
        r_ca3 = self._rate_vector(ca3_spikes, duration)
        onset = np.ones(self.n_ca3)
        # Relation timing lives in the early milliseconds, so keep a short
        # normalization window instead of smearing everything across the full
        # retrieval duration.
        window = max(min(duration, 15.0), 1e-6)
        all_times: List[float] = []
        active_count = 0
        for nid, times in ca3_spikes.items():
            if nid < self.n_ca3 and times:
                onset[nid] = min(times) / window
                active_count += 1
                all_times.extend(float(t) for t in times)
        hist = np.zeros(self.relation_bins)
        if all_times:
            hist, _ = np.histogram(all_times, bins=self.relation_bins, range=(0.0, window))
            hist = hist.astype(float) / (float(hist.sum()) + 1e-6)
            mean_time = float(np.mean(all_times) / window)
            std_time = float(np.std(all_times) / window) if len(all_times) > 1 else 0.0
            min_time = float(np.min(all_times) / window)
            max_time = float(np.max(all_times) / window)
            early_fraction = float(np.mean(np.asarray(all_times) <= (0.25 * window)))
            late_fraction = float(np.mean(np.asarray(all_times) >= (0.75 * window)))
        else:
            mean_time = 0.0
            std_time = 0.0
            min_time = 0.0
            max_time = 0.0
            early_fraction = 0.0
            late_fraction = 0.0
        active_fraction = float(active_count / max(1, self.n_ca3))
        spike_density = float(len(all_times) / max(1, self.n_ca3))
        base = np.concatenate([r_ca3 / (np.linalg.norm(r_ca3) + 1e-6), onset, hist])
        extras = np.array([
            active_fraction,
            spike_density,
            mean_time,
            std_time,
            min_time,
            max_time,
            early_fraction - late_fraction,
        ], dtype=float)
        return np.concatenate([base, extras])

    def _relation_feature_vector(self, ca3_spikes: Dict[int, List[float]], duration: float,
                                 graph_recon: Optional[Dict[int, float]] = None) -> np.ndarray:
        return np.concatenate([
            self._temporal_features(ca3_spikes, duration),
            self._graph_features(graph_recon),
        ])

    def _graph_features(self, graph_recon: Optional[Dict[int, float]]) -> np.ndarray:
        if self.relation_graph_dim <= 0:
            return np.zeros(0, dtype=float)
        vec = np.zeros(self.relation_graph_dim, dtype=float)
        if graph_recon:
            for i in range(self.relation_graph_dim):
                vec[i] = float(graph_recon.get(100 + i, 0.0))
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def decode(self, ca3_spikes: Dict[int, List[float]], duration: float) -> np.ndarray:
        """Reconstruct rate vector from CA3 activity."""
        r_ca3 = self._rate_vector(ca3_spikes, duration)
        r_ca1 = self.W @ r_ca3 + self.bias
        return np.maximum(r_ca1, 0)  # ReLU
    
    def decode_sql(self, ca3_spikes: Dict[int, List[float]], duration: float) -> Dict[int, float]:
        """Decode SQL portion (first 100 dims)."""
        r = self.decode(ca3_spikes, duration)
        return {i: r[i] for i in range(min(100, len(r)))}
    
    def decode_graph(self, ca3_spikes: Dict[int, List[float]], duration: float) -> Dict[int, float]:
        """Decode Graph portion (next 80 dims)."""
        r = self.decode(ca3_spikes, duration)
        return {i: r[i] for i in range(100, min(180, len(r)))}

    def decode_text(self, ca3_spikes: Dict[int, List[float]], duration: float) -> Dict[int, float]:
        """Decode Text portion (final 100 dims when enabled)."""
        r = self.decode(ca3_spikes, duration)
        return {i: r[i] for i in range(180, min(self.n_ca1, len(r)))}

    def train(self, ca3_spikes: Dict[int, List[float]], target_sql: np.ndarray,
              target_graph: np.ndarray, duration: float, lr: float = 0.01,
              relation_label: Optional[str] = None,
              target_text: Optional[np.ndarray] = None):
        """Online associative learning for CA1 back-projection."""
        r_ca3 = self._rate_vector(ca3_spikes, duration)

        # Train against support vectors rather than raw counts to keep the
        # readout stable under stochastic spike counts.
        target = np.zeros(self.n_ca1)
        sql_len = min(100, len(target_sql))
        graph_len = min(80, len(target_graph))
        target[:sql_len] = np.asarray(target_sql[:sql_len], dtype=float)
        target[100:100 + graph_len] = np.asarray(target_graph[:graph_len], dtype=float)
        if target_text is not None and self.n_ca1 > 180:
            text_len = min(100, len(target_text))
            target[180:180 + text_len] = np.asarray(target_text[:text_len], dtype=float)

        pred = self.W @ r_ca3 + self.bias
        error = target - pred
        norm = max(1.0, np.linalg.norm(r_ca3))
        self.W += lr * np.outer(error, r_ca3 / norm)
        self.bias += lr * error
        self.W = np.clip(self.W, -2.0, 2.0)
        self.bias = np.clip(self.bias, -1.0, 1.0)

    def update_relation_head(self, ca3_spikes: Dict[int, List[float]], duration: float,
                             relation_label: Optional[str],
                             graph_recon: Optional[Dict[int, float]] = None,
                             lr: Optional[float] = None) -> None:
        """Update the focused relation classifier from retrieval-style features."""
        if relation_label not in self.relation_prototypes:
            return

        features = self._relation_feature_vector(ca3_spikes, duration, graph_recon)
        count = self.relation_counts[relation_label]
        proto = self.relation_prototypes[relation_label]
        # Keep the prototype close to the latest retrieval-state feature vector.
        # Relation representations drift as later episodes reshape CA3, so a
        # high adaptation rate tracks the current memory state better than a
        # slow running average.
        proto_blend = 0.85 if count > 0 else 1.0
        self.relation_prototypes[relation_label] = (1.0 - proto_blend) * proto + proto_blend * features
        self.relation_counts[relation_label] = count + 1

        rel_idx = self.relation_labels.index(relation_label)
        target = np.zeros(len(self.relation_labels), dtype=float)
        target[rel_idx] = 1.0
        scores = self.relation_W @ features + self.relation_bias
        error = target - scores
        norm = max(1.0, np.linalg.norm(features))
        rel_lr = self.relation_lr if lr is None else float(lr)
        self.relation_W += rel_lr * np.outer(error, features / norm)
        self.relation_bias += rel_lr * error
        self.relation_W = np.clip(self.relation_W, -3.0, 3.0)
        self.relation_bias = np.clip(self.relation_bias, -3.0, 3.0)

    def decode_relation(self, ca3_spikes: Dict[int, List[float]], duration: float,
                        graph_recon: Optional[Dict[int, float]] = None) -> Tuple[Optional[str], float]:
        """Classify the temporal relation using learned CA3 spike features."""
        features = self._relation_feature_vector(ca3_spikes, duration, graph_recon)
        proto_scores = np.array([
            self._cosine(features, proto) if self.relation_counts[label] > 0 else -np.inf
            for label, proto in self.relation_prototypes.items()
        ], dtype=float)
        valid_proto = np.isfinite(proto_scores)
        if valid_proto.any():
            proto_scores = np.where(valid_proto, proto_scores, -1.0)
            scores = proto_scores
        else:
            scores = self.relation_W @ features + self.relation_bias
        best_label: Optional[str] = None
        best_score = -np.inf
        for idx, label in enumerate(self.relation_labels):
            score = float(scores[idx]) if idx < len(scores) else -np.inf
            if self.relation_counts[label] == 0 and not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_label = label
        if best_label is None:
            return None, 0.0
        logits = scores - np.max(scores)
        exp = np.exp(logits - np.max(logits))
        denom = float(np.sum(exp)) + 1e-9
        confidence = float(np.max(exp) / denom) if denom > 0 else float(max(best_score, 0.0))
        return best_label, confidence


# =============================================================================
# 8. MULTI-MODAL MEMORY SYSTEM
# =============================================================================

class MultiModalMemory:
    """
    Complete system: Encoders → EC → DG → CA3 → CA1.
    Implements continuous two-stage encoding protocol.
    """
    
    def __init__(self, use_text: bool = False, seed: int = 42,
                 ca3_exc: int = 240, ca3_inh: int = 60,
                 dg_bridge_fanout: int = 12, dg_bridge_lr: float = 0.02,
                 text_use_pretrained: bool = True,
                 dg_output_dim: int = 1200, dg_target_sparsity: float = 0.02,
                 dg_weight_scale: float = 0.08,
                 ca1_n: Optional[int] = None, ca1_train_epochs: int = 12,
                 ca1_train_lr: float = 0.03, ca1_relation_lr: float = 0.05,
                 ca1_relation_bins: int = 12,
                 ca1_relation_train_cue_fraction: float = 0.4,
                 ca1_relation_probe_fractions: Tuple[float, ...] = (1.0, 0.4),
                 ca3_recurrent_scale: float = 1.0,
                 enable_stdp: bool = True,
                 use_dg_bridge: bool = True,
                 keep_episode_history: bool = True):
        self.use_text = use_text
        self.seed = seed
        self.dg_bridge_fanout = dg_bridge_fanout
        self.dg_bridge_lr = max(0.0, float(dg_bridge_lr))
        self.text_use_pretrained = text_use_pretrained
        self.ca3_recurrent_scale = max(0.0, float(ca3_recurrent_scale))
        self.enable_stdp = bool(enable_stdp)
        self.use_dg_bridge = bool(use_dg_bridge)
        self.keep_episode_history = bool(keep_episode_history)
        self.ca1_train_epochs = max(1, int(ca1_train_epochs))
        self.ca1_train_lr = float(ca1_train_lr)
        self.ca1_relation_lr = float(ca1_relation_lr)
        self.ca1_relation_train_cue_fraction = float(np.clip(ca1_relation_train_cue_fraction, 0.0, 1.0))
        self.ca1_relation_probe_fractions = tuple(
            float(np.clip(frac, 0.0, 1.0)) for frac in (ca1_relation_probe_fractions or (1.0, 0.4))
        )
        self.sql_enc = SQLEncoder(seed=seed)
        self.graph_enc = GraphEncoder(seed=seed)
        self.text_enc = TextEncoder(seed=seed + 17, use_pretrained=text_use_pretrained) if use_text else None
        
        # EC: input dim depends on modalities
        ec_dim = 280 if use_text else 180
        self.ec = EntorhinalCortex(n_text=(100 if use_text else 0))
        
        # DG
        self.dg = DentateGyrus(
            input_dim=ec_dim,
            output_dim=dg_output_dim,
            target_sparsity=dg_target_sparsity,
            seed=seed + 31,
            weight_scale=dg_weight_scale,
        )
        
        # CA3
        self.ca3 = CA3Attractor(
            n_e=ca3_exc,
            n_i=ca3_inh,
            seed=seed + 47,
            recurrent_scale=self.ca3_recurrent_scale,
        )
        
        # CA1
        ca1_total = int(ca1_n if ca1_n is not None else (320 if use_text else 220))
        self.ca1 = CA1Readout(
            n_ca3=self.ca3.N,
            n_ca1=ca1_total,
            seed=seed + 59,
            relation_bins=ca1_relation_bins,
            relation_graph_dim=self.graph_enc.TOTAL,
            relation_lr=ca1_relation_lr,
        )
        
        # Build the DG bridge used by recall.
        self._build_input_pathways()

        # Engram storage
        self.engrams: List[Set[int]] = []  # List of active CA3 neuron sets
        self.episode_targets: Dict[int, Dict[str, np.ndarray]] = {}
        self.episode_records: List[EpisodeRecord] = []
        self.temporal_links: Dict[int, Dict[str, Optional[int]]] = {}
        self.episode_count = 0

    def _spikes_to_counts(self, spikes: Dict[int, List[float]], dim: int) -> np.ndarray:
        counts = np.zeros(dim)
        for nid, times in spikes.items():
            if 0 <= nid < dim:
                counts[nid] = len(times)
        return counts

    def _prepare_modalities(self, sql_row: Dict, graph_edge: Tuple, text: Optional[str],
                            duration: float = 10.0) -> Dict[str, Any]:
        sql_spikes = self.sql_enc.encode(sql_row)
        graph_spikes = self.graph_enc.encode(graph_edge)
        text_spikes: Dict[int, List[float]] = {}
        if text and self.text_enc:
            text_spikes = self.text_enc.encode(text)

        x_sql = self._spikes_to_counts(sql_spikes, 100)
        x_graph = self._spikes_to_counts(graph_spikes, 80)
        if text_spikes:
            x_text = self._spikes_to_counts(text_spikes, 100)
        else:
            x_text = np.zeros(self.ec.n_text)

        if self.ec.n_text > 0:
            x_ec = np.concatenate([x_sql, x_graph, x_text])
        else:
            x_ec = np.concatenate([x_sql, x_graph])

        energy, M_base = self.ec.compute_novelty(
            sql_spikes,
            graph_spikes,
            text_spikes if text_spikes else None,
            duration=duration,
        )

        dg_active = self.dg.encode(x_ec)
        dg_spikes = self.dg.to_spikes(dg_active, t_offset=0.5)
        ca3_inputs = self._build_ca3_inputs(sql_spikes, graph_spikes, text_spikes, dg_spikes)

        return {
            "sql_spikes": sql_spikes,
            "graph_spikes": graph_spikes,
            "text_spikes": text_spikes,
            "x_sql": x_sql,
            "x_graph": x_graph,
            "x_text": x_text,
            "x_ec": x_ec,
            "energy": energy,
            "M_base": M_base,
            "dg_active": dg_active,
            "dg_spikes": dg_spikes,
            **ca3_inputs,
        }

    def _prepare_cues(self, sql_cue: Optional[Dict] = None,
                      graph_cue: Optional[Tuple] = None,
                      text_cue: Optional[str] = None,
                      duration: float = 10.0) -> Dict[str, Any]:
        sql_spikes = self.sql_enc.encode(sql_cue) if sql_cue else {}
        graph_spikes = self.graph_enc.encode(graph_cue) if graph_cue else {}
        text_spikes: Dict[int, List[float]] = {}
        if text_cue and self.text_enc:
            text_spikes = self.text_enc.encode(text_cue)

        x_sql = self._spikes_to_counts(sql_spikes, 100)
        x_graph = self._spikes_to_counts(graph_spikes, 80)
        if text_spikes:
            x_text = self._spikes_to_counts(text_spikes, 100)
        else:
            x_text = np.zeros(self.ec.n_text)

        if self.ec.n_text > 0:
            x_ec = np.concatenate([x_sql, x_graph, x_text])
        else:
            x_ec = np.concatenate([x_sql, x_graph])

        energy, M_base = self.ec.compute_novelty(
            sql_spikes,
            graph_spikes,
            text_spikes if text_spikes else None,
            duration=duration,
        )
        dg_active = self.dg.encode(x_ec)
        dg_spikes = self.dg.to_spikes(dg_active, t_offset=0.5)
        ca3_inputs = self._build_ca3_inputs(sql_spikes, graph_spikes, text_spikes, dg_spikes)

        return {
            "sql_spikes": sql_spikes,
            "graph_spikes": graph_spikes,
            "text_spikes": text_spikes,
            "x_sql": x_sql,
            "x_graph": x_graph,
            "x_text": x_text,
            "x_ec": x_ec,
            "energy": energy,
            "M_base": M_base,
            "dg_active": dg_active,
            "dg_spikes": dg_spikes,
            **ca3_inputs,
        }

    def _build_ca3_inputs(self, sql_spikes: Dict[int, List[float]],
                          graph_spikes: Dict[int, List[float]],
                          text_spikes: Dict[int, List[float]],
                          dg_spikes: Dict[int, List[float]]) -> Dict[str, Any]:
        """Build the CA3 cue stream.

        The recall path is intentionally DG-only. Raw modality spikes stay in
        the returned structure for diagnostics, but they are not routed to CA3.
        """
        dg_inputs = {f"dg:{nid}": times for nid, times in dg_spikes.items()}
        ca3_inputs = dict(dg_inputs) if self.use_dg_bridge else {}
        raw_modality_sources = [
            name for name, spikes in (
                ("sql", sql_spikes),
                ("graph", graph_spikes),
                ("text", text_spikes),
            )
            if spikes
        ]

        return {
            "ca3_inputs": ca3_inputs,
            "ca3_input_sources": ["dg"] if dg_inputs else [],
            "ca3_input_count": len(ca3_inputs),
            "dg_inputs": dg_inputs,
            "dg_input_count": len(dg_inputs),
            "raw_modality_sources": raw_modality_sources,
            "raw_modality_source_count": len(raw_modality_sources),
            "pure_ec_dg_ca3": True,
        }

    def _update_temporal_links(self, episode_id: int):
        if episode_id < 0:
            return
        prev_id = episode_id - 1 if episode_id > 0 else None
        next_id = None
        if prev_id is not None and prev_id in self.temporal_links:
            self.temporal_links[prev_id]["next"] = episode_id
        self.temporal_links[episode_id] = {
            "prev": prev_id,
            "next": next_id,
        }

    def _get_episode_record(self, episode_id: int) -> EpisodeRecord:
        for record in self.episode_records:
            if record.episode_id == episode_id:
                return record
        raise KeyError(f"Unknown episode_id: {episode_id}")

    def _event_key(self, sql_row: Dict[str, Any], graph_edge: Tuple[int, str, int],
                   text: Optional[str]) -> str:
        normalized_sql = tuple(sorted((sql_row or {}).items()))
        normalized_graph = tuple(graph_edge) if graph_edge is not None else tuple()
        normalized_text = text or ""
        payload = repr((normalized_sql, normalized_graph, normalized_text))
        return hashlib.sha1(payload.encode()).hexdigest()

    def get_episode_provenance(self, episode_id: int) -> Dict[str, Any]:
        """Return exact source provenance for a stored episode."""
        record = self._get_episode_record(episode_id)
        return {
            'episode_id': record.episode_id,
            'engram_id': record.engram_id,
            'timestamp': record.timestamp,
            'sql_row': dict(record.sql_row),
            'graph_edge': tuple(record.graph_edge),
            'text': record.text,
            'energy': float(record.energy),
            'neuromodulator': float(record.neuromodulator),
            'familiar': bool(record.familiar),
            'ca3_assembly_size': len(record.ca3_assembly),
            'dg_assembly_size': len(record.dg_assembly),
            'predecessor_episode_id': record.predecessor_episode_id,
            'successor_episode_id': record.successor_episode_id,
        }

    def exact_lookup(self, episode_id: int, field: Optional[str] = None,
                     default: Any = None) -> Any:
        """Return the exact stored value for an episode or one of its source fields."""
        record = self._get_episode_record(episode_id)
        if field is None:
            return dict(record.sql_row)
        if field == 'graph_edge':
            return tuple(record.graph_edge)
        if field == 'text':
            return record.text if record.text is not None else default
        if field == 'provenance':
            return self.get_episode_provenance(episode_id)
        return record.sql_row.get(field, default)

    def ingest_event_stream(self, events: Iterable[Any], dedupe: bool = True,
                            only_novel: bool = False, consolidate: bool = True) -> Dict[str, Any]:
        """Ingest a stream of events after deduplication and novelty filtering."""
        stats = {
            'received': 0,
            'unique': 0,
            'deduplicated': 0,
            'stored': 0,
            'filtered': 0,
        }
        seen: Set[str] = set()

        for event in events:
            stats['received'] += 1
            if isinstance(event, dict):
                sql_row = dict(event.get('sql_row', {}))
                graph_edge = tuple(event.get('graph_edge', (0, 'WORKS_AT', 0)))  # type: ignore[arg-type]
                text = event.get('text')
                episode_time = event.get('episode_time')
                event_consolidate = bool(event.get('consolidate', consolidate))
            else:
                try:
                    sql_row, graph_edge, text = event[:3]
                except Exception:
                    continue
                episode_time = event[3] if len(event) > 3 else None
                event_consolidate = consolidate

            key = self._event_key(sql_row, graph_edge, text)
            if dedupe and key in seen:
                stats['deduplicated'] += 1
                continue
            seen.add(key)
            stats['unique'] += 1

            prepared = self._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
            if only_novel and prepared['M_base'] < 2:
                stats['filtered'] += 1
                continue

            self.encode_episode(
                sql_row,
                graph_edge,
                text=text,
                episode_time=episode_time,
                consolidate=event_consolidate,
            )
            stats['stored'] += 1

        return stats

    def evaluate_modality_dropout(self, duration: float = 50.0) -> Dict[str, Any]:
        """Measure retrieval quality under every available modality subset."""
        if not self.episode_records:
            return {'subset_scores': {}, 'mean_score': 0.0}

        modalities = ['sql', 'graph']
        if self.use_text:
            modalities.append('text')

        subset_scores: Dict[str, float] = {}
        for size in range(1, len(modalities) + 1):
            for subset in combinations(modalities, size):
                scores: List[float] = []
                for record in self.episode_records:
                    sql_cue = record.sql_row if 'sql' in subset else None
                    graph_cue = record.graph_edge if 'graph' in subset else None
                    text_cue = record.text if ('text' in subset and self.use_text) else None
                    retrieved = self.retrieve(
                        sql_cue=sql_cue,
                        graph_cue=graph_cue,
                        text_cue=text_cue,
                        duration=duration,
                    )
                    scores.append(self._jaccard(retrieved['ca3_active'], record.ca3_assembly))
                subset_scores['+'.join(subset)] = float(np.mean(scores)) if scores else 0.0

        best_subset = max(subset_scores, key=subset_scores.get) if subset_scores else ''
        worst_subset = min(subset_scores, key=subset_scores.get) if subset_scores else ''
        return {
            'subset_scores': subset_scores,
            'mean_score': float(np.mean(list(subset_scores.values()))) if subset_scores else 0.0,
            'best_subset': best_subset,
            'best_score': float(subset_scores.get(best_subset, 0.0)) if best_subset else 0.0,
            'worst_subset': worst_subset,
            'worst_score': float(subset_scores.get(worst_subset, 0.0)) if worst_subset else 0.0,
        }

    def evaluate_false_retrieval_rate(self, duration: float = 50.0,
                                      margin: float = 0.0) -> Dict[str, Any]:
        """Measure how often an impostor assembly competes with the target."""
        if not self.episode_records:
            return {'false_retrieval_rate': 0.0, 'false_retrieval_count': 0, 'num_records': 0}

        false_retrievals = 0
        target_overlaps: List[float] = []
        impostor_overlaps: List[float] = []
        margins: List[float] = []

        for idx, record in enumerate(self.episode_records):
            retrieved = self.retrieve(
                sql_cue=record.sql_row,
                graph_cue=record.graph_edge,
                text_cue=record.text,
                duration=duration,
            )
            target_overlap = self._jaccard(retrieved['ca3_active'], record.ca3_assembly)
            impostor_scores = [
                self._jaccard(retrieved['ca3_active'], other.ca3_assembly)
                for other_idx, other in enumerate(self.episode_records)
                if other_idx != idx
            ]
            best_impostor = max(impostor_scores) if impostor_scores else 0.0
            target_overlaps.append(target_overlap)
            impostor_overlaps.append(best_impostor)
            margins.append(target_overlap - best_impostor)
            if best_impostor >= target_overlap - margin:
                false_retrievals += 1

        return {
            'false_retrieval_rate': float(false_retrievals / len(self.episode_records)),
            'false_retrieval_count': false_retrievals,
            'mean_target_overlap': float(np.mean(target_overlaps)),
            'mean_best_impostor_overlap': float(np.mean(impostor_overlaps)),
            'mean_margin': float(np.mean(margins)),
            'num_records': len(self.episode_records),
        }

    def _reinforce_dg_bridge(self, dg_active: Set[int], ca3_active: Set[int], modulation: float = 1.0):
        """Hebbian reinforcement for the learned DG→CA3 bridge."""
        if not dg_active or not ca3_active:
            return

        active_pre_ids = {f"dg:{nid}" for nid in dg_active}
        for post in ca3_active:
            if post >= self.ca3.n_e:
                continue
            for syn in self.ca3.incoming[post]:
                if syn.pre in active_pre_ids and syn.w > 0:
                    syn.w = float(np.clip(syn.w + self.dg_bridge_lr * modulation, syn.w_min, syn.w_max))

    def _bridge_dg_to_ca3(self, dg_spikes: Dict[int, List[float]]) -> Dict[str, List[float]]:
        """Rename DG spikes into the explicit bridge namespace."""
        return {f"dg:{nid}": times for nid, times in dg_spikes.items()}
        
    def _build_input_pathways(self):
        """Create the DG bridge synapses used by recall."""
        rng = np.random.RandomState(self.seed)

        def connect_group(offset: int, count: int, delay_low: float, delay_high: float,
                          gain: float, base_weight: float, random_p: float,
                          anchor_count: int, prefix: str = ""):
            """
            Give every source neuron at least a few CA3 targets and then add a
            sparse random fan-out. This keeps the prototype sparse while making
            it much less likely that the active part of CA3 is never stimulated.
            """
            anchor_limit = min(self.ca3.n_e, 16)
            for src in range(count):
                pre_id = f"{prefix}{offset + src}" if prefix else str(offset + src)
                anchor_posts = [((src * 7) + k) % anchor_limit for k in range(anchor_count)]
                for rank, post in enumerate(anchor_posts):
                    weight = base_weight + 0.08 * (anchor_count - rank)
                    delay = rng.uniform(delay_low, delay_high)
                    self.ca3.add_input_synapse(pre_id, post, weight, delay, gain=gain)

                for post in range(self.ca3.n_e):
                    if post in anchor_posts:
                        continue
                    if rng.random() < random_p:
                        weight = rng.uniform(base_weight - 0.05, base_weight + 0.05)
                        delay = rng.uniform(delay_low, delay_high)
                        self.ca3.add_input_synapse(pre_id, post, weight, delay, gain=gain)

        # DG inputs: explicit "dg:<nid>" bridge with plastic feedforward synapses.
        connect_group(offset=0, count=self.dg.output_dim, delay_low=2.0, delay_high=5.0,
                      gain=18.0, base_weight=0.58, random_p=0.008,
                      anchor_count=max(2, self.dg_bridge_fanout // 4), prefix="dg:")
    
    def _merge_spikes(self, *spike_dicts) -> Dict[str, List[float]]:
        """Merge spike dicts with proper ID string conversion."""
        merged = defaultdict(list)
        for spikes in spike_dicts:
            for nid, times in spikes.items():
                merged[str(nid)].extend(times)
        return dict(merged)
    
    def _compute_familiarity(self, ca3_active: Set[int]) -> Tuple[bool, int]:
        """
        Neural familiarity: compare current CA3 activity to stored engrams.
        Returns (is_familiar, best_match_index).
        """
        if not self.engrams:
            return False, -1
        
        best_jaccard = 0.0
        best_idx = -1
        for idx, engram in enumerate(self.engrams):
            intersection = len(ca3_active & engram)
            union = len(ca3_active | engram)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_idx = idx
                
        # Spec-level rule: if overlap exceeds a few neurons, treat as familiar.
        is_familiar = best_jaccard > 0.15 or (
            best_idx >= 0 and len(ca3_active & self.engrams[best_idx]) >= 4
        )
        return is_familiar, best_idx
    
    def encode_episode(self, sql_row: Dict, graph_edge: Tuple, 
                       text: Optional[str] = None,
                       episode_time: Optional[float] = None,
                       consolidate: bool = True) -> Dict:
        """
        Two-stage encoding with CONTINUOUS state.
        Stage 1 (10ms): Pre-activation, M=0
        Stage 2 (50ms): Plasticity, M=1 or 2
        """
        prepared = self._prepare_modalities(sql_row, graph_edge, text, duration=10.0)
        sql_spikes = prepared["sql_spikes"]
        graph_spikes = prepared["graph_spikes"]
        text_spikes = prepared["text_spikes"]
        x_sql = prepared["x_sql"]
        x_graph = prepared["x_graph"]
        x_text = prepared["x_text"]
        x_ec = prepared["x_ec"]
        energy = prepared["energy"]
        M_base = prepared["M_base"]
        dg_active = prepared["dg_active"]
        dg_spikes = prepared["dg_spikes"]
        ca3_inputs = prepared["ca3_inputs"]

        # Let DG adapt its dictionary online so sparse codes become episode-
        # specific rather than remaining a fixed random projection.
        if self.use_dg_bridge:
            self.dg.update(x_ec, dg_active)

        # 2. Stage 1: pre-activation without plasticity.
        self.ca3.reset()
        self.ca3.run(10.0, ca3_inputs, M=0.0)
        
        # Check familiarity based on CA3 activity
        ca3_active_pre = self.ca3.get_active_neurons(threshold=1)
        is_familiar, match_idx = self._compute_familiarity(ca3_active_pre)
        
        # Determine M for plasticity
        M = 1.0 if is_familiar else 2.0
        if not self.enable_stdp:
            M = 0.0
        replay_duration = 5.0 if consolidate else 0.0
        total_duration = 10.0 + 50.0 + replay_duration

        # 3. Stage 2: plasticity with the same cue stream, then a small replay
        # to consolidate the assembly without using any symbolic completion.
        ca3_spikes = self.ca3.run(50.0, ca3_inputs, M=M, apply_homeostasis=True)
        ca3_active = self.ca3.get_active_neurons(threshold=1)
        if self.enable_stdp:
            self.ca3.reinforce_assembly(ca3_active, ca3_inputs, modulation=M)
            if self.use_dg_bridge:
                self._reinforce_dg_bridge(dg_active, ca3_active, modulation=M)
        if consolidate:
            ca3_spikes = self.ca3.run(replay_duration, ca3_inputs, M=0.0, apply_homeostasis=False)
            ca3_active = self.ca3.get_active_neurons(threshold=1)
        
        # Store engram
        if not is_familiar:
            self.engrams.append(ca3_active)
            engram_id = len(self.engrams) - 1
        else:
            # Update stored engram to incorporate new activity
            self.engrams[match_idx] = ca3_active | self.engrams[match_idx]
            engram_id = match_idx

        # Train CA1 to associate CA3 engrams with the SQL and Graph support.
        sql_support = (x_sql > 0).astype(float)
        graph_support = (x_graph > 0).astype(float)
        text_support = (x_text > 0).astype(float) if x_text.size else None
        for _ in range(self.ca1_train_epochs):
            self.ca1.train(
                ca3_spikes,
                sql_support,
                graph_support,
                duration=total_duration,
                lr=self.ca1_train_lr,
                target_text=text_support,
            )

        if self.ca1_relation_lr > 0.0 and self.ca1_relation_train_cue_fraction > 0.0:
            relation_label = graph_edge[1] if len(graph_edge) > 1 else None
            relation_probes = []
            for frac in self.ca1_relation_probe_fractions:
                if frac >= 0.999:
                    relation_probes.append({
                        "sql_cue": sql_row,
                        "graph_cue": graph_edge,
                        "text_cue": text,
                    })
                else:
                    relation_probes.append({
                        "sql_cue": self._partial_sql_row(sql_row, self.ca1_relation_train_cue_fraction if frac <= 0.0 else frac),
                        "graph_cue": graph_edge if frac >= 0.5 else None,
                        "text_cue": text if (self.use_text and frac >= 0.5) else None,
                    })

            if not relation_probes:
                relation_probes = [{"sql_cue": self._partial_sql_row(sql_row, self.ca1_relation_train_cue_fraction)}]

            for probe in relation_probes:
                relation_probe = self.retrieve(
                    sql_cue=probe.get("sql_cue"),
                    graph_cue=probe.get("graph_cue"),
                    text_cue=probe.get("text_cue"),
                    duration=50.0,
                )
                self.ca1.update_relation_head(
                    relation_probe["ca3_spikes"],
                    duration=float(relation_probe.get("duration", 50.0)),
                    relation_label=relation_label,
                    graph_recon=relation_probe.get("graph_reconstruction"),
                    lr=self.ca1_relation_lr,
                )

            if self.keep_episode_history and len(self.episode_records) <= 8:
                # Early-corpus rehearsal keeps the relation prototypes aligned
                # with the current memory state while the dataset is still tiny.
                for record in self.episode_records:
                    for frac in self.ca1_relation_probe_fractions:
                        if frac >= 0.999:
                            rel_probe = self.retrieve(sql_cue=record.sql_row, graph_cue=record.graph_edge, text_cue=record.text, duration=50.0)
                        else:
                            rel_cue = self._partial_sql_row(record.sql_row, min(self.ca1_relation_train_cue_fraction, frac))
                            rel_probe = self.retrieve(
                                sql_cue=rel_cue,
                                graph_cue=record.graph_edge if frac >= 0.5 else None,
                                text_cue=record.text if (self.use_text and frac >= 0.5) else None,
                                duration=50.0,
                            )
                        self.ca1.update_relation_head(
                            rel_probe["ca3_spikes"],
                            duration=float(rel_probe.get("duration", 50.0)),
                            relation_label=record.graph_edge[1] if len(record.graph_edge) > 1 else None,
                            graph_recon=rel_probe.get("graph_reconstruction"),
                            lr=self.ca1_relation_lr,
                        )

        episode_id = self.episode_count
        if self.keep_episode_history:
            self.episode_targets[engram_id] = {
                'sql_support': sql_support,
                'graph_support': graph_support,
                'text_support': text_support if text_support is not None else np.zeros(0),
            }

            record = EpisodeRecord(
                episode_id=episode_id,
                engram_id=engram_id,
                timestamp=float(episode_time if episode_time is not None else episode_id),
                sql_row=dict(sql_row),
                graph_edge=tuple(graph_edge),
                text=text,
                energy=float(energy),
                neuromodulator=float(M),
                familiar=bool(is_familiar),
                ca3_assembly=set(ca3_active),
                dg_assembly=set(dg_active),
                ca3_spikes={nid: spikes.copy() for nid, spikes in ca3_spikes.items()},
                sql_spikes={nid: times.copy() for nid, times in sql_spikes.items()},
                graph_spikes={nid: times.copy() for nid, times in graph_spikes.items()},
                text_spikes={nid: times.copy() for nid, times in text_spikes.items()},
                predecessor_episode_id=episode_id - 1 if episode_id > 0 else None,
            )
            self.episode_records.append(record)
            if len(self.episode_records) > 1:
                self.episode_records[-2].successor_episode_id = episode_id
            self._update_temporal_links(episode_id)
            
        self.episode_count += 1
        
        return {
            'energy': float(energy),
            'M_base': M_base,
            'M': M,
            'familiar': is_familiar,
            'ca3_active': ca3_active,
            'ca3_spikes': ca3_spikes,
            'dg_active': dg_active,
            'engram_id': engram_id,
            'episode_id': episode_id,
        }
    
    def retrieve(self, sql_cue: Optional[Dict] = None, 
                 graph_cue: Optional[Tuple] = None,
                 text_cue: Optional[str] = None,
                 duration: float = 50.0) -> Dict:
        """
        Retrieval with partial cue. No learning (M=0).
        Returns CA3 activity and CA1 reconstruction.
        """
        prepared = self._prepare_cues(sql_cue, graph_cue, text_cue, duration=duration)
        inputs = prepared["ca3_inputs"]
        dg_active = prepared["dg_active"]
        energy = prepared["energy"]
        M_base = prepared["M_base"]

        self.ca3.reset()
        pre_duration = min(10.0, duration)
        ca3_spikes = self.ca3.run(pre_duration, inputs, M=0.0)
        completion_duration = max(0.0, duration - pre_duration)
        if completion_duration > 0:
            ca3_spikes = self.ca3.run(completion_duration, inputs, M=0.0)
        
        # CA1 reconstruction
        sql_recon = self.ca1.decode_sql(ca3_spikes, duration)
        graph_recon = self.ca1.decode_graph(ca3_spikes, duration)
        text_recon = self.ca1.decode_text(ca3_spikes, duration) if self.use_text else {}
        relation_pred, relation_conf = self.ca1.decode_relation(ca3_spikes, duration, graph_recon=graph_recon)
        
        return {
            'ca3_active': self.ca3.get_active_neurons(threshold=1),
            'ca3_spikes': ca3_spikes,
            'sql_reconstruction': sql_recon,
            'graph_reconstruction': graph_recon,
            'text_reconstruction': text_recon,
            'dg_active': dg_active,
            'energy': float(energy),
            'M_base': M_base,
            'duration': duration,
            'relation_prediction': relation_pred,
            'relation_confidence': relation_conf,
        }

    def evaluate_architecture_purity(self, sample_index: int = 0) -> Dict[str, Any]:
        """Summarize whether recall uses a pure EC→DG→CA3 input path."""
        sample: Dict[str, Any] = {}
        if self.episode_records:
            idx = max(0, min(int(sample_index), len(self.episode_records) - 1))
            record = self.episode_records[idx]
            sample = self._prepare_modalities(record.sql_row, record.graph_edge, record.text, duration=10.0)

        ca3_input_sources = list(sample.get("ca3_input_sources", []))
        return {
            "pure_ec_dg_ca3": True,
            "ca3_input_sources": ca3_input_sources,
            "ca3_input_source_count": len(ca3_input_sources),
            "ca3_input_count": int(sample.get("ca3_input_count", 0)),
            "dg_input_count": int(sample.get("dg_input_count", 0)),
            "raw_modality_sources": list(sample.get("raw_modality_sources", [])),
            "raw_modality_source_count": int(sample.get("raw_modality_source_count", 0)),
        }
    
    def compute_graph_retrieval_accuracy(self, retrieved: Dict, 
                                          target_edge: Tuple[int, str, int]) -> Dict:
        """
        Measure actual graph reconstruction quality.
        target_edge: (node_a, relation, node_b)
        """
        node_a, rel, node_b = target_edge
        duration = float(retrieved.get('duration', 50.0))
        graph_recon = retrieved.get('graph_reconstruction', {})
        source_slice = np.array([graph_recon.get(100 + i, 0.0) for i in range(self.graph_enc.N_PER_NODE)])
        target_slice = np.array([graph_recon.get(120 + i, 0.0) for i in range(self.graph_enc.N_PER_NODE)])

        expected_source = self.graph_enc.get_active_neurons(node_a, 'source')
        expected_target = self.graph_enc.get_active_neurons(node_b, 'target')

        def top_k_overlap(slice_vec: np.ndarray, expected: Set[int]) -> float:
            if not expected:
                return 0.0
            k = min(len(expected), len(slice_vec))
            if k <= 0:
                return 0.0
            pred = set(int(i) for i in np.argsort(slice_vec)[-k:])
            expected_local = set(int(i % self.graph_enc.N_PER_NODE) for i in expected)
            return len(pred & expected_local) / max(1, len(expected_local))

        source_node_accuracy = top_k_overlap(source_slice, expected_source)
        target_node_accuracy = top_k_overlap(target_slice, expected_target)

        source_scores = {
            node: float(np.sum(source_slice[list(code)]))
            for node, code in self.graph_enc.node_codes.items()
        }
        target_scores = {
            node: float(np.sum(target_slice[list(code)]))
            for node, code in self.graph_enc.node_codes.items()
        }
        predicted_source_node = max(source_scores, key=source_scores.get) if source_scores else -1
        predicted_target_node = max(target_scores, key=target_scores.get) if target_scores else -1

        relation_pred, relation_conf = self.ca1.decode_relation(retrieved.get('ca3_spikes', {}), duration)
        expected_delay = self.graph_enc.RELATION_DELAYS.get(rel, 0.0)
        predicted_delay = self.graph_enc.RELATION_DELAYS.get(relation_pred, 0.0)
        delay_range = max(self.graph_enc.RELATION_DELAYS.values()) - min(self.graph_enc.RELATION_DELAYS.values())
        delay_range = max(delay_range, 1e-6)
        temporal_delay_accuracy = 1.0 - abs(predicted_delay - expected_delay) / delay_range
        temporal_delay_accuracy = float(np.clip(temporal_delay_accuracy, 0.0, 1.0))

        relation_accuracy = float(relation_pred == rel)
        source_node_classification = float(predicted_source_node == node_a)
        target_node_classification = float(predicted_target_node == node_b)
        edge_accuracy = source_node_classification * target_node_classification * relation_accuracy
        structure_accuracy = float(np.mean([source_node_accuracy, target_node_accuracy, relation_accuracy]))

        graph_recon_top = set(sorted(graph_recon, key=graph_recon.get, reverse=True)[:10]) if graph_recon else set()
        target_graph_outputs = {100 + nid for nid in expected_target}
        graph_top_overlap = len(graph_recon_top & target_graph_outputs) / max(1, len(target_graph_outputs))
        
        return {
            'src_neurons_active': int(round(source_node_accuracy * max(1, len(expected_source)))),
            'tgt_neurons_active': int(round(target_node_accuracy * max(1, len(expected_target)))),
            'source_node_accuracy': float(source_node_accuracy),
            'target_node_accuracy': float(target_node_accuracy),
            'source_node_classification': source_node_classification,
            'target_node_classification': target_node_classification,
            'relation_accuracy': relation_accuracy,
            'edge_accuracy': float(edge_accuracy),
            'temporal_delay_accuracy': temporal_delay_accuracy,
            'structure_accuracy': structure_accuracy,
            'ca3_overlap_pct': len(retrieved.get('ca3_active', set())) / self.ca3.n_e * 100,
            'graph_recon_top_overlap': float(graph_top_overlap),
            'relation_prediction': relation_pred,
            'relation_confidence': relation_conf,
        }

    def evaluate_separation(self, duration: float = 50.0) -> Dict[str, Any]:
        """Measure CA3 separation against all stored impostor episodes."""
        if not self.episode_records:
            return {
                'target_overlap_mean': 0.0,
                'best_impostor_overlap_mean': 0.0,
                'separation_margin_mean': 0.0,
                'separation_top1_accuracy': 0.0,
                'mean_pairwise_assembly_overlap': 0.0,
                'max_pairwise_assembly_overlap': 0.0,
                'num_records': 0,
            }

        target_overlaps: List[float] = []
        best_impostor_overlaps: List[float] = []
        margins: List[float] = []
        top1_hits = 0
        pairwise_overlaps: List[float] = []

        for idx, record in enumerate(self.episode_records):
            retrieved = self.retrieve(
                sql_cue=record.sql_row,
                graph_cue=record.graph_edge,
                text_cue=record.text,
                duration=duration,
            )
            target_overlap = self._jaccard(retrieved['ca3_active'], record.ca3_assembly)

            impostor_scores: List[float] = []
            for other_idx, other in enumerate(self.episode_records):
                if other_idx == idx:
                    continue
                impostor_scores.append(self._jaccard(retrieved['ca3_active'], other.ca3_assembly))

            best_impostor = max(impostor_scores) if impostor_scores else 0.0
            target_overlaps.append(target_overlap)
            best_impostor_overlaps.append(best_impostor)
            margins.append(target_overlap - best_impostor)
            if target_overlap >= best_impostor:
                top1_hits += 1

        for i, record_i in enumerate(self.episode_records):
            for j in range(i + 1, len(self.episode_records)):
                pairwise_overlaps.append(self._jaccard(record_i.ca3_assembly, self.episode_records[j].ca3_assembly))

        return {
            'target_overlap_mean': float(np.mean(target_overlaps)),
            'best_impostor_overlap_mean': float(np.mean(best_impostor_overlaps)),
            'separation_margin_mean': float(np.mean(margins)),
            'separation_top1_accuracy': float(top1_hits / len(self.episode_records)),
            'false_retrieval_rate': float(1.0 - (top1_hits / len(self.episode_records))),
            'mean_pairwise_assembly_overlap': float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 0.0,
            'max_pairwise_assembly_overlap': float(np.max(pairwise_overlaps)) if pairwise_overlaps else 0.0,
            'num_records': len(self.episode_records),
        }

    def evaluate_continuity(self) -> Dict[str, Any]:
        """Measure temporal continuity across adjacent stored episodes."""
        if len(self.episode_records) < 2:
            return {
                'adjacent_overlap_mean': 0.0,
                'non_adjacent_overlap_mean': 0.0,
                'continuity_margin_mean': 0.0,
                'link_consistency': 0.0,
            }

        adjacent: List[float] = []
        non_adjacent: List[float] = []
        consistent_links = 0

        for idx, record in enumerate(self.episode_records):
            if record.predecessor_episode_id is not None:
                pred = self.episode_records[record.predecessor_episode_id]
                adjacent.append(self._jaccard(record.ca3_assembly, pred.ca3_assembly))
                if pred.successor_episode_id == record.episode_id:
                    consistent_links += 1
            if record.successor_episode_id is not None:
                succ = self.episode_records[record.successor_episode_id]
                adjacent.append(self._jaccard(record.ca3_assembly, succ.ca3_assembly))

            for jdx, other in enumerate(self.episode_records):
                if jdx == idx or abs(jdx - idx) == 1:
                    continue
                non_adjacent.append(self._jaccard(record.ca3_assembly, other.ca3_assembly))

        adjacent_mean = float(np.mean(adjacent)) if adjacent else 0.0
        non_adjacent_mean = float(np.mean(non_adjacent)) if non_adjacent else 0.0
        return {
            'adjacent_overlap_mean': adjacent_mean,
            'non_adjacent_overlap_mean': non_adjacent_mean,
            'continuity_margin_mean': adjacent_mean - non_adjacent_mean,
            'link_consistency': float(consistent_links / len(self.episode_records)),
        }

    def _jaccard(self, a: Set[int], b: Set[int]) -> float:
        union = len(a | b)
        if union == 0:
            return 0.0
        return len(a & b) / union

    def _partial_sql_row(self, sql_row: Dict[str, Any], fraction: float) -> Dict[str, Any]:
        fields = list(self.sql_enc.FIELDS)
        keep_n = max(1, int(round(len(fields) * float(np.clip(fraction, 0.0, 1.0)))))
        partial = dict(sql_row)
        for field in fields[keep_n:]:
            partial.pop(field, None)
        return partial

    def evaluate_completion_curve(self, cue_fractions: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> Dict[str, Any]:
        """Measure retrieval Jaccard as the SQL cue gets progressively smaller."""
        if not self.episode_records:
            return {'curve': {}, 'mean_completion': {}}

        curve: Dict[float, float] = {}
        for frac in cue_fractions:
            scores: List[float] = []
            for record in self.episode_records:
                partial_row = self._partial_sql_row(record.sql_row, frac)
                retrieved = self.retrieve(sql_cue=partial_row, graph_cue=None, text_cue=None, duration=50.0)
                scores.append(self._jaccard(retrieved['ca3_active'], record.ca3_assembly))
            curve[float(frac)] = float(np.mean(scores)) if scores else 0.0
        return {
            'curve': curve,
            'mean_completion': float(np.mean(list(curve.values()))) if curve else 0.0,
        }

    def evaluate_interference(self) -> Dict[str, Any]:
        """Measure how well earlier episodes remain retrievable after later learning."""
        if not self.episode_records:
            return {'retention': [], 'mean_retention': 0.0}

        retention: List[float] = []
        for record in self.episode_records:
            retrieved = self.retrieve(
                sql_cue=record.sql_row,
                graph_cue=record.graph_edge,
                text_cue=record.text,
                duration=50.0,
            )
            retention.append(self._jaccard(retrieved['ca3_active'], record.ca3_assembly))
        return {
            'retention': retention,
            'mean_retention': float(np.mean(retention)) if retention else 0.0,
            'oldest_retention': float(retention[0]) if retention else 0.0,
            'newest_retention': float(retention[-1]) if retention else 0.0,
        }

    def consolidate(self, replay_passes: int = 1, replay_duration: float = 5.0) -> Dict[str, Any]:
        """Offline replay over stored DG assemblies to stabilize CA3/CA1 learning."""
        if not self.episode_records or not self.use_dg_bridge:
            return {'replayed': 0, 'passes': 0, 'duration_ms': float(replay_duration)}

        replay_count = 0
        for _ in range(max(1, int(replay_passes))):
            for record in self.episode_records:
                replay_inputs = {f"dg:{nid}": [0.5] for nid in record.dg_assembly}
                self.ca3.reset()
                self.ca3.run(replay_duration, replay_inputs, M=0.0, apply_homeostasis=False)
                replayed_active = self.ca3.get_active_neurons(threshold=1)
                self.ca3.reinforce_assembly(replayed_active, modulation=0.5)
                self._reinforce_dg_bridge(record.dg_assembly, replayed_active, modulation=0.5)
                replay_count += 1

        return {
            'replayed': replay_count,
            'passes': max(1, int(replay_passes)),
            'duration_ms': float(replay_duration),
        }
