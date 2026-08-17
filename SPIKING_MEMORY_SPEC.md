# Polychronous Multi-Modal Spiking Memory
## Complete Engineering Specification for Coding Agent

**Version:** 1.1
**Date:** 2026-08-11
**Purpose:** Implement a biologically-inspired spiking neural network that binds heterogeneous data modalities (SQL structured data, graph relationships, text) into unified episodic memories via spike-timing-dependent plasticity (STDP).

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Neural Primitives](#2-neural-primitives)
3. [Synapse Model](#3-synapse-model)
4. [Layer 1: Modality Encoders](#4-layer-1-modality-encoders)
5. [Layer 2: Entorhinal Convergence](#5-layer-2-entorhinal-convergence)
6. [Layer 3: DG Sparse Separator](#6-layer-3-dg-sparse-separator)
7. [Layer 4: CA3 Recurrent Attractor](#7-layer-4-ca3-recurrent-attractor)
8. [Layer 5: CA1 Readout](#8-layer-5-ca1-readout)
9. [Two-Stage Encoding Protocol](#9-two-stage-encoding-protocol)
10. [Retrieval Protocol](#10-retrieval-protocol)
11. [Test Specifications](#11-test-specifications)
12. [Parameters Reference](#12-parameters-reference)
13. [Risk Register](#13-risk-register)
14. [Implementation Checklist](#14-implementation-checklist)
15. [Version Roadmap and Open Issues](#15-version-roadmap-and-open-issues)

---

## 1. System Overview

### 1.1 Core Claim
A spiking neural network with delay-heterogeneous synapses and STDP can bind arbitrarily heterogeneous data modalities into unified episodic memories without explicit synchronization, shared embedding spaces, or backpropagation.

Exact SQL and graph stores remain authoritative for scalar facts and canonical records. The neural system is the episodic and associative layer, and every stored engram should retain provenance back to source records and timestamps.

### 1.2 Architecture Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                        │
│  SQL Encoder:     100 neurons (5 fields × 20 population code)      │
│  Graph Encoder:     80 neurons (4 nodes × 20 ensemble code)        │
│  Text Encoder:     100 neurons (semantic encoder → spikes)        │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2: Entorhinal Convergence (EC)                               │
│  180 neurons base (280 with text enabled)                           │
│  Computes novelty score from support, energy, and prediction error  │
│  Triggers ACh boost when novelty crosses θ_novelty                  │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3: Dentate Gyrus (DG) Sparse Separator                       │
│  900 neurons (5× expansion, 3% activity)                            │
│  Learned dictionary W_dg via online sparse coding                   │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 4: CA3 Recurrent Attractor                                   │
│  300 neurons: 240 RS excitatory + 60 FS inhibitory                 │
│  Recurrent connectivity: 15% E→E, 20% E→I, 30% I→E                │
│  STDP with neuromodulatory gating (M = 0/1/2)                      │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 5: CA1 Readout                                               │
│  180 neurons base (280 with text enabled)                           │
│  Reconstructs original modality patterns from CA3 engram           │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow

**Learning:**
```
SQL row ──► SQL Encoder ──┐
                          ├──► EC ──► DG ──► CA3 ──► STDP storage
Graph edge ──► Graph Enc ──┘         (pre-activation → plasticity)

Novelty? ──► ACh boost (M=2, lower threshold, enhance STDP)
Familiar? ──► Reinforce existing engram (M=1)
```

**Retrieval:**
```
SQL cue ──► SQL Encoder ──► CA3 (partial activation)
                              │
                              ▼
                        Recurrent dynamics settle
                              │
                              ▼
                        CA1 readout ──► Graph pattern reactivated
```

---

## 2. Neural Primitives

### 2.1 Leaky Integrate-and-Fire (LIF)

**Dynamics:**

$$\tau_m \frac{dv}{dt} = -(v - v_{rest}) + I_{ext}(t)$$

**Reset condition:** If $v \geq v_{thresh}$:
- Emit spike at time $t$
- Reset $v \leftarrow v_{reset}$
- Enter refractory period $\tau_{ref}$ during which $v$ is clamped

**Parameters:**

| Parameter | RS (Excitatory) | FS (Inhibitory) | Unit |
|-----------|-----------------|-----------------|------|
| $\tau_m$ | 20.0 | 15.0 | ms |
| $v_{rest}$ | -70.0 | -70.0 | mV |
| $v_{thresh}$ | -55.0 | -50.0 | mV |
| $v_{reset}$ | -75.0 | -75.0 | mV |
| $\tau_{ref}$ | 2.0 | 1.0 | ms |

**Implementation note:** Use Euler integration with $dt = 0.1$ ms. During refractory period, skip integration and clamp $v = v_{reset}$.

### 2.2 Izhikevich Neuron (Optional Upgrade)

For richer dynamics (bursting, adaptation):

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$
$$\frac{du}{dt} = a(bv - u)$$

**Reset:** $(v, u) \leftarrow (c, u + d)$ when $v \geq 30$ mV

**Cell type parameters:**

| Type | $a$ | $b$ | $c$ | $d$ | Behavior |
|------|-----|-----|-----|-----|----------|
| Regular Spiking (RS) | 0.02 | 0.2 | -65 | 8 | Standard pyramidal |
| Fast Spiking (FS) | 0.1 | 0.2 | -65 | 2 | Inhibitory interneuron |
| Intrinsically Bursting (IB) | 0.02 | 0.2 | -55 | 4 | Burst encoding |

---

## 3. Synapse Model

### 3.1 Delayed Synapse with Exponential PSC

**Postsynaptic Current:**

$$I_{syn}(t) = w \sum_{f} \alpha(t - t_{pre}^f - d_{ij})$$

where $\alpha(s) = \frac{s}{\tau_{syn}} e^{-s/\tau_{syn}}$ for $s \geq 0$, else 0.

**Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\tau_{syn}$ | 3.0 ms | PSC decay time constant |
| $w_{init}$ | 0.5–0.6 | Initial weight (input→CA3) |
| $w_{max}$ | 10.0 | Maximum weight |
| $w_{min}$ | 0.0 | Minimum weight |

### 3.2 Spike-Timing Dependent Plasticity (STDP)

**Weight update when post-synaptic neuron fires at $t_{post}$:**

$$\Delta w_{ij} = \sum_{f} M \cdot F(t_{post} - t_{pre}^f - d_{ij})$$

where $M$ is the neuromodulatory gain and:

$$F(\Delta t) = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \Delta t \geq 0 \\ -A_- e^{\Delta t / \tau_-} & \Delta t < 0 \end{cases}$$

**STDP Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| $A_+$ | 0.5 | LTP amplitude |
| $A_-$ | 0.48 | LTD amplitude |
| $\tau_+$ | 12.0 ms | LTP time constant |
| $\tau_-$ | 12.0 ms | LTD time constant |

**Neuromodulatory gating:**
- $M = 0$: No plasticity (pre-activation phase)
- $M = 1$: Normal STDP (familiar memory reinforcement)
- $M = 2$: Boosted STDP (novel memory encoding, simulates ACh)

**Critical implementation detail:** The STDP argument is $t_{post} - (t_{pre} + d_{ij})$, not $t_{post} - t_{pre}$. The axonal delay is part of the causal structure.

---

## 4. Layer 1: Modality Encoders

### 4.1 SQL Encoder

**Input:** SQL row as dictionary `{'field': value, ...}`

**Output:** Spike times dictionary `{neuron_id: [t1, t2, ...]}`

**Encoding scheme:**
- **Numeric fields** (age, salary): Population rate coding
  - 20 neurons per field with Gaussian tuning curves
  - Center frequencies linearly spaced across value range
  - Firing rate $r_i(v) = r_{max} \exp(-(v - c_i)^2 / 2\sigma^2)$
  - Generate spikes via Poisson process over stimulation window

- **Categorical fields** (city, role, dept): Winner-take-all
  - Hash category string to consistent neuron index
  - Single spike at $t = 0$ ms

**Parameters:**
- Fields: `['age', 'salary', 'city', 'role', 'dept']`
- Population size per field: 20 neurons
- Total SQL neurons: 100
- Stimulation duration: 8–10 ms
- Value range: $(0, 100000)$ for salary normalization

**Decoding:**
- Numeric: Weighted average of tuning curve centers by spike count
- Categorical: Reverse hash lookup from winner neuron

### 4.2 Graph Encoder

**Input:** Graph edge as tuple `(node_a, relation, node_b)`

**Output:** List of `(neuron_id, spike_time)` tuples

**Encoding scheme:**
- **Nodes:** Sparse ensemble code ($k$-of-$n$)
  - Each node: 20 neurons, 5 active (25% sparsity)
  - Fixed random binary code per node

- **Edges:** Delay-coded
  - Source node fires at $t_{base}$
  - Target node fires at $t_{base} + d_{relation}$
  - Relationship type is encoded in the **delay**, not neuron identity

**Delay mapping (information-bearing):**

| Relation | Delay | Rationale |
|----------|-------|-----------|
| WORKS_AT | 3.0 ms | Short delay: direct relationship |
| FRIENDS_WITH | 7.0 ms | Medium delay: social relationship |
| MANAGES | 5.0 ms | Medium-short: hierarchical |
| REPORTS_TO | 2.0 ms | Short: reporting line |

**Parameters:**
- Nodes: 4 (expandable)
- Population per node: 20 neurons
- Active per node ($k$): 5 neurons
- Total graph neurons: 80 (40 source-side + 40 target-side)

### 4.3 Text Encoder

**Version target:** V3

**Design:** Pretrained semantic encoder followed by semantic event compression and spike conversion.
- Normalize text into tokens and phrases.
- Produce a dense semantic embedding with a pretrained encoder when available.
- Compress the semantic event into a sparse intermediate code.
- Convert the sparse code into phase-coded spike bursts.
- If a pretrained model is unavailable, fall back to deterministic semantic projection plus character n-gram features.

**Why this matters:** Random or phase-coded token lookup does not provide enough semantic structure for cross-modal binding.

---

## 5. Layer 2: Entorhinal Convergence

**Function:** Gateway layer that detects novelty and triggers neuromodulation.

**Mechanism:**
1. Receive concatenated spike patterns from all available modality encoders.
2. Convert spikes into rate/support vectors.
3. Compute a novelty score from support novelty, energy deviation, prediction error, and optional salience.
4. If novelty is high:
   - Trigger ACh boost: lower CA3 threshold by 5 mV for 50 ms.
   - Set STDP gain $M = 2$.
5. Else:
   - Normal operation.
   - Set STDP gain $M = 1$.

**Parameters:**
- $\theta_{novelty}$: 2× standard deviation of recent input energies
- ACh threshold shift: -5 mV
- ACh duration: 50 ms

**Implementation note:** The production path should keep multi-factor novelty scoring. A single membership test is only acceptable for toy smoke tests.

---

## 6. Layer 3: DG Sparse Separator

**Function:** Pattern separation — map similar inputs to decorrelated sparse codes.

**Mechanism:** Learned sparse coding with online dictionary learning.

**Optimization:**

$$\min_{W, h} \|x - Wh\|_2^2 + \lambda \|h\|_1 + \mu \sum_{i<j} (h_i^\top h_j)^2$$

subject to $h \geq 0$, $\|h\|_0 \leq k$

**Parameters:**
- Input dimension: 180 base (280 with text enabled)
- Output dimension: 900 (5× expansion)
- Target sparsity: 3% (27 active neurons)
- $\lambda$: 0.1 (L1 penalty)
- $\mu$: 0.01 (orthogonality penalty)

**Implementation note:** Random or pretrained initialization is acceptable, but the DG dictionary itself must be updated online. k-WTA is the sparse activation rule, not the learning rule.

---

## 7. Layer 4: CA3 Recurrent Attractor

### 7.1 Neuron Composition

| Type | Count | Parameters | Role |
|------|-------|------------|------|
| RS Excitatory | 240 | $\tau_m=20$, $v_{thresh}=-55$ | Memory storage, pattern completion |
| FS Inhibitory | 60 | $\tau_m=15$, $v_{thresh}=-50$ | Lateral competition, sparse activity |

### 7.2 Connectivity

**Recurrent synapses (CA3→CA3):**

| Source | Target | Probability | Weight | Delay |
|--------|--------|-------------|--------|-------|
| E | E | 15% | 0.2–0.4 | 1.0–3.0 ms |
| E | I | 20% | 0.3–0.5 | 0.5–2.0 ms |
| I | E | 30% | -1.2 to -0.8 | 0.5–1.5 ms |

**Input synapses (Encoder→CA3):**

| Source | Target | Probability | Weight | Delay |
|--------|--------|-------------|--------|-------|
| SQL | E | 8% | 0.5–0.6 | 3.0–6.0 ms |
| Graph | E | 8% | 0.5–0.6 | 0.0–2.0 ms |

**Delay rationale:** SQL inputs have longer delays so they arrive simultaneously with Graph inputs at CA3, enabling polychronous convergence.

### 7.3 Simulation Loop

```python
for t in time_steps:
    # 1. Inject forced input spikes into synapses
    for each scheduled input spike:
        synapse.add_spike(t)

    # 2. Compute synaptic currents for all neurons
    for each neuron:
        I_syn = sum(synapse.current(t) for incoming synapses)

    # 3. Step neurons
    for each neuron:
        I_total = I_syn + I_inject(t)
        spiked = neuron.step(t, dt, I_total)

        if spiked:
            # Propagate to post-synaptic targets
            for outgoing_synapse:
                outgoing_synapse.add_spike(t)

            # STDP on incoming synapses
            for incoming_synapse:
                incoming_synapse.stdp_update(
                    t_post=t,
                    pre_spikes=pre_neuron.spike_times,
                    M=modulatory_gain
                )
```

---

## 8. Layer 5: CA1 Readout

**Function:** Map CA3 engram activity back to modality encoder space.

**Architecture:**
- 180 neurons base (280 with text enabled)
- Each CA3 E neuron connects to subset of CA1 neurons
- CA1 neurons connect back to encoder reconstruction layers

**Implementation note:** CA1 back-projections are part of the baseline. CA3 overlap is a diagnostic metric, not a substitute for readout. The CA1 layer should reconstruct modality support and support provenance-backed explanation.

---

## 9. Two-Stage Encoding Protocol

### Stage 1: Pre-activation (Familiarity Detection)

**Duration:** $T_{pre} = 10$ ms  
**STDP gain:** $M = 0$ (no plasticity)

**Purpose:** Let network dynamics settle toward any existing attractor.

**Process:**
1. Present full multi-modal input (SQL + Graph spikes)
2. Run CA3 dynamics for 10 ms without weight changes
3. Count CA3 neurons that fired $\geq 1$ spike
4. If overlap with stored engram $> 3$ neurons: classify as **familiar**
5. Else: classify as **novel**

### Stage 2: Plasticity (Storage)

**Duration:** $T_{plastic} = 50$ ms  
**STDP gain:** $M = 1$ (familiar) or $M = 2$ (novel)

**Purpose:** Store or reinforce the engram.

**Process:**
1. Continue presenting the same input
2. Enable STDP with appropriate gain
3. For novel memories: strong potentiation creates new attractor basin
4. For familiar memories: weak reinforcement strengthens existing basin

**Why this works:** Pre-activation causes familiar inputs to fall into existing attractor basins. The same neurons pre-activate, so when plasticity turns on, those same neurons are strengthened — stabilizing engram reuse.

---

## 10. Retrieval Protocol

### 10.1 Partial Cue Retrieval

**Input:** Single modality cue (e.g., SQL row only)

**Process:**
1. Encode cue to spike pattern
2. Present to CA3 for $T_{retrieve} = 50$ ms
3. STDP gain $M = 0$ (no learning during retrieval)
4. Let recurrent dynamics settle
5. Measure CA3 activity pattern

### 10.2 Success Metrics

**Binding test:**
- Learned cue should activate more CA3 neurons than random control cue
- Difference indicates attractor completion (cross-modal reactivation)

**Specificity test:**
- Cue from different episode should NOT activate the target engram
- Jaccard overlap $< 20\%$ with non-matching engrams

**Exactness boundary:**
- Exact scalar facts remain the job of the source database.
- Neural retrieval should answer contextual questions such as "what episode resembles this cue?" not canonical fact lookups.

---

## 11. Test Specifications

### Test 1: Encoder Reconstructibility

**Objective:** Verify that SQL and Graph encoders preserve information.

**Procedure:**
1. Encode sample SQL row: `{'age': 30, 'salary': 85000, 'city': 'NYC', ...}`
2. Decode age and salary fields from spike counts
3. Encode graph edge: `(0, 'WORKS_AT', 1)`
4. Measure delay between source and target spikes

**Pass criteria:**
- Age decode error $< 10$ years
- Salary decode error $< 5000$
- Category decode exact match
- Graph delay within $\pm 1$ ms of target

### Test 2: CA3 Pattern Completion

**Objective:** Verify CA3 acts as an auto-associative memory.

**Procedure:**
1. Store 10 random sparse patterns (15 neurons each, 6% sparsity)
2. For each pattern, present 50% cue (8 neurons)
3. Measure which stored pattern has highest overlap with retrieved activity

**Pass criteria:**
- Retrieval accuracy $\geq 70\%$ (7/10 patterns correctly identified)
- Mean overlap with target pattern $\geq 0.5$

### Test 3: Multi-Modal Binding

**Objective:** Verify SQL + Graph presentation creates retrievable cross-modal association.

**Procedure:**
1. Learn episode: SQL(Alice) + Graph(WORKS_AT → Google)
2. Retrieve with SQL cue only
3. Compare active CA3 neurons to control (random SQL row)

**Pass criteria:**
- Learned cue activates $> 2$ more CA3 neurons than control
- Indicates attractor completion beyond cue-driven activity

### Test 4: Engram Stability

**Objective:** Verify same episode recruits consistent CA3 neurons across presentations.

**Procedure:**
1. Present identical episode 10 times
2. Record CA3 engram (set of active neurons) each time
3. Compute Jaccard overlap: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$

**Pass criteria:**
- Mean Jaccard overlap $\geq 0.80$ (vs. first engram)

---

## 12. Parameters Reference

### Global Simulation

| Parameter | Value | Description |
|-----------|-------|-------------|
| $dt$ | 0.1 ms | Euler integration timestep |
| $T_{pre}$ | 10.0 ms | Pre-activation duration |
| $T_{plastic}$ | 50.0 ms | Plasticity duration |
| $T_{retrieve}$ | 50.0 ms | Retrieval duration |

### LIF Neurons

| Parameter | RS | FS |
|-----------|-----|-----|
| $\tau_m$ | 20.0 | 15.0 |
| $v_{rest}$ | -70.0 | -70.0 |
| $v_{thresh}$ | -55.0 | -50.0 |
| $v_{reset}$ | -75.0 | -75.0 |
| $\tau_{ref}$ | 2.0 | 1.0 |

### Synapses

| Parameter | Value |
|-----------|-------|
| $\tau_{syn}$ | 3.0 ms |
| $w_{init}$ (input→CA3) | 0.5–0.6 |
| $w_{init}$ (E→E recurrent) | 0.2–0.4 |
| $w_{init}$ (E→I) | 0.3–0.5 |
| $w_{init}$ (I→E) | -1.2 to -0.8 |
| $w_{max}$ | 10.0 |
| $w_{min}$ | 0.0 |

### STDP

| Parameter | Value |
|-----------|-------|
| $A_+$ | 0.5 |
| $A_-$ | 0.48 |
| $\tau_+$ | 12.0 ms |
| $\tau_-$ | 12.0 ms |
| $M_{novel}$ | 2.0 |
| $M_{familiar}$ | 1.0 |
| $M_{pre}$ | 0.0 |

### Encoders

| Parameter | SQL | Graph | Text |
|-----------|-----|-------|------|
| Population size | 20/field | 20/node | Sparse semantic code, 5 active neurons/word |
| Total neurons | 100 | 80 | 100 |
| Numeric coding | Gaussian rate | N/A | N/A |
| Categorical coding | WTA hash (baseline; learned entity codes in V3/V5) | N/A | Semantic entity / token embeddings |
| Node coding | N/A | k-of-n (k=5) | N/A |
| Edge coding | N/A | Delay-coded | N/A |
| Semantic coding | N/A | N/A | Pretrained embedding → semantic event compression → phase code |
| Stimulation duration | 8–10 ms | 5–8 ms | 5 ms/word |

### CA3 Network

| Parameter | Value |
|-----------|-------|
| Excitatory neurons | 240 |
| Inhibitory neurons | 60 |
| E→E connectivity | 15% |
| E→I connectivity | 20% |
| I→E connectivity | 30% |
| Input→E connectivity | 8% |
| Target activity | 5–10% |

---

## 13. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Engram stability <80% | 20% | High | Increase $T_{pre}$ to 20 ms; add noise during pre-activation |
| Single-trial binding weak | 30% | Medium | Increase $w_{init}$ to 1.0; use $M=3$ for first presentation |
| DG not orthogonal enough | 15% | Medium | Train longer; increase expansion to 10× |
| Recurrent seizures | 10% | High | Monitor E/I ratio; add homeostatic inhibition |
| Graph delay ambiguity | 10% | Low | Wider delay separation (3/8/13 ms) |
| Timing jitter breaks STDP | 15% | Medium | Population coding ($N=20$) reduces jitter by $\sqrt{20} \approx 4.5×$ |
| STDP saturation / instability | 20% | High | Weight normalization, homeostatic plasticity, bounded STDP, metaplasticity |
| Catastrophic forgetting | 20% | High | Replay, consolidation, multiple memory timescales, synaptic stability |
| False attractors | 15% | Medium | Stronger DG separation, higher inhibition, sparse coding, measure FalseRetrievalRate |
| Missing modality | 20% | Medium | Modality dropout training across all modality subsets |
| Explainability gap | 15% | Medium | Provenance layer with episode IDs, source records, and timestamps |
| Throughput bottleneck | 25% | Medium | Event stream deduplication, aggregation, and novelty filtering before SNN ingestion |

---

## 14. Implementation Checklist

### Phase 1: Core Primitives (Week 1)
- [ ] LIF neuron with refractory period
- [ ] Delayed synapse with exponential PSC
- [ ] STDP with neuromodulatory gating
- [ ] Network simulation loop
- [ ] **Verify:** Single neuron fires with $I = 100$ pA input

### Phase 2: Encoders (Week 1)
- [ ] SQL encoder (rate + WTA)
- [ ] Graph encoder (ensemble + delay)
- [ ] **Test 1:** Reconstructibility passes

### Phase 3: CA3 Attractor (Week 2)
- [ ] Build recurrent connectivity
- [ ] Store random patterns
- [ ] Retrieve from partial cues
- [ ] **Test 2:** Pattern completion $\geq 70\%$

### Phase 4: Multi-Modal Integration (Week 3)
- [ ] Input pathways (SQL→CA3, Graph→CA3)
- [ ] Two-stage encoding (pre-activation → plasticity)
- [ ] **Test 3:** Binding retrieval works

### Phase 5: Stability (Week 4)
- [ ] Repeated presentation protocol
- [ ] Jaccard overlap measurement
- [ ] **Test 4:** Stability $\geq 80\%$

### Phase 6: Scale (Month 2–3)
- [ ] Modular sharding (100 CA3 modules)
- [ ] Conventional cortical autoencoder
- [ ] Replay-based consolidation

### Phase 7: System Hardening (V3–V5)
- [ ] Learned entity codes with collision handling and OOV fallback
- [ ] Log / quantile / multi-resolution numeric coding
- [ ] Modality dropout tests across SQL, Graph, and Text
- [ ] Throughput pipeline: event stream → dedupe → aggregation → novelty filter
- [ ] Provenance lookups for every stored episode
- [ ] Explicit exact-DB / neural-memory boundary
- [ ] False-attractor and FalseRetrievalRate reporting

---

## 15. Version Roadmap and Open Issues

The following items correspond to the user-provided problem list. Rows marked as implemented are already present in the current Python path, but the spec should keep them explicit so the architecture does not drift back toward the prototype shortcuts.

| # | Gap | Required change | Target | Current status |
|---|-----|-----------------|--------|----------------|
| 1 | Current text encoder is primitive | Pretrained semantic encoder → semantic event compression → spike conversion | V3 | Partial in code; spec updated |
| 2 | Symbolic retrieval shortcut | Remove dictionary lookup / engram injection; use cue → DG → CA3 recurrence → attractor | V2 | Enforced in current retrieval path |
| 3 | DG dictionary | Learn `W_DG` online | V2 | Implemented in code; keep it core |
| 4 | CA1 is incomplete | Build CA3 → CA1 → modality decoders | V3 | Implemented in code; keep it core |
| 5 | Fixed categorical hashing | Learned entity codes + collision management + OOV representation | V3/V5 | Open |
| 6 | Numeric encoding does not match enterprise distributions | Log coding, quantile coding, adaptive population coding, multi-resolution coding | V3 | Open |
| 7 | Graph explosion | Entity memory allocation, sparse dynamic codes, hierarchical graph encoding, hash-based routing, learned graph representations | V3/V5 | Open |
| 8 | Relation delay collisions | Composite temporal coding: base delay + phase + population identity + burst pattern | V3 | Partial; current relation bursts help but do not solve scale |
| 9 | CA3 capacity | Sparse recurrent connectivity with approximately `O(kN)` scaling | V2 | Partial; sparse CA3 exists, scaling plan remains |
| 10 | STDP instability | Weight normalization, homeostatic plasticity, synaptic scaling, metaplasticity, bounded STDP | V2/V3 | Partial; clipping + homeostasis exist |
| 11 | Catastrophic forgetting | Replay, consolidation, multiple memory timescales, synaptic stability | V5 | Partial; replay exists, long-horizon stability remains |
| 12 | False attractors | Increase DG separation, CA3 capacity, inhibition, sparse coding, orthogonality; measure `FalseRetrievalRate` | V3 | Open |
| 13 | False familiarity | Combine reconstruction error, pattern similarity, temporal context, entity novelty, and prediction error | V3 | Implemented in code path |
| 14 | Missing modality | Modality dropout training over SQL + Graph + Text combinations | V3 | Open |
| 15 | Temporal noise | Temporal windows, jitter tolerance, phase coding, population redundancy | V3 | Partial; phase and population coding exist |
| 16 | Enterprise throughput | Kafka / event stream → deduplication → aggregation → novelty filter → SNN | V3 | Open |
| 17 | Explainability | Parallel episodic provenance layer: CA3 engram → episode ID → source records → timestamps | V3 | Partial; provenance exists in `EpisodeRecord` |
| 18 | Exactness | Exact SQL remains authoritative; neural memory is for contextual recall only | Baseline / V5 | Must stay explicit in architecture |

---

## Appendix A: Mathematical Derivations

### A.1 Polychronous Group Capacity

For $N$ neurons with maximum delay $D_{max}$ and timing precision $\delta t$:

$$\mathcal{C} \sim N \binom{N}{k} \left(\frac{D_{max}}{\delta t}\right)^{k-1}$$

For $N = 240$, $k = 5$, $D_{max} = 10$ ms, $\delta t = 1$ ms:

$$\mathcal{C} \sim 240 \times \binom{240}{5} \times 10^4 \gg 2^{240}$$

### A.2 Hopfield Capacity (Comparison)

For synchronous binary patterns with sparsity $p$:

$$P_{max} \approx \frac{0.15 N}{k \log N}$$

where $k$ is the number of active neurons per pattern. For $N = 240$, $k = 15$:

$$P_{max} \approx \frac{0.15 \times 240}{15 \times \log 240} \approx 3$$

**The temporal dimension provides exponential capacity advantage.**

### A.3 STDP Convergence

For repeated presentation of the same pre-post pair with interval $\Delta t$:

$$w_{n+1} = w_n + A_+ e^{-\Delta t / \tau_+}$$

After $n$ trials:

$$w_n = w_0 + n A_+ e^{-\Delta t / \tau_+}$$

Saturation occurs at $w_{max}$ when $n \approx (w_{max} - w_0) / (A_+ e^{-\Delta t / \tau_+})$.

For $w_0 = 0.5$, $w_{max} = 10$, $A_+ = 0.5$, $\Delta t = 2$ ms, $\tau_+ = 12$ ms:

$$n_{sat} \approx 9.5 / (0.5 \times e^{-2/12}) \approx 22 \text{ trials}$$

---

## Appendix B: Biological Plausibility Notes

| Component | Biological Analog | Plausibility |
|-----------|-------------------|--------------|
| LIF neuron | Cortical pyramidal cell | High (standard model) |
| Delayed synapse | Axonal conduction delays | High (measured 0.1–20 ms) |
| STDP | Hebbian timing-dependent plasticity | High (observed across species) |
| Population coding | Cortical column coding | High (primary sensory areas) |
| DG sparse separator | Dentate gyrus granule cells | High (2–5% activity observed) |
| CA3 attractor | CA3 recurrent collaterals | High (Marr 1971, Rolls 2013) |
| Pre-activation | Entorhinal-hippocampal loop | Medium (theta-modulated gating) |
| ACh boost | Acetylcholine from medial septum | High (enhances LTP) |

---

## Appendix C: File Structure

```
project/
├── neurons.py          # LIF, Izhikevich
├── synapses.py         # DelayedSynapse with STDP
├── encoders.py         # SQLEncoder, GraphEncoder, TextEncoder
├── networks.py         # CA3Attractor, MultiModalMemory
├── tests.py            # Test 1-4 implementations
├── benchmark.py        # Full benchmark runner
├── config.yaml         # All parameters
└── notebooks/
    └── demo.ipynb      # Interactive demonstration
```

---

## References

1. Izhikevich, E.M. (2003). Simple model of spiking neurons. *IEEE Trans. Neural Networks*, 14(6), 1569-1572.
2. Izhikevich, E.M. (2006). Polychronization: Computation with spikes. *Neural Computation*, 18, 245-282.
3. O'Reilly, R.C., et al. (2014). Complementary Learning Systems. *Cognitive Science*, 38(6), 1229-1248.
4. Rolls, E.T. (2013). Mechanisms for pattern completion and separation. *Front. Systems Neuroscience*, 7, 74.
5. Caporale, N. & Dan, Y. (2008). Spike timing-dependent plasticity. *Annual Review of Neuroscience*, 31, 25-46.
6. Ferdaoussi, A.E., et al. (2024). Maximizing information in neuron populations. *Neuromorphic Computing and Engineering*.
7. Garg, N., et al. (2022). Voltage-dependent synaptic plasticity. *Frontiers in Neuroscience*, 16, 983950.

---

**End of Specification**
