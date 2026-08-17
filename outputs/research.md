# End-to-End Research Program for Polychronous Multimodal Episodic Memory

## 1. Research Objective

The project should no longer be evaluated as a collection of SNN components. It should be evaluated as a **memory system** with a falsifiable scientific claim:

> **Can a sparse, delay-heterogeneous, locally plastic spiking network bind heterogeneous structured, relational, and sequential information into a unified episodic memory, and subsequently retrieve or reconstruct that memory from partial or cross-modal cues without an explicit symbolic retrieval index?**

This is a materially different claim from “SNNs can perform graph reasoning.”

The closest work you identified establishes three important neighboring capabilities:

* **Brain-inspired Graph Spiking Neural Networks** uses population coding, STDP/R-STDP, and graph-spiking representations for commonsense knowledge representation and reasoning, using ConceptNet. ([arXiv][1])
* **GRSNN — Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning** demonstrates that synaptic delay can encode relation information and evaluates transductive/inductive KG reasoning on FB15k-237 and WN18RR, plus homogeneous graph link prediction on Cora, CiteSeer, and PubMed. It reports MRR/Hits metrics and analyzes energy/parameter efficiency. ([Proceedings of Machine Learning Research][2])
* **HippoMM** is the closest multimodal-memory competitor: it uses hippocampus-inspired episodic segmentation, memory consolidation, and cross-modal retrieval for long audiovisual streams, reporting 78.2% accuracy on HippoVlog and faster response time than the compared retrieval baseline. ([arXiv][3])

Your proposed system is broader:

[
\boxed{
SQL + Graph + Text + Time
\rightarrow
EC \rightarrow DG \rightarrow CA3 \rightarrow CA1
}
]

and its scientific novelty should be **neural binding and attractor-based retrieval**, rather than merely adding another modality.

---

# 2. The Most Important Decision: Do Not Use One Benchmark

There is currently no standard benchmark that simultaneously gives you:

[
\text{SQL}+\text{knowledge graph}+\text{text}+\text{episodic time}
]

with ground-truth cross-modal memory questions.

Therefore, the correct research strategy is a **benchmark suite**, not one dataset.

I would build the evaluation around **four tracks**.

---

# 3. Benchmark Track A — Controlled Neural Memory Benchmark

This is the most important benchmark for proving the actual mechanism.

Call it something like:

> **PMEM-Control: Polychronous Multimodal Episodic Memory Control Benchmark**

Do not start with real enterprise data here.

You need complete control over the generative process so that you can answer:

> Exactly what caused the network to succeed or fail?

## Dataset construction

Generate episodes:

```text
Episode 001

SQL:
person = Alice
age = 31
department = AI
salary = 85k

Graph:
Alice ──WORKS_AT──> Google

Text:
"Alice works in Google's AI division."
```

Then:

```text
Episode 002

SQL:
person = Alice
age = 31
department = AI
salary = 85k

Graph:
Alice ──WORKS_AT──> Microsoft

Text:
"Alice works in Microsoft's AI division."
```

The two episodes differ by only one relation.

This is precisely the kind of case where DG + CA3 should matter.

---

# 4. Generate Difficulty Levels

## Level 1 — One modality

```text
SQL → SQL
Graph → Graph
Text → Text
```

## Level 2 — Two modalities

```text
SQL + Graph
SQL + Text
Graph + Text
```

## Level 3 — Three modalities

```text
SQL + Graph + Text
```

## Level 4 — Partial cues

```text
SQL 30%
SQL 50%
SQL 70%
```

## Level 5 — Cross-modal cues

```text
SQL → Graph
Graph → SQL
Text → SQL
Text → Graph
SQL + Text → Graph
```

## Level 6 — Temporal episodes

```text
t1 Alice → Google
t2 Alice → Microsoft
t3 Alice → Amazon
```

Then query:

> Where did Alice work immediately before Microsoft?

---

# 5. Controlled Adversarial Cases

This benchmark is critical.

Generate pairs such as:

```text
A:
Alice → Google

B:
Alice → Microsoft
```

and:

```text
A:
Alice manages Bob

B:
Bob manages Alice
```

and:

```text
A:
Alice works at Google

B:
Alice does not work at Google
```

and:

```text
A:
Alice joined Google in 2024

B:
Alice joined Google in 2026
```

The network must distinguish them.

---

# 6. The Four Core Mechanistic Metrics

Before comparing against any competitor, demonstrate that the architecture itself works.

## 6.1 Pattern separation

For two inputs (i,j):

[
J_{input}(i,j)
==============

\frac{|X_i\cap X_j|}
{|X_i\cup X_j|}
]

and after DG:

[
J_{DG}(i,j)
]

Define separation gain:

[
\boxed{
SG =
1-
\frac{J_{DG}}{J_{input}}
}
]

Higher is better.

You want to show:

[
J_{DG}<J_{input}
]

for similar episodes.

---

# 7. Pattern Completion

Store an engram:

[
E={n_1,n_2,\ldots,n_k}
]

Then provide only a fraction (c) of it.

Test:

[
c\in
{0.1,0.2,0.4,0.6,0.8}
]

Measure:

[
J(E_{retrieved},E_{target})
]

and define:

[
CompletionAccuracy(c)
]

This should produce a curve:

```text
cue strength
10% ────────────────► 80%

retrieval
low                  high
```

You should report the **area under the completion curve**, not only one number.

---

# 8. Engram Stability

For repeated presentations:

[
E_1,E_2,\ldots,E_n
]

measure:

[
J(E_i,E_1)
]

and report:

* mean Jaccard,
* standard deviation,
* coefficient of variation,
* engram size variance.

The current implementation already uses repeated engram presentations and Jaccard stability, with the prototype reporting 0.91 mean overlap. 

---

# 9. Polychronous Binding Accuracy

This should become one of your signature metrics.

Suppose:

```text
SQL:
t = 0 ms

Graph:
t = 4 ms

Text:
t = 6 ms
```

with learned delays:

```text
SQL delay = 6 ms
Graph delay = 2 ms
Text delay = 0 ms
```

Then:

[
0+6=4+2=6+0=6ms
]

Measure:

[
\epsilon_{arrival}
==================

|t_i+d_i-t^*|
]

and report:

* mean absolute arrival error,
* standard deviation,
* coincidence probability within ±0.5/1/2 ms.

This tells you whether **temporal convergence is actually happening**.

---

# 10. STDP Causality Metric

Do not only report final accuracy.

Measure:

[
\Delta w
]

as a function of:

[
\Delta t
========

t_{post}-(t_{pre}+d)
]

Plot:

[
\Delta w=f(\Delta t)
]

Your expected result should look like:

```text
Δw
 ↑
 │ \████
 │   ███
 │     ██
 │       \________
 └────────────────────► Δt
      0
```

Then conduct a causal ablation:

```text
real timing
vs
shuffled timing
vs
zero delays
vs
random delays
```

If your central hypothesis is correct:

[
Performance_{real}

>

Performance_{shuffled}
]

---

# 11. Track B — Public Structured/Graph Memory Benchmark

Now move toward real-world information.

## Graph benchmark

Use:

### FB15k-237

GRSNN already uses FB15k-237 and reports filtered-ranking metrics such as MRR and Hits@K, making it an essential direct comparison. ([ResearchGate][4])

### WN18RR

Also directly used by GRSNN. ([ResearchGate][4])

### ConceptNet

This is especially important for comparison with **Brain-inspired Graph Spiking Neural Networks** because that work explicitly uses ConceptNet. ([brain-cog.network][5])

---

# 12. Graph Evaluation

Use standard:

[
MRR
]

[
Hits@1
]

[
Hits@3
]

[
Hits@10
]

and:

[
MR
]

These are exactly the metrics used by GRSNN for KG completion. ([ResearchGate][4])

But you should additionally measure:

### Temporal relation accuracy

Because your system claims something GRSNN also investigates:

[
relation
\rightarrow
temporal\ delay
]

Measure:

[
Accuracy_{relation}
]

and:

[
MAE_{delay}
]

---

# 13. Your Comparison with GRSNN Must Be Specific

Do not claim:

> “Our system beats GRSNN.”

That would be comparing different problems.

Instead:

### Experiment G1 — Reproduce GRSNN graph task

Run your graph component independently on:

* FB15k-237
* WN18RR
* Cora
* CiteSeer
* PubMed

using the same evaluation protocol.

Compare:

| Model          | MRR | Hits@1 | Hits@10 | Params | Spike rate | Energy |
| -------------- | --: | -----: | ------: | -----: | ---------: | -----: |
| R-GCN          |     |        |         |        |            |        |
| NBFNet         |     |        |         |        |            |        |
| GRSNN          |     |        |         |        |            |        |
| **PMEM-Graph** |     |        |         |        |            |        |

GRSNN reports strong graph results and specifically demonstrates that synaptic delay improves relation representation; that makes it a **direct baseline for your delay-coded graph module**, not necessarily your complete memory system. ([ResearchGate][4])

---

# 14. Track C — Long-Term Memory

This is where your system should compete with modern memory architectures.

## LongMemEval

Use:

> **LongMemEval**

It evaluates:

* information extraction,
* multi-session reasoning,
* knowledge updates,
* temporal reasoning,
* abstention.

The original benchmark contains 500 questions, and its maintainers now also provide LongMemEval-V2. ([arXiv][6])

This is highly relevant to your:

```text
episodic memory
continual learning
temporal reasoning
knowledge updates
```

---

# 15. LoCoMo

Use:

> **LoCoMo**

It contains long-term conversations with timestamps and annotations for QA and event summarization, and includes image-linked turns in the data format. ([GitHub][7])

This gives you:

```text
conversation
+
time
+
multimodal elements
+
long-term memory
```

---

# 16. LoCoMo-Plus

Because the field has moved beyond simple factual recall, also include **LoCoMo-Plus**.

It specifically targets cases where the cue and the remembered constraint are semantically disconnected and evaluates cognitive memory/constraint consistency. ([arXiv][8])

This is useful because your system should not merely memorize exact strings.

---

# 17. HippoVlog

This is your strongest external multimodal-memory benchmark.

HippoVlog contains:

* 25 long-form daily vlogs,
* 682 minutes of audiovisual content,
* 1,000 validated multiple-choice questions.

It evaluates audiovisual and semantic memory across long-form content. ([GitHub][9])

HippoMM reports 78.2% average accuracy and 20.4s average response time on this benchmark, compared with a Video RAG baseline at 64.2% and 112.5s. ([HippoMM][10])

That gives you an excellent **direct external memory competitor**.

---

# 18. Your Comparison with HippoMM

Do not attempt to compare your raw SNN directly against HippoMM's complete audiovisual stack.

Instead establish two tracks.

### Memory-only track

Convert audiovisual episodes into modality representations and compare:

```text
HippoMM memory
vs
PMEM memory
```

Measure:

* retrieval accuracy,
* cross-modal recall,
* temporal recall,
* memory footprint,
* latency.

### End-to-end task track

Your system receives the same raw modalities and must answer the same questions.

This is much harder.

For this track, report:

[
Accuracy
]

and:

[
Latency
]

separately.

HippoMM itself explicitly frames episodic segmentation, consolidation, and hierarchical/cross-modal retrieval as its core mechanisms, making this comparison particularly relevant. ([arXiv][3])

---

# 19. Track D — Structured Enterprise Benchmark

This is where I recommend building your **own benchmark**.

There is no public benchmark that perfectly matches:

[
SQL+Graph+Text+Time
]

as one unified episodic memory.

So create:

# PMEM-Enterprise

The benchmark should be generated from public sources rather than private company data.

---

# 20. PMEM-Enterprise Dataset

Each episode should contain:

```text
SQL snapshot
+
graph neighborhood
+
text evidence
+
timestamp
+
event type
+
entity identities
```

Example:

```text
Episode 3821

SQL:
employee:
    id=1829
    age=31
    department=AI
    salary=85000

Graph:
Alice ──WORKS_AT──> Google
Alice ──REPORTS_TO──> Bob

Text:
"Alice joined Google's AI group and reports to Bob."

Timestamp:
2026-03-11
```

---

# 21. How to Construct PMEM-Enterprise

The strongest route is a **multi-source synthetic-real benchmark**.

### Source 1 — YAGO

YAGO provides a cleaned public knowledge graph with human-readable identifiers and semantic constraints. YAGO 4.5 currently reports about 49 million entities and 109 million facts. ([Yago Project][11])

Use:

```text
entities
relations
types
labels
descriptions
```

to create graph episodes.

### Source 2 — Text

Use entity descriptions, relation evidence, and relevant public textual sources.

### Source 3 — Relational projection

Convert graph facts into normalized relational tables:

```text
Person
Company
Employment
Location
Department
```

Thus one underlying fact exists in:

```text
SQL
Graph
Text
```

simultaneously.

This creates exactly the environment your architecture needs.

---

# 22. Enterprise Event Generation

Add temporal modifications:

```text
2024:
Alice → Google

2025:
Alice becomes Manager

2026:
Alice → Microsoft
```

Now the same entity exists across multiple states.

This allows evaluation of:

* temporal reasoning,
* updates,
* contradiction,
* forgetting,
* episodic reconstruction.

---

# 23. Benchmark Data Splits

Do **not** randomly split rows.

That would create enormous leakage.

Use:

### Entity-disjoint split

Entities in test do not appear in training.

### Relation-disjoint split

Some relation types occur only in test.

### Temporal split

Train:

```text
2020–2024
```

Test:

```text
2025–2026
```

### Episode-disjoint split

Complete episodes are separated.

### Composition split

The model sees:

```text
Alice + Google
Bob + Microsoft
```

during training but:

```text
Alice + Microsoft
```

during testing.

This is extremely important.

It tests whether the network **binds compositional structure** rather than memorizing combinations.

---

# 24. Core Task 1 — Exact Memory Recall

Query:

```text
Who did Alice report to?
```

Expected:

```text
Bob
```

Metrics:

[
Accuracy
]

[
MRR
]

[
Recall@K
]

---

# 25. Core Task 2 — Cross-Modal Recall

Input:

```text
SQL only
```

Target:

```text
Graph edge
```

Measure:

[
Recall@1
]

[
Recall@5
]

[
MRR
]

This is your **signature task**.

---

# 26. Core Task 3 — Multimodal Recall

Input:

```text
Text + partial SQL
```

Target:

```text
full graph + missing SQL
```

Measure each modality independently:

[
Acc_{SQL}
]

[
Acc_{Graph}
]

[
Acc_{Text}
]

and a joint score:

[
JointAccuracy
=============

I(SQL\ correct)
I(Graph\ correct)
I(Text\ correct)
]

---

# 27. Core Task 4 — Temporal Recall

Question:

> Where did Alice work immediately before Microsoft?

This tests:

[
TemporalAccuracy
]

Also measure:

[
MAE_{time}
]

and event-order accuracy.

---

# 28. Core Task 5 — Knowledge Update

Sequence:

```text
Alice → Google
Alice → Microsoft
```

Query:

> Where does Alice currently work?

Correct answer:

```text
Microsoft
```

Also ask:

> Where did Alice previously work?

Correct:

```text
Google
```

This tests whether the system preserves **history rather than overwriting it**.

LongMemEval explicitly includes knowledge updates and temporal reasoning, so it is an important external validation point. ([arXiv][6])

---

# 29. Core Task 6 — Contradiction

Store:

```text
Alice works at Google.
```

Then:

```text
Alice does not work at Google.
```

The system should not simply average the memories.

It must identify:

```text
conflict
```

and use:

```text
time
source
confidence
context
```

to resolve it.

Metrics:

[
ConflictDetectionF1
]

[
TemporalResolutionAccuracy
]

---

# 30. Core Task 7 — Abstention

Give the system:

```text
What is Alice's blood type?
```

when that information was never stored.

Correct output:

```text
Unknown / insufficient evidence
```

This is extremely important.

A memory system that always retrieves something is dangerous.

LongMemEval explicitly evaluates abstention. ([arXiv][6])

---

# 31. Core Task 8 — Memory Capacity

Gradually increase:

```text
10
100
1K
10K
100K
1M
```

episodes.

Measure:

[
Accuracy(N)
]

[
FalseRetrievalRate(N)
]

[
Interference(N)
]

[
MemoryFootprint(N)
]

The primary graph should be:

```text
retrieval accuracy
      │
100%  │───────
      │       \
      │        \
      │         \
      │          \____
      └──────────────────► memories
```

The point where accuracy collapses is your empirical capacity.

---

# 32. Core Task 9 — Continual Learning

Stream episodes:

[
E_1,E_2,\ldots,E_N
]

Evaluate old memories periodically.

Define:

[
Retention(N)=
\frac{
\text{old memories still correctly retrieved}
}{
\text{old memories}
}
]

Then compare:

```text
STDP only
STDP + DG
STDP + DG + replay
STDP + DG + replay + homeostasis
```

This becomes one of your strongest experiments.

---

# 33. Core Task 10 — One-Shot Learning

Give:

```text
Episode E
```

**once**.

Immediately test retrieval.

Compare:

```text
1 presentation
2 presentations
5 presentations
10 presentations
```

Measure:

[
Acc(n_{presentations})
]

The original research motivation explicitly targets one-shot/fast attractor formation. 

---

# 34. Core Efficiency Metrics

Do not publish only accuracy.

You need:

### Parameter count

[
P
]

### Memory footprint

[
MB/10^3\ episodes
]

### Spike rate

[
spikes/sec
]

### Synaptic operations

[
SynOps/event
]

### Latency

[
ms/query
]

### Throughput

[
episodes/sec
]

### Energy

[
J/query
]

or:

[
J/episode
]

### Energy-delay product

[
EDP=Energy\times Latency
]

This is especially important for the SNN claim.

GRSNN explicitly reports spike rate, operation counts, parameter efficiency, and theoretical energy, so your paper should report comparable quantities rather than only accuracy. ([ResearchGate][4])

---

# 35. Biological Metrics

Since this is computational neuroscience as well as AI, report:

### Sparsity

[
S=1-\frac{\text{active neurons}}{N}
]

### Firing rate

[
r_i
]

### E/I balance

[
R_{EI}
======

\frac{Excitation}{|Inhibition|}
]

### Temporal precision

[
\sigma_{arrival}
]

### STDP distribution

[
P(\Delta w)
]

### Engram overlap

[
J(E_i,E_j)
]

### Attractor basin size

Fraction of perturbed cues that converge to the correct attractor.

---

# 36. The Most Important Metric for Your Paper

I would define a new metric:

# Cross-Modal Episodic Retrieval Accuracy — CERA

For an episode containing:

[
M=
{SQL,Graph,Text}
]

give only modality (i) and ask for modality (j).

Define:

[
CERA_{i\rightarrow j}
=====================

\frac{
\text{correctly reconstructed target elements}
}{
\text{target elements}
}
]

Then report the matrix:

| Cue → Target |  SQL | Graph | Text |
| ------------ | ---: | ----: | ---: |
| SQL          |    — |  CERA | CERA |
| Graph        | CERA |     — | CERA |
| Text         | CERA |  CERA |    — |

This becomes the **core signature figure of your paper**.

---

# 37. Another Important Metric: Binding Index

You also need to distinguish:

```text
the system retrieved the right memory
```

from:

```text
the system actually bound modalities together.
```

Define:

[
BI =
Acc_{cross-modal}
-----------------

Acc_{unimodal-control}
]

For example:

```text
SQL-only cue → Graph
full SQL+Graph control
```

If:

[
BI\gg0
]

then cross-modal association is contributing information beyond independent retrieval.

---

# 38. Another Important Metric: Polychronous Binding Gain

Run:

```text
normal delay
```

versus:

```text
all delays = 0
```

and:

```text
random delay
```

Define:

[
PBG=
Accuracy_{structured\ delay}
----------------------------

Accuracy_{random\ delay}
]

This directly tests your central hypothesis.

---

# 39. Required Ablation Matrix

Your paper should contain at least:

| Model | EC | DG | CA3 recurrence | Delay | STDP | Replay |
| ----- | -- | -- | -------------- | ----- | ---- | ------ |
| Full  | ✓  | ✓  | ✓              | ✓     | ✓    | ✓      |
| A     | ✗  | ✓  | ✓              | ✓     | ✓    | ✓      |
| B     | ✓  | ✗  | ✓              | ✓     | ✓    | ✓      |
| C     | ✓  | ✓  | ✗              | ✓     | ✓    | ✓      |
| D     | ✓  | ✓  | ✓              | ✗     | ✓    | ✓      |
| E     | ✓  | ✓  | ✓              | ✓     | ✗    | ✓      |
| F     | ✓  | ✓  | ✓              | ✓     | ✓    | ✗      |

The critical experiment is:

```text
Full model
vs
No temporal delay
```

because this directly attacks the paper's central claim.

---

# 40. Competitor Matrix

I would divide competitors into four groups rather than one leaderboard.

## Group A — Biological / SNN competitors

### 1. Brain-inspired Graph SNN

Tests:

```text
ConceptNet
commonsense reasoning
population coding
STDP/R-STDP
```

([arXiv][1])

Your advantage:

```text
multimodal
episodic
CA3 attractor
cross-modal retrieval
temporal memory
```

---

### 2. GRSNN

Tests:

```text
FB15k-237
WN18RR
Cora
CiteSeer
PubMed
```

Metrics:

```text
MRR
Hits@1
Hits@3
Hits@10
AUROC
AP
energy
parameters
```

([ResearchGate][4])

Your system should reproduce the **graph-only task**, then move beyond it.

---

## Group B — Hippocampal multimodal memory

### 3. HippoMM

Dataset:

```text
HippoVlog
```

Metrics:

```text
A+V
A
V
semantic
overall
latency
```

HippoMM reports 78.2% overall accuracy on its benchmark and 20.4s average response time in the reported setup. ([HippoMM][10])

This is probably your **closest architectural competitor**.

---

# 41. Group C — Modern Memory Systems

You need:

### LongMemEval baselines

Compare against:

* vector retrieval,
* BM25/hybrid retrieval,
* modern memory systems,
* long-context models.

LongMemEval evaluates information extraction, multi-session reasoning, knowledge updates, temporal reasoning, and abstention. ([GitHub][12])

### LoCoMo / LoCoMo-Plus

Use them for:

```text
long-term conversation
temporal memory
implicit constraints
cross-session recall
```

LoCoMo provides long conversations with timestamps and multimodal elements, while LoCoMo-Plus targets semantic/cognitive memory beyond surface factual recall. ([GitHub][7])

---

# 42. Group D — Conventional Strong Baselines

You absolutely need these.

## Vector-memory baseline

```text
embedding
→ FAISS/HNSW
→ top-k
→ reranking
```

## Hybrid baseline

```text
BM25
+
dense embedding
+
metadata filtering
```

## Graph baseline

```text
GraphRAG
```

## Classical associative memory

```text
Hopfield
modern Hopfield
```

## Neural sequence baseline

```text
Transformer
```

## SNN baseline

```text
LIF SNN
no delay
no recurrent CA3
```

The purpose is not to beat every model everywhere.

The purpose is to establish:

> **What does this system gain from being polychronous, recurrent, sparse, and locally plastic?**

---

# 43. The Correct Comparison Protocol

Never compare:

```text
your SNN
```

against:

```text
LLM
```

on a task where your SNN has only structured inputs and the LLM sees the raw text plus all context.

That is invalid.

Instead:

### Same information

```text
same episodes
same modalities
same train/test split
same query
same target
```

### Three deployment budgets

#### Budget A — Parameter-matched

Same parameter count.

#### Budget B — Memory-matched

Same memory footprint.

#### Budget C — Compute-matched

Same inference compute budget.

Then compare:

[
Accuracy
]

and:

[
Energy
]

and:

[
Latency
]

---

# 44. You Need Three Separate Leaderboards

## Leaderboard 1 — Memory quality

```text
Recall@1
Recall@5
MRR
NDCG
CERA
Temporal accuracy
Abstention
```

## Leaderboard 2 — Neural properties

```text
sparsity
Jaccard stability
polychronous binding gain
STDP stability
attractor basin
capacity
interference
```

## Leaderboard 3 — Efficiency

```text
parameters
memory MB
SynOps
spikes
latency
throughput
energy
EDP
```

This prevents an enormous model from winning simply because it has more computation.

---

# 45. Your Final Experimental Table

Your main paper table should eventually look approximately like:

| Model           | CERA | Temporal | Completion | Capacity | Continual Retention | Params | Energy | Latency |
| --------------- | ---: | -------: | ---------: | -------: | ------------------: | -----: | -----: | ------: |
| Vector Memory   |      |          |            |          |                     |        |        |         |
| GraphRAG        |      |          |            |          |                     |        |        |         |
| Hopfield        |      |          |            |          |                     |        |        |         |
| Modern Hopfield |      |          |            |          |                     |        |        |         |
| SNN-no-delay    |      |          |            |          |                     |        |        |         |
| GRSNN           |      |          |            |          |                     |        |        |         |
| HippoMM         |      |          |            |          |                     |        |        |         |
| **PMEM**        |      |          |            |          |                     |        |        |         |

But **do not fill cells with incompatible measurements**. Each competitor should only be compared on tasks it actually supports.

---

# 46. The Figure Set I Would Target for the Paper

A serious paper could contain approximately these figures.

### Figure 1 — Architecture

```text
SQL / Graph / Text
        ↓
EC → DG → CA3 → CA1
```

### Figure 2 — Polychronous binding

Show:

```text
SQL spikes
Graph spikes
Text spikes
```

converging at the same CA3 neuron.

### Figure 3 — STDP

[
\Delta w\ vs\ \Delta t
]

### Figure 4 — Pattern separation

Before/after DG similarity distributions.

### Figure 5 — Attractor basins

Partial cues projected into CA3 and colored by recovered memory.

### Figure 6 — Cross-modal retrieval matrix

[
CERA_{i\rightarrow j}
]

### Figure 7 — Memory capacity

[
Accuracy\ vs\ number\ of\ memories
]

### Figure 8 — Continual learning

[
Retention\ vs\ time
]

with and without replay.

### Figure 9 — Energy/latency

PMEM vs GRSNN vs vector memory vs conventional neural baseline.

### Figure 10 — Ablations

Contribution of:

```text
delay
STDP
DG
CA3 recurrence
EC
replay
```

---

# 47. What Would Count as a Real Research Result?

I would **not** consider:

```text
10/10 synthetic pattern completion
```

enough.

I would consider the work genuinely interesting if you can demonstrate all of the following:

### Result 1

Structured delays outperform randomized/zero-delay controls:

[
PBG>0
]

with statistical significance.

### Result 2

DG improves separation:

[
J_{DG}<J_{input}
]

while CA3 improves completion:

[
J_{retrieved,target}>J_{cue,target}
]

### Result 3

Cross-modal retrieval works:

[
SQL\rightarrow Graph
]

[
Graph\rightarrow SQL
]

[
Text\rightarrow Graph
]

without symbolic lookup.

### Result 4

The performance survives increasing memory load.

### Result 5

Replay prevents catastrophic forgetting.

### Result 6

The system retains a meaningful efficiency advantage.

### Result 7

Removing temporal delays causes a significant degradation.

That final result is especially important because otherwise the paper could simply be interpreted as:

> “another sparse associative memory implemented with spikes.”

---

# 48. The Scientific Hypothesis You Should Ultimately Test

Your paper should revolve around something this precise:

[
\boxed{
H_1:
\text{Delay-heterogeneous CA3 networks bind asynchronous modalities more effectively than identical networks without temporal coding.}
}
]

Then:

[
\boxed{
H_2:
\text{DG sparse separation improves multimodal episodic capacity and reduces interference.}
}
]

[
\boxed{
H_3:
\text{CA3 recurrence produces genuine cross-modal pattern completion from partial cues.}
}
]

[
\boxed{
H_4:
\text{Local STDP + novelty modulation enables one/few-shot continual memory.}
}
]

[
\boxed{
H_5:
\text{Replay-based consolidation reduces catastrophic forgetting under long streaming workloads.}
}
]

And:

[
\boxed{
H_6:
\text{The combination provides an efficiency advantage under event-driven/neuromorphic execution.}
}
]

These are experimentally falsifiable.

---

# 49. The actual research roadmap

I would execute this in **seven experimental phases**.

## Phase 1 — Mechanistic validation

```text
LIF
↓
delay
↓
polychronization
↓
STDP
```

Deliverable:

[
\Delta w(\Delta t)
]

and temporal convergence figures.

---

## Phase 2 — CA3 memory

Implement **pure neural retrieval**.

No:

```text
signature lookup
engram injection
symbolic completion
```

Deliverables:

* attractor basin measurements,
* completion curves,
* false-attractor rate,
* memory capacity.

---

## Phase 3 — Multimodal binding

Use:

```text
SQL
Graph
Text
```

Deliverable:

[
CERA
]

cross-modal retrieval matrix.

---

## Phase 4 — Public benchmarks

Run:

```text
FB15k-237
WN18RR
ConceptNet
LoCoMo
LongMemEval
LoCoMo-Plus
HippoVlog
```

where applicable.

The graph tasks give direct comparability with the two SNN competitors, while the long-term/multimodal memory benchmarks give comparability with HippoMM and modern memory systems. ([brain-cog.network][5])

---

## Phase 5 — PMEM-Enterprise

Construct the aligned:

[
SQL+Graph+Text+Time
]

benchmark.

This becomes your **main novel benchmark**.

---

## Phase 6 — Continual memory

Run:

```text
10K
100K
1M
```

episodes.

Measure:

```text
retention
interference
forgetting
capacity
energy
```

with and without replay.

---

## Phase 7 — Neuromorphic/efficiency study

Implement:

```text
event-driven simulator
```

then map to:

```text
GPU
neuromorphic hardware
```

and measure:

[
J/episode,\quad ms/query,\quad SynOps/query
]

rather than relying on theoretical energy alone.

GRSNN's energy analysis is a useful template here because it separately reports spike rate, operation estimates, memory overhead, and theoretical energy rather than treating “SNN = efficient” as sufficient evidence. ([ResearchGate][4])

---

# 50. The Final Research Stack

The end state should therefore be:

```text
                         PMEM RESEARCH PROGRAM
                                  │
              ┌───────────────────┼────────────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
        MECHANISTIC             MEMORY             APPLICATION
        BENCHMARKS             BENCHMARKS          BENCHMARKS
              │                   │                    │
       ┌──────┼──────┐       ┌────┼────┐          ┌────┼─────┐
       │      │      │       │    │    │          │    │     │
     STDP   Delay   DG     LoCoMo LongMem HippoVlog SQL Graph Text
       │      │      │         │    │      │          │    │
       └──────┼──────┘         └────┼──────┘          └────┼──┘
              │                     │                    │
              └──────────────┬──────┴────────────────────┘
                             ▼
                     PMEM-Enterprise
                             │
                 ┌───────────┼────────────┐
                 ▼           ▼            ▼
             Accuracy     Capacity     Efficiency
                 │           │            │
                 └───────────┼────────────┘
                             ▼
                    Ablation + Statistics
                             │
                             ▼
                      Scientific Claim
```

## The most important strategic point

Your **main competitor is not actually GRSNN**.

GRSNN is a competitor for the **graph/delay component**. The brain-inspired ConceptNet SNN is a competitor for the **population/STDP graph representation**. HippoMM is the closest competitor for **hippocampal-inspired multimodal memory**. Modern memory systems and vector/graph retrieval systems are the engineering baselines. ([arXiv][1])

Your actual claim is narrower and potentially more interesting:

> **Can temporally structured spike dynamics and local plasticity provide a unified associative memory mechanism for heterogeneous modalities, where cross-modal retrieval emerges from recurrent neural dynamics rather than from an explicit retrieval index?**

Everything in the experimental design should be arranged to answer that question.

And the **single experiment I would prioritize above everything else** is:

```text
Train:
SQL + Graph + Text
        ↓
     one CA3 engram

Test:
SQL only
 ↓
EC → DG → CA3
 ↓
CA3 settles WITHOUT symbolic lookup
 ↓
CA1
 ↓
Graph + Text reconstructed
```

Then repeat the exact same experiment with:

```text
1. no delay
2. random delay
3. no STDP
4. no DG
5. no CA3 recurrence
6. full model
```

If the full model wins **statistically and consistently**, that is the experiment that begins turning the architecture from an interesting implementation into a real research contribution.

[1]: https://arxiv.org/abs/2207.05561 "[2207.05561] Brain-inspired Graph Spiking Neural Networks for Commonsense Knowledge Representation and Reasoning"
[2]: https://proceedings.mlr.press/v235/xiao24f.html "Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning"
[3]: https://arxiv.org/abs/2504.10739 "[2504.10739] HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding"
[4]: https://www.researchgate.net/publication/380906704_Temporal_Spiking_Neural_Networks_with_Synaptic_Delay_for_Graph_Reasoning?utm_source=chatgpt.com "(PDF) Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning"
[5]: https://www.brain-cog.network/docs/examples/Knowledge_Representation_and_Reasoning/CKRGSNN.html?utm_source=chatgpt.com "Commonsense Knowledge Representation SNN — braincog 0.2.7.11 文档"
[6]: https://arxiv.org/abs/2410.10813?utm_source=chatgpt.com "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
[7]: https://github.com/snap-research/locomo?utm_source=chatgpt.com "GitHub - snap-research/locomo · GitHub"
[8]: https://arxiv.org/abs/2602.10715?utm_source=chatgpt.com "Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents"
[9]: https://github.com/linyueqian/HippoVlog/?utm_source=chatgpt.com "GitHub - linyueqian/HippoVlog · GitHub"
[10]: https://hippomultimodalmemory.github.io/?utm_source=chatgpt.com "HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding"
[11]: https://yago-knowledge.org/downloads/yago-4-5?utm_source=chatgpt.com "Downloads/yago 4 5 | Yago Project"
[12]: https://github.com/xiaowu0162/LongMemEval?utm_source=chatgpt.com "GitHub - xiaowu0162/LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (ICLR 2025) · GitHub"
