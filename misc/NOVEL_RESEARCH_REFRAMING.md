# Novel Research Framing and Implementation Assessment

This note is intended to answer three questions clearly:

1. What the current implementation actually does
2. Whether the current system is genuinely novel
3. Where the phrases “engram injection,” “symbolic retrieval,” “signature mapping,” and “familiarity bookkeeping” came from, and how to replace them with more defensible research language

---

## 1. What the current implementation is doing

The current system is a multimodal episodic memory model built around a recurrent spiking architecture with plasticity and novelty control. At a high level, it does the following.

### A. Contextual novelty gate
The system measures whether an incoming support pattern is novel or familiar before deciding how to update memory.

- The EC layer computes a support signature from active modality channels.
- It checks whether that pattern has been seen before using `seen_signatures`.
- The novelty score is then combined with energy deviation and prediction error to decide a modulation state.

Evidence in code:
- [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L619-L661)
- [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1514-L1538)

This is effectively a novelty gate. If the current event is novel, the network increases the modulation and allocates or strengthens a memory trace. If it is familiar, the system routes the event through a matching or consolidation path rather than creating a new memory.

### B. Sparse latent support encoding
The model converts multimodal inputs into sparse support vectors and projects them into the DG/CA3 memory pathway.

- SQL, graph, and text support are converted into sparse activity patterns.
- The DG layer performs a sparse projection and online adaptation.
- The result is a sparse support representation that is used to drive CA3 reactivation and association.

This is not a symbolic index. It is a distributed sparse code encoding the support structure of the episode.

### C. Assembly reinforcement / memory consolidation
The system does not inject a symbolic engram into a dictionary. Instead, it activates a CA3 assembly and then reinforces the recurrent associative structure.

- In encoding, CA3 is reset and run under a cue stream.
- Familiarity is estimated from overlap with stored CA3 assemblies.
- If the pattern is familiar, the stored assembly is updated.
- If the pattern is novel, a new memory assembly is appended.

Evidence in code:
- [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1569-L1597)
- [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1147-L1148)

This is not a symbolic injection process; it is a plastic recurrent assembly update process.

### D. Attractor-driven retrieval
Retrieval is performed by cue-driven activation and recurrent settling, followed by CA1 decoding.

- Partial SQL, graph, or text cues are used to generate inputs.
- CA3 is run with the cue stream and allowed to settle.
- CA1 decoders reconstruct the associated SQL, graph, and text outputs.

Evidence in code:
- [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1660-L1696)

This is attractor-based completion, not symbolic lookup.

---

## 2. Is the current implementation novel?

### Short answer
It is novel in the sense that the whole system is a coherent architecture combining:

- multimodal spike-based encoding,
- novelty-gated memory allocation,
- recurrent CA3 assembly formation,
- cross-modal reconstruction,
- partial-cue retrieval without a dictionary lookup path.

### But it is not radically original in the abstract
The field already has related ideas in several areas:

- episodic memory in hippocampal-inspired models,
- novelty detection and modulation in neuromodulatory systems,
- sparse coding and attractor networks,
- recurrent memory retrieval in CA3-like architectures,
- multimodal associative memory.

So the implementation is not “entirely new” as an idea, but it can still be novel as a system-level integration and evaluation framework.

### Where the novelty actually resides
The real novelty is not the existence of these mechanisms alone, but how they are combined:

- novelty-gated episodic encoding,
- online sparse support coding,
- recurrent CA3 assembly reinforcement,
- cross-modal retrieval under partial signal loss,
- explicit evaluation of false retrieval, graph reconstruction, and multimodal binding.

That combination is more defensible as a research contribution than the older labels.

---

## 3. If it isn’t novel, how can it be improved?

The architecture is already structurally strong, but the main weakness is conceptual framing rather than mechanism alone. The system can be improved by making the following additions or shifts.

### A. Replace “implementation labels” with “mechanistic claims”
The current code contains practical implementation terms, which can read as ad hoc or engineering-oriented. To make the work stronger, present the mechanisms as scientific hypotheses:

- Instead of “familiarity bookkeeping,” describe a novelty-sensitive memory gate.
- Instead of “signature mapping,” describe sparse latent support encoding.
- Instead of “engram injection,” describe assembly reinforcement or recurrent consolidation.
- Instead of “symbolic retrieval,” describe attractor-driven completion.

### B. Add a stronger theoretical contribution
The research becomes more compelling if it states a clear claim such as:

> episodic memory is formed through a novelty-gated sparse latent code that recurs through CA3 attractors, allowing modular cross-modal reconstruction under partial cue degradation.

This turns the system from an engineering implementation into a model of memory dynamics.

### C. Strengthen the novelty with a clear ablation story
The strongest way to make the system feel genuinely new is to run and present ablations around:

- novelty gate on/off,
- sparse support encoding vs dense encoding,
- assembly reinforcement vs no reinforcement,
- attractor retrieval vs direct symbolic matching,
- multimodal cue dropout vs unimodal cue input.

This gives a clean story of why each mechanism matters.

### D. Make the memory path truly non-symbolic
Your spec already points to this direction. The current path is already close to that goal, but the paper should present it more explicitly:

> The system avoids symbolic indexing; memory identity emerges from recurrent attractor dynamics and sparse population patterns.

That is a much stronger research claim than saying “there is no symbolic retrieval.”

---

## 4. Where the old phrases came from

These phrases did not arise from some hidden formalism in the codebase. They are mostly descriptive shorthand drawn from the implementation and the design notes.

### “engram injection”
This phrase likely came from the fact that the system stores and updates CA3 assemblies as episodic memory traces.

- Evidence: `self.engrams` in [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1147-L1148)
- Update logic: [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1571-L1597)

The real mechanism is not injecting a symbolic key. It is updating a learned assembly in CA3 based on overlap and reinforcement.

### “symbolic retrieval”
This phrase is likely a conceptual warning, not a literal implementation. It reflects the concern that a model might fallback to dictionary-like lookup or an explicit index.

- The spec explicitly calls this out as a gap and then states it is avoided in the current retrieval path: [SPIKING_MEMORY_SPEC.md](SPIKING_MEMORY_SPEC.md#L643-L643)

So this phrase came from a design critique: “avoid symbolic lookup.” It is not a real code path in the model itself.

### “signature mapping”
This likely came from the fact that the EC layer tracks a binary or sparse code signature of active support and tests whether it has been seen before.

- Evidence: `seen_signatures` and support hashing in [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L619-L661)

This is really a novelty signal or sparse support encoding, not a literal symbol map.

### “familiarity bookkeeping”
This phrase comes from the explicit familiarity comparison function.

- Evidence: [spiking_multimodal_memory.py](spiking_multimodal_memory.py#L1514-L1538)

The system literally checks whether current CA3 activity overlaps with known engrams. That is a familiarity test, but describing it as “bookkeeping” makes it sound like a data-structure layer. In reality it is a neural familiarity gate.

---

## 5. Better research language for your current implementation

Here is a more defensible and publication-ready description:

> The model implements a novelty-gated episodic memory system in which sparse multimodal support codes drive recurrent CA3 assembly formation. Familiarity is evaluated through overlap with previously stored assemblies, and the system selectively reinforces or updates memory traces depending on the novelty of the input. Retrieval is performed via attractor-driven completion, where partial multimodal cues converge to a stable recurrent memory state and CA1 decoders reconstruct the associated modality-specific content.

This captures the actual behavior while avoiding the weaker implementation-style language.

---

## 6. Final recommendation

If your goal is to make the work sound genuinely novel without changing the architecture too much, do this:

- keep the mechanisms,
- rename the conceptual language,
- emphasize the system-level novelty,
- present the architecture as a recurrence-driven, novelty-gated, sparse multimodal memory model.

This is the strongest and safest path.

### Recommended replacement set
- “engram injection” → “assembly reinforcement”
- “symbolic retrieval” → “attractor-driven retrieval”
- “signature mapping” → “sparse support encoding” or “latent support encoding”
- “familiarity bookkeeping” → “contextual novelty gate”

That gives you both scientific credibility and a stronger story for publication.

---

## 7. One-line research claim

> We propose a novelty-gated, sparse recurrent episodic memory that binds multimodal evidence through assembly reinforcement and retrieves memories via attractor-driven completion rather than symbolic indexing.
