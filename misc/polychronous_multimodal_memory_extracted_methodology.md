# Polychronous Multi-Modal Spiking Memory — Extracted Methodology, LaTeX Equations, Tables, and Experimental Results

## 0. Source scope

This extraction is based on the uploaded notebook `spiking_multimodal_memory(3).ipynb` and implementation `spiking_multimodal_memory.py`.

Important distinction:

- The notebook itself contains only a small amount of explicit research prose; it mainly demonstrates versions v2–v6 of the implementation.
- The source files contain **very little literal LaTeX**. The equations below are therefore **LaTeX renderings reconstructed directly from the mathematical operations implemented in the Python source**, not equations that were already written as LaTeX in the notebook.
- Experimental values below are copied from the notebook execution outputs.

---

# 1. System title and high-level methodology

## 1.1 Title

**Polychronous Multi-Modal Spiking Memory**

The implementation describes the complete pathway as:

\[
\text{Encoders} \rightarrow \text{EC} \rightarrow \text{DG} \rightarrow \text{CA3} \rightarrow \text{CA1}
\]

The notebook describes the development roadmap as:

- **v2:** pure neural retrieval without symbolic shortcuts
- **v3:** multimodal association, semantic text encoding, and CA1 readout
- **v4:** episodic sequencing and provenance
- **v5:** continual memory, consolidation, and false-retrieval checks
- **v6:** research harness with dropout, separation, continuity, and exactness metrics

---

# 2. Architectural table

| Component | Role | Dimension / configuration | Main mechanism |
|---|---|---:|---|
| SQL encoder | Encode structured records | 100 neurons | 5 fields × 20-neuron population codes |
| Graph encoder | Encode graph edges | 80 neurons | 4 nodes × 20-neuron sparse ensembles + temporal relation delays |
| Text encoder | Encode natural-language text | 100 neurons | 5% sparse semantic code + phase-of-firing |
| Entorhinal Cortex (EC) | Multimodal convergence + novelty | 180 without text; 280 with text | Rate vectors, support signatures, energy and novelty |
| Dentate Gyrus (DG) | Pattern separation | 1200 in main system default; configurable | Random projection + k-WTA sparse code |
| CA3 | Recurrent attractor / engram | 240 excitatory + 60 inhibitory by default | LIF neurons, delayed synapses, recurrent connectivity, STDP |
| CA1 | Reconstruction / readout | 220 without text; 320 with text by default | Linear readout + ReLU + relation classifier |
| DG→CA3 bridge | Recall pathway | Configurable fan-out | Sparse feed-forward input to CA3 |
| Episode store | Provenance / replay / temporal queries | One record per episode | SQL, graph, text, energy, neuromodulator, assemblies, spikes, predecessor/successor |

---

# 3. Core neural primitive: LIF neuron

The implementation uses a discrete-time Leaky Integrate-and-Fire update.

## 3.1 Continuous-form interpretation

From the implemented Euler update:

\[
\Delta v_t =
\frac{-(v_t-v_{\mathrm{rest}})+I_{\mathrm{syn}}(t)}
{\tau_m}\Delta t
\]

so the corresponding continuous-time form is:

\[
\tau_m \frac{dv}{dt}
=
-(v-v_{\mathrm{rest}})+I_{\mathrm{syn}}(t).
\]

## 3.2 Euler update used by the implementation

\[
v_{t+\Delta t}
=
v_t+
\frac{-(v_t-v_{\mathrm{rest}})+I_{\mathrm{syn}}(t)}
{\tau_m}\Delta t.
\]

## 3.3 Spike condition

A spike occurs when:

\[
v_t \geq v_{\mathrm{th}} + b_i,
\]

where \(b_i\) is the homeostatic offset.

After a spike:

\[
v \leftarrow v_{\mathrm{reset}},
\]

and a refractory interval is enforced:

\[
t_{\mathrm{refractory}} \leftarrow t+\tau_{\mathrm{ref}}.
\]

## 3.4 Default LIF parameters

| Parameter | Excitatory CA3 cells | Inhibitory CA3 cells |
|---|---:|---:|
| \(\tau_m\) | 20 ms | 15 ms |
| \(v_{\mathrm{rest}}\) | -70 mV | -70 mV |
| \(v_{\mathrm{th}}\) | -55 mV | -50 mV |
| \(v_{\mathrm{reset}}\) | -75 mV | -75 mV |
| \(\tau_{\mathrm{ref}}\) | 2 ms | 1 ms |
| \(\Delta t\) | 0.1 ms | 0.1 ms |

---

# 4. Delayed synapse and postsynaptic current

Each synapse has:

- pre-synaptic identity
- post-synaptic neuron
- weight \(w\)
- delay \(d\)
- synaptic time constant \(\tau_{\mathrm{syn}}\)
- gain
- weight bounds \([w_{\min},w_{\max}]\)

The implementation states that an original alpha kernel is approximated by a cached exponential trace.

## 4.1 Exponential synaptic trace

The cached trace follows:

\[
s(t+\Delta t)
=
s(t)\exp\left(-\frac{\Delta t}{\tau_{\mathrm{syn}}}\right)
+\sum_k w_k\,\mathbf{1}[t_k+d_k\le t].
\]

The returned postsynaptic current is:

\[
I_{\mathrm{syn}}(t)
=
g\,s(t),
\]

where \(g\) is the synaptic gain.

## 4.2 Default synapse parameters

| Parameter | Default |
|---|---:|
| \(\tau_{\mathrm{syn}}\) | 3 ms |
| \(w_{\max}\) | 10 |
| \(w_{\min}\) | 0 |
| Gain | 1 unless overridden |
| \(A_+\) | 0.5 |
| \(A_-\) | 0.48 |
| \(\tau_+\) | 12 ms |
| \(\tau_-\) | 12 ms |

---

# 5. STDP methodology

The update is applied when the post-synaptic neuron spikes.

Define the delay-adjusted spike-time difference:

\[
\Delta t
=
t_{\mathrm{post}}
-
(t_{\mathrm{pre}}+d).
\]

The update implemented is:

\[
\Delta w =
\begin{cases}
M A_+ \exp\left(-\dfrac{\Delta t}{\tau_+}\right),
& \Delta t\ge 0,\\[8pt]
-M A_- \exp\left(\dfrac{\Delta t}{\tau_-}\right),
& \Delta t<0.
\end{cases}
\]

and

\[
w \leftarrow
\operatorname{clip}
\left(
w+\Delta w,\,
w_{\min},\,
w_{\max}
\right).
\]

The neuromodulatory state \(M\) is explicitly interpreted in the source as:

- \(M=0\): no plasticity
- \(M=1\): normal plasticity
- \(M=2\): boosted plasticity

---

# 6. SQL / structured-data encoder

## 6.1 Input fields

The structured encoder uses five fields:

\[
\{\text{age},\text{salary},\text{city},\text{role},\text{dept}\}.
\]

Each field receives 20 neurons:

\[
5\times20=100.
\]

## 6.2 Numeric population rate coding

For a numeric variable \(x\), neuron \(i\) centered at \(c_i\) uses a Gaussian tuning curve:

\[
r_i(x)
=
r_{\mathrm{base}}
\exp\left(
-\frac{(x-c_i)^2}{2\sigma^2}
\right).
\]

Expected spike count over an encoding duration \(T\) is:

\[
\lambda_i
=
r_i(x)\frac{T}{1000}.
\]

The implementation rounds and clips the spike count:

\[
N_i
=
\operatorname{clip}
\left(
\operatorname{round}(\lambda_i),
0,
N_{\max}
\right),
\]

with \(N_{\max}=5\).

Spike times are sampled uniformly within the encoding interval.

## 6.3 Age profile

- centers: 20 to 80, 20 neurons
- \(\sigma=8\)
- base rate \(=240\) Hz
- clipping interval: \([0,120]\)

## 6.4 Salary: multi-resolution / two-band encoding

Salary uses two bands.

### Low band

\[
x_{\mathrm{low}}
=
\operatorname{clip}(x,0,10^6).
\]

- 10 neurons
- centers uniformly spanning \(0\) to \(10^5\)
- \(\sigma=15000\)
- base rate \(=280\) Hz

### High / tail band

\[
x_{\mathrm{high}}
=
\log(1+x).
\]

- 10 neurons
- centers spanning \(\log(1+10^5)\) to \(\log(1+10^{10})\)
- \(\sigma=0.9\)
- base rate \(=220\) Hz

The inverse transform is:

\[
x=\exp(z)-1.
\]

The full salary range is clipped to:

\[
0\le x\le 10^{10}.
\]

## 6.5 Categorical sparse code

Categorical values are mapped through a deterministic sparse codebook.

A candidate code is generated from a deterministic hash seed and selects \(k\) distinct neurons from a field-specific 20-neuron population.

For code \(C_y\subset\{1,\ldots,20\}\):

\[
|C_y|=k,
\]

with the configured default:

\[
k=3
\]

for categorical fields.

Collision-aware retries are attempted up to a bounded number of attempts; a reserved OOV code is used as fallback.

## 6.6 Numeric decoding

For numeric fields, the reconstructed value is a weighted average of neuron centers:

\[
\hat{x}
=
\frac{\sum_i n_i c_i}
{\sum_i n_i+\epsilon},
\]

followed by the corresponding inverse transform.

For salary, the low and high bands are decoded separately and the high-band estimate is selected when its total support is sufficiently larger than the low-band support.

---

# 7. Graph encoder

The graph encoder represents an edge:

\[
e=(u,r,v),
\]

where \(u\) is the source node, \(r\) is the relation, and \(v\) is the target node.

## 7.1 Architecture

| Item | Value |
|---|---:|
| Number of nodes | 4 |
| Neurons per node | 20 |
| Active neurons per node | 5 |
| Total graph neurons | 80 |
| Relation-code neurons | 3 per relation |

## 7.2 Relation delays

| Relation | Delay (ms) |
|---|---:|
| WORKS_AT | 3 |
| FRIENDS_WITH | 7 |
| MANAGES | 5 |
| REPORTS_TO | 2 |

## 7.3 Temporal code

Let the base time be:

\[
t_0=1\text{ ms}.
\]

Source node activity:

\[
t_{\mathrm{src}}=t_0.
\]

Relation burst:

\[
t_{\mathrm{rel}}
=
t_0+\frac{d_r}{2}.
\]

Target node activity:

\[
t_{\mathrm{tgt}}
=
t_0+d_r.
\]

Thus the graph relation is represented explicitly in spike timing, not solely by the final target delay.

---

# 8. Text encoder

The text encoder is a 100-neuron semantic spiking encoder with phase-of-firing.

## 8.1 Sparsity

\[
\text{SPARSITY}=0.05.
\]

Therefore the number of active neurons per word is:

\[
k=\max(1,\lfloor100\times0.05\rfloor)=5.
\]

## 8.2 Semantic representation

The semantic space is:

\[
d_{\mathrm{semantic}}=48.
\]

When a pretrained embedding model is enabled, the implementation uses:

**`all-MiniLM-L6-v2`**

and projects its embedding into the 48-dimensional internal semantic space.

Without the pretrained model, the fallback representation uses deterministic character \(n\)-grams and token identity hashing.

## 8.3 Phase-of-firing

For word position \(p\), with four phase bins and gamma-cycle duration \(G\):

\[
\phi_p
=
(p\bmod4)\frac{G}{4}.
\]

The implementation uses:

\[
G=25\text{ ms}.
\]

The emitted spike time is:

\[
t=\phi_p+1\text{ ms}.
\]

---

# 9. Entorhinal Cortex (EC): multimodal convergence

The EC converts modality-specific spikes into rate vectors.

## 9.1 Spike-to-rate conversion

For neuron \(i\):

\[
x_i
=
\frac{N_i}{T/1000},
\]

where \(N_i\) is the number of spikes in duration \(T\) ms.

Thus \(x_i\) is in Hz.

## 9.2 Multimodal concatenation

Without text:

\[
x=
[x_{\mathrm{SQL}};x_{\mathrm{Graph}}],
\]

with dimension:

\[
100+80=180.
\]

With text:

\[
x=
[x_{\mathrm{SQL}};x_{\mathrm{Graph}};x_{\mathrm{Text}}],
\]

with dimension:

\[
100+80+100=280.
\]

## 9.3 Energy

The implemented multimodal energy is:

\[
E
=
\|x\|_2^2
=
\sum_i x_i^2.
\]

## 9.4 Support novelty

Define binary support:

\[
s_i=\mathbf{1}[x_i>0].
\]

A hash signature is computed from this support vector.

Then:

\[
N_{\mathrm{support}}
=
\begin{cases}
0,&\text{if the support signature was already seen},\\
1,&\text{otherwise}.
\end{cases}
\]

## 9.5 Energy z-score

Running statistics are maintained over a finite history window:

\[
z_E
=
\frac{E-\mu_E}{\sigma_E}.
\]

If no external prediction error is provided:

\[
e_{\mathrm{pred}}=|z_E|.
\]

## 9.6 Novelty score

The exact weighted combination implemented is:

\[
S_{\mathrm{novel}}
=
0.45N_{\mathrm{support}}
+
0.35\,\sigma(z_E)
+
0.10\,\operatorname{clip}(e_{\mathrm{pred}},0,1)
+
0.10\,\operatorname{clip}(s,0,1),
\]

where

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

is the logistic function and \(s\) is the salience term.

The neuromodulatory state is:

\[
M=
\begin{cases}
2,&S_{\mathrm{novel}}\ge0.5,\\
1,&S_{\mathrm{novel}}<0.5.
\end{cases}
\]

---

# 10. Dentate Gyrus (DG): pattern separation

## 10.1 Projection

The DG uses a random dictionary / projection matrix:

\[
W\in\mathbb{R}^{K\times D}.
\]

Given EC vector \(x\):

\[
h=Wx.
\]

Rows of \(W\) are normalized.

## 10.2 k-Winner-Take-All

The active DG assembly is:

\[
A_{\mathrm{DG}}
=
\operatorname{TopK}(h,k),
\]

where:

\[
k=
\operatorname{round}
(K\cdot s_{\mathrm{target}})
\]

after clipping the sparsity to the supported range.

The main system uses:

\[
K=1200,\qquad s_{\mathrm{target}}=0.02,
\]

so nominally:

\[
k\approx24.
\]

## 10.3 Online dictionary refinement

The input is normalized:

\[
\hat{x}
=
\frac{x}{\|x\|_2}.
\]

The matrix is decayed:

\[
W\leftarrow\gamma W,
\qquad
\gamma=0.995.
\]

For active DG neurons \(j\):

\[
W_j
\leftarrow
(1-\eta)W_j+\eta\hat{x},
\]

with:

\[
\eta=0.02.
\]

Afterward, rows are renormalized.

## 10.4 DG spike conversion

Each active DG neuron emits a synchronous burst with a small deterministic jitter:

\[
t_j
=
t_{\mathrm{offset}}+0.5+\delta_j.
\]

The jitter is a deterministic function of the neuron index.

---

# 11. CA3 recurrent attractor

## 11.1 Population

Default:

\[
N_E=240,\qquad N_I=60,\qquad N=300.
\]

Thus:

\[
N=N_E+N_I=300.
\]

The implementation characterizes the population as:

- 240 RS excitatory neurons
- 60 FS inhibitory neurons

## 11.2 Recurrent connectivity probabilities

| Projection | Probability | Initial weight range | Delay range |
|---|---:|---:|---:|
| E→E | 15% | 0.2–0.4 | 1–3 ms |
| E→I | 20% | 0.3–0.5 | 0.5–2 ms |
| I→E | 30% | -1.2 to -0.8 | 0.5–1.5 ms |

This implements a sign-constrained excitatory/inhibitory recurrent network.

## 11.3 CA3 current integration

For neuron \(i\):

\[
I_i(t)
=
\sum_{s\in\mathcal{S}_i} I_s(t)
+
b_i,
\]

where \(b_i\) is the learned assembly bias.

The LIF dynamics then use this total input.

---

# 12. Homeostatic threshold adaptation

The firing rate over a duration \(T\) is:

\[
r_i
=
\frac{N_i^{\mathrm{spike}}}{T/1000}.
\]

The target firing rate is:

\[
r^*=6\text{ Hz}.
\]

The homeostatic update is:

\[
\Delta b_i
=
\eta_h(r_i-r^*).
\]

The offset is then clipped:

\[
b_i
\leftarrow
\operatorname{clip}
\left(
b_i+\Delta b_i,
-6,
12
\right).
\]

The default homeostatic learning rate is:

\[
\eta_h=0.02.
\]

---

# 13. Assembly reinforcement / consolidation

For an active CA3 excitatory assembly \(A\), the code raises intrinsic excitability:

\[
b_i
\leftarrow
\operatorname{clip}
\left(
b_i+
\eta_A\cdot0.5\cdot M,
-2,4
\right),
\qquad i\in A.
\]

where:

\[
\eta_A=0.03.
\]

For co-active recurrent E→E synapses:

\[
w_{ij}
\leftarrow
\operatorname{clip}
\left(
w_{ij}+\eta_A M,
w_{\min},w_{\max}
\right).
\]

For active cue-to-assembly synapses:

\[
w_{ij}
\leftarrow
\operatorname{clip}
\left(
w_{ij}+0.5\eta_A M,
w_{\min},w_{\max}
\right).
\]

---

# 14. DG→CA3 bridge

The recall pathway explicitly converts DG spikes into namespaced inputs:

\[
\texttt{dg:}j.
\]

The bridge is built using sparse anchor connections plus random fan-out.

The bridge also receives Hebbian reinforcement:

\[
w_{ij}
\leftarrow
\operatorname{clip}
\left(
w_{ij}+\eta_{\mathrm{DG}}M,
w_{\min},w_{\max}
\right),
\]

for active DG neurons and active CA3 excitatory targets.

Default bridge learning rate:

\[
\eta_{\mathrm{DG}}=0.02.
\]

---

# 15. Two-stage episodic encoding protocol

The implementation explicitly describes a **continuous two-stage encoding protocol**.

## Stage 1: pre-activation

Duration:

\[
T_1=10\text{ ms}.
\]

Plasticity:

\[
M=0.
\]

Purpose:

1. drive CA3 using the current multimodal cue
2. estimate pre-activation
3. measure familiarity
4. decide whether the later plasticity should be normal or boosted

## Familiarity matching

For retrieved/current CA3 active set \(A\) and stored engram \(E_j\), the overlap is measured by Jaccard similarity:

\[
J(A,E_j)
=
\frac{|A\cap E_j|}
{|A\cup E_j|}.
\]

The best matching stored engram is:

\[
j^*
=
\arg\max_j J(A,E_j).
\]

The implementation treats the episode as familiar if either:

\[
J(A,E_{j^*})>0.15
\]

or

\[
|A\cap E_{j^*}|\ge4.
\]

## Stage 2: plasticity

Duration:

\[
T_2=50\text{ ms}.
\]

Neuromodulation:

\[
M=
\begin{cases}
1,&\text{familiar}\\
2,&\text{novel}.
\end{cases}
\]

If STDP is disabled:

\[
M=0.
\]

The same cue stream is used during stage 2.

## Optional replay

An additional replay stage uses:

\[
T_{\mathrm{replay}}=5\text{ ms}.
\]

---

# 16. Engram creation / update

If an episode is novel:

\[
E_{\mathrm{new}}=A_{\mathrm{CA3}}.
\]

If familiar, the stored engram is merged:

\[
E_{j^*}
\leftarrow
E_{j^*}\cup A_{\mathrm{CA3}}.
\]

This creates a persistent CA3 assembly representing the episode.

---

# 17. CA1 readout

CA1 reconstructs encoder-space activity from CA3.

## 17.1 Rate vector

\[
r_{\mathrm{CA3},i}
=
\frac{N_i}{T/1000}.
\]

## 17.2 Linear projection

\[
r_{\mathrm{CA1}}
=
Wr_{\mathrm{CA3}}+b.
\]

The output is rectified:

\[
\hat{y}
=
\max(r_{\mathrm{CA1}},0).
\]

Thus CA1 behaves as a nonnegative linear readout.

## 17.3 Output partition

Without text, the implementation uses:

- SQL output: first 100 dimensions
- graph output: next 80 dimensions

With text, the final 100 dimensions are text-related.

Typical default CA1 dimension:

\[
220\quad(\text{without text}),
\]

\[
320\quad(\text{with text}).
\]

---

# 18. CA1 online associative learning

The target vector concatenates modality support vectors:

\[
y=
[y_{\mathrm{SQL}};y_{\mathrm{Graph}};y_{\mathrm{Text}}].
\]

Prediction:

\[
\hat{y}=Wr+b.
\]

Error:

\[
e=y-\hat{y}.
\]

The implementation uses normalized CA3 activity:

\[
\tilde{r}
=
\frac{r}
{\max(1,\|r\|_2)}.
\]

Weight update:

\[
W
\leftarrow
W+\eta\,e\tilde{r}^{\top}.
\]

Bias update:

\[
b
\leftarrow
b+\eta e.
\]

Weights and biases are clipped after the update.

---

# 19. Relation classification in CA1

The relation head uses a temporal feature vector.

The temporal feature vector contains:

1. normalized CA3 rate vector
2. neuron onset vector
3. early spike-time histogram
4. seven timing summary statistics
5. optional reconstructed graph features

## 19.1 Cosine similarity

For two feature vectors \(a,b\):

\[
\operatorname{cos}(a,b)
=
\frac{a^\top b}
{\|a\|_2\|b\|_2+\epsilon}.
\]

## 19.2 Prototype update

For relation class \(r\), prototype \(p_r\) is blended toward the current feature vector \(f\):

\[
p_r
\leftarrow
(1-\alpha)p_r+\alpha f.
\]

The implementation uses:

\[
\alpha=
\begin{cases}
1.0,&\text{first observation},\\
0.85,&\text{later observations}.
\end{cases}
\]

## 19.3 Linear relation head

Scores are:

\[
z=W_r f+b_r.
\]

With one-hot target \(y\):

\[
e=y-z.
\]

Update:

\[
W_r
\leftarrow
W_r+\eta_r e
\left(
\frac{f}{\max(1,\|f\|_2)}
\right)^\top,
\]

\[
b_r\leftarrow b_r+\eta_r e.
\]

## 19.4 Confidence

The implementation converts relation scores into a normalized exponential confidence:

\[
p_k
=
\frac{\exp(z_k)}
{\sum_j \exp(z_j)}.
\]

The returned confidence is:

\[
\max_k p_k.
\]

---

# 20. Retrieval methodology

Retrieval is explicitly described as **partial-cue retrieval with no learning**:

\[
M=0.
\]

The flow is:

\[
\text{partial cue}
\rightarrow
\text{modal encoders}
\rightarrow
\text{EC}
\rightarrow
\text{DG}
\rightarrow
\text{CA3}
\rightarrow
\text{CA1 reconstruction}.
\]

The retrieval API supports:

- SQL cue only
- graph cue
- text cue
- arbitrary partial combinations

The implementation runs CA3 in a short pre-activation stage followed by a completion stage.

---

# 21. Pattern separation metric

The implementation evaluates a target-vs-impostor comparison using Jaccard overlap.

## Jaccard similarity

\[
J(A,B)
=
\frac{|A\cap B|}
{|A\cup B|}.
\]

For each target episode:

\[
J_{\mathrm{target}}
=
J(A_{\mathrm{retrieved}},E_{\mathrm{target}}).
\]

For all non-target episodes:

\[
J_{\mathrm{impostor},j}
=
J(A_{\mathrm{retrieved}},E_j).
\]

The strongest impostor is:

\[
J_{\mathrm{imp}}^{\max}
=
\max_j J_{\mathrm{impostor},j}.
\]

The separation margin is:

\[
\Delta_{\mathrm{sep}}
=
J_{\mathrm{target}}
-
J_{\mathrm{imp}}^{\max}.
\]

Top-1 separation is counted when:

\[
J_{\mathrm{target}}
\ge
J_{\mathrm{imp}}^{\max}.
\]

The reported false-retrieval rate for this metric is:

\[
\mathrm{FRR}
=
1-\mathrm{Top1Accuracy}.
\]

---

# 22. Temporal continuity metric

Adjacent episodes are compared separately from non-adjacent episodes.

Adjacent mean overlap:

\[
\bar{J}_{\mathrm{adj}}
=
\operatorname{mean}
\left(
J(E_t,E_{t-1}),
J(E_t,E_{t+1})
\right)
\]

over available adjacent pairs.

Non-adjacent mean overlap:

\[
\bar{J}_{\mathrm{nonadj}}
=
\operatorname{mean}
J(E_i,E_j)
\]

over pairs with:

\[
|i-j|>1.
\]

Continuity margin:

\[
\Delta_{\mathrm{cont}}
=
\bar{J}_{\mathrm{adj}}
-
\bar{J}_{\mathrm{nonadj}}.
\]

Link consistency is:

\[
\mathrm{LinkConsistency}
=
\frac{\#\text{consistent predecessor/successor links}}
{\#\text{stored episodes}}.
\]

---

# 23. Completion curve

A SQL cue is progressively reduced.

Default cue fractions:

\[
f\in\{0.2,0.4,0.6,0.8\}.
\]

For each fraction \(f\), the implementation retrieves with a partial row and calculates:

\[
C(f)
=
\operatorname{mean}_{n}
J(A_{\mathrm{retrieved},n}(f),E_n).
\]

Mean completion is:

\[
\bar{C}
=
\frac{1}{|F|}
\sum_{f\in F}C(f).
\]

---

# 24. Interference / retention

For every stored episode, the memory retrieves the episode after later learning.

Per-episode retention:

\[
R_i
=
J(A_{\mathrm{retrieved},i},E_i).
\]

Mean retention:

\[
\bar{R}
=
\frac{1}{N}
\sum_{i=1}^N R_i.
\]

The implementation also reports:

- oldest retention
- newest retention
- full retention vector

---

# 25. Offline consolidation

The system can replay every stored DG assembly.

For each stored episode:

\[
\{j:j\in A_{\mathrm{DG}}\}
\rightarrow
\text{DG→CA3 bridge}
\rightarrow
\text{CA3 replay}.
\]

Then the replayed CA3 assembly is reinforced and the DG→CA3 bridge is strengthened.

The notebook experiment used:

\[
\text{replay passes}=2,
\]

\[
T_{\mathrm{replay}}=5\text{ ms}.
\]

For the 12 stored episodes in the continual-memory experiment:

\[
24\text{ replay events}
\]

were performed.

---

# 26. Modality dropout evaluation

The implementation tests all available modality subsets.

For a modality subset \(S\), the score is the mean completion / overlap metric for that subset.

The experiment reports:

\[
S_{\max}
=
\max_S \operatorname{score}(S),
\]

\[
S_{\min}
=
\min_S \operatorname{score}(S),
\]

and

\[
\bar{S}
=
\operatorname{mean}_S \operatorname{score}(S).
\]

---

# 27. Exactness and provenance

The system separately supports exact stored lookup.

A stored episode contains:

\[
\{
\text{episode\_id},
\text{engram\_id},
\text{timestamp},
\text{SQL row},
\text{graph edge},
\text{text},
E,
M,
\text{familiarity},
A_{\mathrm{CA3}},
A_{\mathrm{DG}},
\text{spike traces},
\text{predecessor},
\text{successor}
\}.
\]

This enables exact value recovery and provenance inspection in addition to neural reconstruction.

---

# 28. Episode provenance table schema

| Field | Meaning |
|---|---|
| `episode_id` | Sequential episode identifier |
| `engram_id` | Associated persistent CA3 engram |
| `timestamp` | Episode time |
| `sql_row` | Structured source record |
| `graph_edge` | Source graph relation |
| `text` | Optional textual context |
| `energy` | EC multimodal energy |
| `neuromodulator` | \(M\) |
| `familiar` | Novel/familiar decision |
| `ca3_assembly` | Stored CA3 active set |
| `dg_assembly` | Stored DG active set |
| `ca3_spikes` | CA3 spike trains |
| `sql_spikes` | SQL encoder spikes |
| `graph_spikes` | Graph encoder spikes |
| `text_spikes` | Text encoder spikes |
| `predecessor_episode_id` | Previous temporal episode |
| `successor_episode_id` | Next temporal episode |

---

# 29. Notebook demonstration results — v3-style multimodal retrieval

## Experiment setup

The notebook creates:

\[
\texttt{MultiModalMemory(use\_text=True, seed=42)}
\]

with pretrained text encoding enabled.

The episode:

- age = 30
- salary = 85000
- city = NYC
- role = Engineer
- dept = AI
- graph edge = `(0, WORKS_AT, 1)`
- text = “Alice works at Google”

is encoded five times.

A partial SQL cue contains only:

- age = 30
- salary = 85000

## Result table

| Metric | Value |
|---|---:|
| Stored CA3 assembly size | 5 |
| Retrieved CA3 size from partial cue | 7 |
| Control CA3 size | 6 |
| Source node accuracy | 0.20 |
| Target node accuracy | 0.40 |
| Source node classification | 0.00 |
| Target node classification | 0.00 |
| Relation accuracy | 1.00 |
| Edge accuracy | 0.00 |
| Temporal-delay accuracy | 1.00 |
| Structure accuracy | 0.5333 |
| CA3 active percentage of 240 excitatory neurons | 2.9167% |
| Graph reconstruction top overlap | 0.00 |
| Relation prediction | `WORKS_AT` |
| Relation confidence | 0.7109 |

## Completion curve

| SQL cue fraction | Mean Jaccard |
|---:|---:|
| 0.20 | 0.6000 |
| 0.40 | 0.6857 |
| 0.60 | 0.8000 |
| 0.80 | 0.6857 |

Mean completion:

\[
\bar{C}=0.692857.
\]

## Exact lookup

The notebook recovered:

| Field | Value |
|---|---|
| age | 30 |
| city | NYC |
| dept | AI |
| role | Engineer |
| salary | 85000 |

## Episode provenance

| Field | Value |
|---|---|
| CA3 assembly size | 5 |
| DG assembly size | 24 |
| Energy | 930000 |
| Engram ID | 0 |
| Episode ID | 4 |
| Familiar | True |
| Graph edge | `(0, WORKS_AT, 1)` |
| Neuromodulator | 1 |
| Predecessor episode | 3 |
| Successor episode | None |
| Timestamp | 4.0 |

Text-only relation retrieval also predicted:

\[
\texttt{WORKS\_AT}
\]

with confidence approximately:

\[
0.7111.
\]

---

# 30. v4 — episodic sequence and temporal provenance

The notebook stores four sequential episodes:

| Episode | Time | Previous | Next | Text |
|---:|---:|---:|---:|---|
| 5 | 0.0 | 4 | 6 | Alice joins Google |
| 6 | 1.0 | 5 | 7 | Alice becomes manager |
| 7 | 2.0 | 6 | 8 | Alice moves to Microsoft |
| 8 | 3.0 | 7 | — | Alice becomes CTO |

The notebook explicitly demonstrates predecessor retrieval:

**Current:** Alice becomes CTO

**Previous:** Alice moves to Microsoft

---

# 31. v5 — continual memory experiment

A second memory instance uses:

\[
\texttt{use\_text=False},\quad seed=7.
\]

There are six semantic episodes, each encoded twice, giving:

\[
N=12
\]

stored episode records.

## Retention before consolidation

\[
\bar{R}=0.7355769.
\]

Oldest retention:

\[
0.3333333.
\]

Newest retention:

\[
1.0.
\]

## Retention after consolidation

The reported result is unchanged:

\[
\bar{R}=0.7355769.
\]

This is an important empirical observation from the current implementation: in this particular run, the replay pass did not change the reported retention metric.

## Consolidation

| Parameter | Value |
|---|---:|
| Replay duration | 5 ms |
| Replay passes | 2 |
| Replayed episodes/events | 24 |

## Completion after consolidation

| Cue fraction | Mean Jaccard |
|---:|---:|
| 0.20 | 0.73485 |
| 0.40 | 0.73752 |
| 0.60 | 0.74733 |
| 0.80 | 0.74733 |

Mean:

\[
\bar{C}=0.7417565.
\]

---

# 32. v5 separation results

| Metric | Value |
|---|---:|
| Target overlap mean | 0.73558 |
| Best impostor overlap mean | 0.97436 |
| Separation margin mean | -0.23878 |
| Top-1 separation accuracy | 0.16667 |
| False-retrieval rate | 0.83333 |
| Mean pairwise assembly overlap | 0.72465 |
| Max pairwise assembly overlap | 1.00000 |
| Number of records | 12 |

Interpretation from the raw implementation metrics: the current configuration shows **high interference / poor pattern separation** in this experiment because impostor overlap is greater than target overlap on average.

---

# 33. v5 continuity results

| Metric | Value |
|---|---:|
| Adjacent overlap mean | 0.91270 |
| Non-adjacent overlap mean | 0.68704 |
| Continuity margin | 0.22566 |
| Link consistency | 0.91667 |

Thus the observed continuity margin is positive:

\[
\Delta_{\mathrm{cont}}
=
0.9126951-0.6870386
=
0.2256566.
\]

---

# 34. Modality dropout result

For the same `mem2` continual-memory experiment:

| Modality subset | Score |
|---|---:|
| Graph | 0.61261 |
| SQL | 0.72550 |
| SQL + Graph | 0.73558 |

Summary:

| Metric | Value |
|---|---:|
| Best subset | SQL + Graph |
| Best score | 0.73558 |
| Worst subset | Graph |
| Worst score | 0.61261 |
| Mean score | 0.69123 |

---

# 35. False-retrieval result

The notebook reports:

| Metric | Value |
|---|---:|
| False retrieval count | 12 |
| False retrieval rate | 1.00000 |
| Mean best-impostor overlap | 0.97436 |
| Mean margin | -0.23878 |
| Mean target overlap | 0.73558 |
| Number of records | 12 |

This is a stricter false-retrieval criterion than the simpler separation top-1 metric, and the two reported values should not be conflated.

---

# 36. Exact lookup and provenance for first continual-memory episode

## Exact lookup

| Field | Value |
|---|---|
| age | 28 |
| city | SF |
| dept | Core |
| role | Engineer |
| salary | 70000 |

## Provenance

| Field | Value |
|---|---|
| CA3 assembly size | 4 |
| DG assembly size | 24 |
| Energy | 670000 |
| Engram ID | 0 |
| Episode ID | 0 |
| Familiar | False |
| Graph edge | `(0, WORKS_AT, 1)` |
| Neuromodulator | 2 |
| Predecessor | None |
| Successor | 1 |
| Timestamp | 0.0 |

---

# 37. v6 micro-benchmark table

The notebook evaluates four configurations.

| Config | Text | CA3 exc. | Consolidation | Mean completion | Mean retention | Separation margin | Separation top-1 | False retrieval | Dropout mean | Best dropout subset | Continuity margin | Link consistency | Final CA3 assembly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| A | Yes | 240 | Yes | 0.52173 | 0.93333 | 0.04444 | 0.66667 | 0.33333 | 0.59359 | SQL+Graph+Text | 0.15000 | 0.66667 | 6 |
| B | No | 240 | Yes | 0.58590 | 0.74643 | 0.08717 | 0.33333 | 0.66667 | 0.62050 | SQL | 0.09206 | 0.66667 | 9 |
| C | Yes | 480 | Yes | 0.36254 | 0.77778 | -0.03175 | 0.33333 | 0.66667 | 0.48997 | SQL+Graph+Text | 0.22857 | 0.66667 | 7 |
| D | Yes | 240 | No | 0.52173 | 0.93333 | 0.04444 | 0.66667 | 0.33333 | 0.59359 | SQL+Graph+Text | 0.15000 | 0.66667 | 6 |

Additional recorded values:

| Config | DG→CA3 bridge weight delta | Homeostatic offset mean |
|---|---:|---:|
| A | 0.01414 | -0.32833 |
| B | 0.02147 | -0.32333 |
| C | 0.00991 | -0.34583 |
| D | 0.01414 | -0.32833 |

---

# 38. Main algorithmic pipeline in compact mathematical form

For an episode containing structured input \(q\), graph input \(g\), and optional text \(z\):

### Modality encoding

\[
s_q = \mathcal{E}_{\mathrm{SQL}}(q),
\qquad
s_g = \mathcal{E}_{\mathrm{graph}}(g),
\qquad
s_z = \mathcal{E}_{\mathrm{text}}(z).
\]

### Rate conversion / EC convergence

\[
x=
\left[
\mathcal{R}(s_q);
\mathcal{R}(s_g);
\mathcal{R}(s_z)
\right].
\]

### Novelty

\[
E=\|x\|_2^2,
\]

\[
S_{\mathrm{novel}}
=
0.45N_{\mathrm{support}}
+
0.35\sigma(z_E)
+
0.10\operatorname{clip}(e_{\mathrm{pred}},0,1)
+
0.10\operatorname{clip}(s,0,1).
\]

### DG separation

\[
h=Wx,
\]

\[
A_{\mathrm{DG}}=\operatorname{TopK}(h,k).
\]

### DG→CA3 drive

\[
u_{\mathrm{CA3}}
=
\mathcal{B}(A_{\mathrm{DG}}).
\]

### CA3 recurrent dynamics

\[
\tau_m\frac{dv_i}{dt}
=
-(v_i-v_{\mathrm{rest}})+I_i(t)+b_i.
\]

### STDP

\[
\Delta w_{ij}
=
\begin{cases}
M A_+e^{-\Delta t/\tau_+}, & \Delta t\ge 0,\\
-MA_-e^{\Delta t/\tau_-}, & \Delta t<0.
\end{cases}
\]

### Engram

\[
E_n=A_{\mathrm{CA3}}.
\]

### CA1 readout

\[
\hat{y}
=
\operatorname{ReLU}
(W_{\mathrm{CA1}}r_{\mathrm{CA3}}+b).
\]

### Reconstruction / completion

\[
\mathrm{score}
=
J(A_{\mathrm{retrieved}},E_{\mathrm{target}}).
\]

---

# 39. Research-methodology summary

The implemented methodology can be organized into the following research stages:

1. **Multimodal sparse encoding**
   - structured population coding
   - graph ensemble coding
   - semantic text coding
   - temporal spike phases

2. **Convergent EC representation**
   - spike-rate conversion
   - multimodal concatenation
   - energy
   - support novelty
   - neuromodulation

3. **Pattern separation**
   - random projection
   - k-WTA
   - sparse DG assemblies
   - online dictionary refinement

4. **Pattern completion / attractor memory**
   - recurrent CA3
   - excitatory/inhibitory dynamics
   - delayed synapses
   - STDP
   - homeostasis
   - assembly reinforcement

5. **Readout**
   - CA1 linear reconstruction
   - relation classifier
   - temporal feature extraction
   - relation prototypes

6. **Continual memory**
   - episode storage
   - predecessor/successor links
   - exact provenance
   - replay
   - consolidation
   - interference testing

7. **Evaluation**
   - completion curves
   - retention
   - separation margin
   - false retrieval
   - modality dropout
   - continuity
   - graph reconstruction
   - relation classification
   - exact lookup / provenance

---

# 40. Paper-ready table: default hyperparameters

| Category | Parameter | Value |
|---|---|---:|
| Simulation | \(\Delta t\) | 0.1 ms |
| Simulation | Encoding duration | 10 ms |
| Encoding | SQL neurons | 100 |
| Encoding | Graph neurons | 80 |
| Encoding | Text neurons | 100 |
| Text | Sparsity | 5% |
| Text | Gamma cycle | 25 ms |
| Text | Semantic dimension | 48 |
| EC | History window | 10 |
| EC | Novelty threshold parameter | 2.0 (stored) |
| DG | Output dimension | 1200 |
| DG | Target sparsity | 2% |
| DG | Online learning rate | 0.02 |
| DG | Weight decay | 0.995 |
| CA3 | Excitatory neurons | 240 |
| CA3 | Inhibitory neurons | 60 |
| CA3 | Homeostatic target rate | 6 Hz |
| CA3 | Homeostatic learning rate | 0.02 |
| CA3 | Assembly learning rate | 0.03 |
| STDP | \(A_+\) | 0.5 |
| STDP | \(A_-\) | 0.48 |
| STDP | \(\tau_+\) | 12 ms |
| STDP | \(\tau_-\) | 12 ms |
| STDP | Neuromodulation | 0 / 1 / 2 |
| CA1 | Train epochs | 12 |
| CA1 | Readout LR | 0.03 |
| CA1 | Relation LR | 0.05 |
| CA1 | Relation bins | 12 |
| Consolidation | Replay duration | 5 ms |
| Consolidation | Default replay passes | 1 |
| Retrieval | Standard duration | 50 ms |

---

# 41. Important source-derived implementation notes

## 41.1 Literal LaTeX status

The uploaded source is primarily executable Python plus notebook prose. It does **not** contain a full pre-written LaTeX methodology section.

Therefore, this document converts the mathematical operations that are actually implemented into publication-style LaTeX.

## 41.2 EC dimension naming

The `EntorhinalCortex` class contains a docstring describing it as “180 neurons,” while its instantiated dimension is:

\[
180\quad\text{without text},
\]

and

\[
280\quad\text{with text}.
\]

The actual system initialization uses:

\[
n_{\mathrm{SQL}}=100,\quad
n_{\mathrm{graph}}=80,\quad
n_{\mathrm{text}}=100.
\]

## 41.3 DG dimension

The DG class default constructor shown in isolation uses 900 neurons, but the complete `MultiModalMemory` system configures:

\[
\text{DG output dimension}=1200,
\]

with 2% target sparsity, yielding approximately 24 active DG neurons.

---

# 42. Suggested paper section mapping

A research manuscript based directly on this implementation can map the extracted material as:

### 3. Method
3.1 Problem formulation  
3.2 Multimodal spike encoders  
3.3 Entorhinal multimodal convergence  
3.4 Novelty-dependent neuromodulation  
3.5 DG sparse pattern separation  
3.6 CA3 recurrent attractor memory  
3.7 STDP and homeostasis  
3.8 CA1 reconstruction and relation decoding  
3.9 Episodic storage and provenance  
3.10 Continual consolidation and replay

### 4. Experimental Protocol
4.1 Partial-cue retrieval  
4.2 Modality dropout  
4.3 Completion curves  
4.4 Pattern separation  
4.5 Temporal continuity  
4.6 Interference / retention  
4.7 False retrieval  
4.8 Micro-benchmarks

### 5. Results
Use the benchmark and continual-memory tables extracted above.

---

# 43. Source-to-math traceability

The equations in this extraction correspond to implemented operations including:

- LIF Euler dynamics
- spike threshold/reset/refractory logic
- exponentially decaying synaptic trace
- delayed-spike STDP
- Gaussian population coding
- salary log-tail coding
- graph temporal delay coding
- phase-of-firing text coding
- EC rate conversion and energy
- novelty scoring
- DG projection and k-WTA
- DG online dictionary update
- CA3 homeostasis
- assembly reinforcement
- CA1 linear readout
- CA1 associative update
- cosine relation features
- relation classifier update
- Jaccard overlap
- separation margin
- continuity margin
- completion curve
- retention / interference
- replay consolidation
