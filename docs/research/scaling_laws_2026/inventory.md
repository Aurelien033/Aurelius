# Scaling-Law / Efficiency Paper Inventory

Output directory: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026`

Lilian Weng page: fetched 35125 chars

| ID | Title | arXiv | KB | Relevance hits |
|---|---|---:|---:|---|
| 1 | Tapered Language Models | 2606.23670v1 | 66.4 | taper:83, parameter:54, perplexity:33, tokens:14, flops:11 |
| 2 | O UTRAGEOUSLY L ARGE N EURAL N ETWORKS : | 1701.06538v1 | 77.3 | moe:148, expert:142, parameter:40, perplexity:32, mixture:18 |
| 3 | Do Neural Networks Lose Plasticity in a Gradually Changing World? | 2602.09234v2 | 92.3 | plasticity:101, benchmark:24, parameter:16, reinforcement:7, compute:4 |
| 4 | Neural Garbage Collection: | 2604.18002v1 | 86.8 | tokens:57, kv cache:56, parameter:21, compute:15, reinforcement:14 |
| 5 | Why Larger Models Learn More: Effects of Capacity, | 2605.29548v1 | 166.6 | parameter:31, compute:28, mixture:21, scaling law:18, tokens:17 |
| 6 | Fara-1.5: Scalable Learning Environments for | 2606.20785v1 | 162.4 | agent:178, compute:40, benchmark:34, verifier:33, parameter:7 |
| 7 | Fara-1.5: Scalable Learning Environments for | 2606.20785v1 | 162.4 | agent:178, compute:40, benchmark:34, verifier:33, parameter:7 |
| 8 | The Pitfall of Scaling Up: Uncovering and Mitigating Popularity | 2606.21911v1 | 233.3 | parameter:32, aggregation:23, scaling law:11, compute:8, tokens:2 |
| 9 | Priority-Aware Learning-Unlearning Correction | 2606.22878v1 | 160.6 | aggregation:36, parameter:32, compute:11, mixture:5, moe:3 |
| 10 | FORGE: Fused On-Register Gradient Elimination | 2606.22932v1 | 200.4 | parameter:47, parallel:38, compute:13, tokens:11, benchmark:8 |
| 11 | Provable Benefits of RLVR over SFT for Reasoning Models: | 2606.22938v1 | 198.6 | reinforcement:16, compute:15, verifier:7, agent:6, expert:5 |
| 12 | PeLAP-A: Adaptive Latent Pruning | 2606.23086v1 | 30.0 | parameter:8, compute:4 |
| 13 | SPIRAL: LEARNING TO SEARCH AND AGGREGATE | 2606.23595v1 | 110.9 | compute:103, aggregation:80, parallel:68, reinforcement:57, inference compute:38 |
| 14 | Tapered Language Models | 2606.23670v1 | 66.4 | taper:83, parameter:54, perplexity:33, tokens:14, flops:11 |
| 15 | Are We Ready For An Agent-Native Memory System? | 2606.24775v1 | 276.9 | agent:110, benchmark:14, tokens:9, kv cache:4, aggregation:3 |
| 16 | ConSolv: Solvent-Conditional Machine Learning Implicit Solvent | 2606.24983v1 | 77.1 | parameter:23, compute:11, benchmark:10, parallel:2, aggregation:1 |
| 17 | Minimax PAC Bounds for Learning in Exogenous | 2606.25170v1 | 200.7 | reinforcement:16, parameter:14, compute:11, agent:9, benchmark:2 |
| 18 | S2-CAR: Segmentation-Supervised Complexity-Adaptive | 2606.25415v1 | 117.3 | parameter:26, benchmark:9, compute:3, aggregation:3, expert:2 |
| 19 | A functional central limit theorem for kernel | 2606.25494v1 | 166.7 | parameter:8, compute:2 |
| 20 | BitNet Text Embeddings | 2606.25674v1 | 87.2 | benchmark:7, compute:4, parameter:3, tokens:3 |
| 21 | Improving Neural Network Training by | 2606.25971v1 | 184.5 | parameter:69, tokens:51, compute:35, moe:33, expert:25 |

## Abstract snippets

### 1. Tapered Language Models (2606.23670v1)
PDF: `/Users/christienantonio/Downloads/Tapered Language Models-with-annotations.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/Tapered_Language_Models-with-annotations.txt`
Modern language models, including transformer, recurrent, and memory-based variants, share a common chassis: a stack of identical layers in which parameters are allocated uniformly across depth. This is a default inherited from the original transformer and largely unchanged since, yet a growing body of evidence suggests that layers contribute non-uniformly to the final output, with later layers refining the residual stream rather than transforming it. We ask whether parameter capacity should reflect this asymmetry. Our controlled experiment shows that, under a fixed budget, allocating more capacity to earlier layers and less to later layers improves perplexity over a uniform-width baseline, 

### 2. O UTRAGEOUSLY L ARGE N EURAL N ETWORKS : (1701.06538v1)
PDF: `/Users/christienantonio/Downloads/1701.06538v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/1701.06538v1.txt`
_No abstract captured by regex._

### 3. Do Neural Networks Lose Plasticity in a Gradually Changing World? (2602.09234v2)
PDF: `/Users/christienantonio/Downloads/2602.09234v2.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2602.09234v2.txt`
2023), using alternative activation functions (Berariu et al., 2021; Lee et al., 2023; Abbas et al., 2023), controlling the Continual learning has become a trending topic in loss landscape sharpness (Lyle et al., 2023), resetting less- machine learning. Recent studies have discovered used neurons (Dohare et al., 2024; Sokar et al., 2023), and an interesting phenomenon called loss of plastic- arXiv:2602.09234v2 [cs.LG] 16 Jun 2026 selectively forgetting memorized noise (Shin et al., 2024). ity, referring to neural networks gradually losing the ability to learn new tasks. However, existing A common thread across these studies is the use of bench- plasticity research largely relies on benchmark

### 4. Neural Garbage Collection: (2604.18002v1)
PDF: `/Users/christienantonio/Downloads/2604.18002v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2604.18002v1.txt`
Chain-of-thought reasoning has driven striking advances in language model capability, yet every reasoning step grows the KV cache, creating a bottleneck to scaling this paradigm further. Current approaches manage these constraints on the model’s behalf using hand-designed criteria. A more scalable approach would let end-to-end learning subsume this design choice entirely, following a broader pattern in deep learning. After all, if a model can learn to reason, why can’t it learn to forget? We introduce Neural Garbage Collection (NGC), in which a language model learns to forget while learning to reason, trained end-to-end from outcome-based task reward alone. As the model reasons, it periodica

### 5. Why Larger Models Learn More: Effects of Capacity, (2605.29548v1)
PDF: `/Users/christienantonio/Downloads/2605.29548v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2605.29548v1.txt`
Larger models learn tasks smaller models do not. What drives this phenomenon? We develop a simple phenomenological argument that power-law scaling already suggests that a larger model will be able to learn a part of the data distribution that a smaller model fails to learn, even with infinite training data. To validate this claim and identify its causes, we study the effects of model scaling on a synthetic setup consisting of a mixture of tasks that show monotonic scaling curves. The results point to a data-induced competition over resources (neurons). Specifically, smaller models allocate their neurons to high frequency or low complexity tasks, and so they learn solutions that perform poorl

### 6. Fara-1.5: Scalable Learning Environments for (2606.20785v1)
PDF: `/Users/christienantonio/Downloads/2606.20785v1-2.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.20785v1-2.txt`
_No abstract captured by regex._

### 7. Fara-1.5: Scalable Learning Environments for (2606.20785v1)
PDF: `/Users/christienantonio/Downloads/2606.20785v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.20785v1.txt`
_No abstract captured by regex._

### 8. The Pitfall of Scaling Up: Uncovering and Mitigating Popularity (2606.21911v1)
PDF: `/Users/christienantonio/Downloads/2606.21911v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.21911v1.txt`
Keywords We identify a critical pitfall in scaling transformer-based sequential Recommender Systems; Sequential Recommendation; Scaling Laws; recommenders: while increasing model size improves recommen- Popularity Bias; Spectral Regularization dation accuracy, it simultaneously amplifies popularity bias. This ACM Reference Format: bias drives systems to over-recommend popular items at the ex- Weiqin Yang, Yue Pan, Chongming Gao, Sheng Zhou, Xiang Wang, Can pense of niche ones, which not only undermines fairness but also Wang, and Jiawei Chen. 2026. The Pitfall of Scaling Up: Uncovering and degrades the broader ecosystem by reinforcing the Matthew effect Mitigating Popularity Bias Amplificati

### 9. Priority-Aware Learning-Unlearning Correction (2606.22878v1)
PDF: `/Users/christienantonio/Downloads/2606.22878v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.22878v1.txt`
_No abstract captured by regex._

### 10. FORGE: Fused On-Register Gradient Elimination (2606.22932v1)
PDF: `/Users/christienantonio/Downloads/2606.22932v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.22932v1.txt`
Reverse-mode differentiation computes every weight gradient, writes it to memory, and only then lets the optimizer read it back. This two-phase schedule sets the memory ceiling of modern training: at the seam between the phases, every layer’s gradient is live at once. We argue that this materialized gradient is an artifact of how differentiation is staged, not a quantity that learning requires—and we eliminate it. FORGE folds the optimizer step into the backward pass and applies it one tile at a time, entirely in registers, so each gradient tile is consumed the instant it is produced and never becomes a tensor. The fusion changes only when the update happens, not what it computes: in full pr

### 11. Provable Benefits of RLVR over SFT for Reasoning Models: (2606.22938v1)
PDF: `/Users/christienantonio/Downloads/2606.22938v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.22938v1.txt`
2023). Reasoning models treat such tasks as a sequential multi-step decision process and deploy strategies utilizing Recent advances in large language models additional test-time compute budget, such as sampling, tree arXiv:2606.22938v1 [cs.LG] 22 Jun 2026 (LLMs) have demonstrated that reinforcement search, aggregation, and backtracking. This approach has fine-tuning of pretrained base models can lead proved to yield efficient and scalable gains over initial pre- to significant gains in reasoning performance at training (Snell et al., 2024; Muennighoff et al., 2025), and inference time. In this work, we theoretically has been adopted with great success in various frontier and analyze why rei

### 12. PeLAP-A: Adaptive Latent Pruning (2606.23086v1)
PDF: `/Users/christienantonio/Downloads/2606.23086v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.23086v1.txt`
contain significant redundancy, since not all channels nec- essarily carry equal information for the denoising task. Latent diffusion models achieve strong genera- This raises a natural and underexplored research ques- tive performance by operating in a compressed latent tion: can we identify and selectively suppress redundant space produced by a variational autoencoder (VAE). latent channels without degrading generation quality? Un- However, it remains unclear whether all latent chan- like approaches that compress the diffusion process it- nels contribute equally to the diffusion process, or self [Salimans & Ho, 2022, Song et al., 2023] or prune whether significant redundancy exists. We int

### 13. SPIRAL: LEARNING TO SEARCH AND AGGREGATE (2606.23595v1)
PDF: `/Users/christienantonio/Downloads/2606.23595v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.23595v1.txt`
Language model reasoning can be substantially improved at test time via scaffolds that scale inference compute across different primitives—sequential reasoning within a trace, independently sampled parallel traces, and aggregation of multiple reasoning traces into a final response. During post-training, however, language models are optimized only for sequential reasoning within a single trace. We introduce Sequential-Parallel-Aggregative Reinforcement Learning (Spiral), a framework in which a language model is trained to use all three primitives, as part of a unified inference compute pipeline. Concretely, the language model first samples a set of independent traces in parallel, each produce

### 14. Tapered Language Models (2606.23670v1)
PDF: `/Users/christienantonio/Downloads/2606.23670v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.23670v1.txt`
Modern language models, including transformer, recurrent, and memory-based variants, share a common chassis: a stack of identical layers in which parameters are allocated uniformly across depth. This is a default inherited from the original transformer and largely unchanged since, yet a growing body of evidence suggests that layers contribute non-uniformly to the final output, with later layers refining the residual stream rather than transforming it. We ask whether parameter capacity should reflect this asymmetry. Our controlled experiment shows that, under a fixed budget, allocating more capacity to earlier layers and less to later layers improves perplexity over a uniform-width baseline, 

### 15. Are We Ready For An Agent-Native Memory System? (2606.24775v1)
PDF: `/Users/christienantonio/Downloads/2606.24775v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.24775v1.txt`
Represent. & Storage Extraction Retrieval / Routing Maintenance Memory for large language model (LLM) agents has rapidly evolved retrieved Core from simple retrieval-augmented mechanisms into a data manage- Observation / Experience Memory Stream query Retrieval Scorer memory Reflection User core_memory_append core_memory_replace Memory evict recency × importance (LLM) Message migrate promote (periodic) ment system that supports persistent information storage, retrieval, (Timestamped Log) × relevance LLM Function Call search Recall inform Agent (self-directed) Storage update, consolidation, and dynamic lifecycle governance through- User Response insight write-back plan LLM Agent Planner archi

### 16. ConSolv: Solvent-Conditional Machine Learning Implicit Solvent (2606.24983v1)
PDF: `/Users/christienantonio/Downloads/2606.24983v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.24983v1.txt`
Implicit solvent machine learning potentials (MLPs) offer a powerful route to bridging the gap be- tween accuracy and efficiency in molecular simulations. However, existing models have largely focused on aqueous environments, overlooking the diverse and important roles of non-aqueous solvents in areas such as organic synthesis and battery technology. Here, we present ConSolv, a solvent-conditional MLP architecture that explicitly incorporates solvent effects on solute interactions through an attention-based solvent-embedding block. By combining experimental solvation free energy data with ab initio data, we train a single implicit solvent MLP that is transferable across 66 common organic sol

### 17. Minimax PAC Bounds for Learning in Exogenous (2606.25170v1)
PDF: `/Users/christienantonio/Downloads/2606.25170v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.25170v1.txt`
We study PAC learning in tabular discounted Markov decision processes with exogenous i.i.d. contexts, with discount factor γ, finite state space X , action space A, and context space Z. At each time step, a context is drawn independently from an unknown distribution µ and revealed before the agent acts. This context may affect both rewards and transitions, while remaining uncontrolled by the agent. Depending on the regime, the learner has access either to a sampling oracle for µ, to a sampling oracle for the transition kernel conditioned on state-context-action tuples, or to both. Oracles can be accessed before and during policy execution. The sample complexity is measured by a couple (n, m)

### 18. S2-CAR: Segmentation-Supervised Complexity-Adaptive (2606.25415v1)
PDF: `/Users/christienantonio/Downloads/2606.25415v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.25415v1.txt`
arXiv:2606.25415v1 [cs.IR] 24 Jun 2026 Sequential recommendation aims to predict user preferences from interaction histories, yet existing models often struggle when behavior patterns become complex and hetero- geneous. A key reason is that interaction histories are rarely uniform: users’ interests shift in a latent way over time, yet existing models either treat the full sequence as a homogeneous context or rely on rigid time-window segmentation that misaligns with true intent boundaries. This mis-segmentation not only introduces cross-intent inter- ference at intermediate sequence positions but also leads to over-reliance on short-term interest signals. To address this, we propose S2-CAR, 

### 19. A functional central limit theorem for kernel (2606.25494v1)
PDF: `/Users/christienantonio/Downloads/2606.25494v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.25494v1.txt`
Building on the large-sample analysis of infinitesimal gradient boosting (Dombry and Duchamps, 2024b), we study the fluctuations of the process around its deterministic limit and establish a functional central limit theorem: the rescaled deviations converge in distribution to a Gaussian process. The analysis is carried out in a reproducing kernel Hilbert space (RKHS) naturally associated with the softmax gradient tree base learner, in which the boosting process is characterized as the solution of an autonomous ordinary differential equation (ODE). The proof rests on a general stochastic perturbation analysis of ODEs in Banach spaces, which is of independent interest: whenever a sequence of v

### 20. BitNet Text Embeddings (2606.25674v1)
PDF: `/Users/christienantonio/Downloads/2606.25674v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.25674v1.txt`
LLM-based text embedders have substantially improved retrieval and semantic representation quality, but their deployment remains costly: large backbone models slow down embedding inference, while high-dimensional full-precision embed- dings impose substantial storage and bandwidth overhead on large-scale indexes. In this paper, we present B IT E MBED, an extreme low-bit framework for LLM- based text embedding that jointly targets encoding efficiency and vector storage. B IT E MBED converts pretrained LLM backbones into BitNet-style embedding en- coders with ternary weights, quantized activations, and lightweight normalization refinement. The converted model is adapted to representation learn

### 21. Improving Neural Network Training by (2606.25971v1)
PDF: `/Users/christienantonio/Downloads/2606.25971v1.pdf`
TXT: `/Users/christienantonio/aurelius/docs/research/scaling_laws_2026/2606.25971v1.txt`
Modern neural network training relies on optimizers such as Adam and Muon which act on each weight matrix as a single object. Yet every weight matrix carries two distinct quantities — a magnitude and a direction — and all optimizers stepping in the matrix as a whole couple their dynamics: the directional change from an update depends on the current magnitude, while the magnitude drifts as a byproduct of learning the direction, so neither is governed directly by the learning rate. Typical training therefore leans on surrounding recipes such as weight decay and warmup to keep learning stable at scale, though these regulate the coupling only indirectly; other recent methods instead constrain th