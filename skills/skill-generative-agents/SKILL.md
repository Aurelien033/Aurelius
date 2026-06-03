---
name: skill-generative-agents
description: >
  Research knowledge skill for Park et al. (2023 / UIST '23) "Generative Agents:
  Interactive Simulacra of Human Behavior" (arXiv 2304.03442). Covers the
  architecture (memory stream, retrieval, reflection, planning), the Smallville
  sandbox, evaluation methodology, emergent social behaviors, and synthetic data
  generation for social science. Load when working on LLM-based agent simulation,
  believable NPC behavior, interactive sandbox environments, or human-behavior
  benchmarking.
---

# Generative Agents: Interactive Simulacra of Human Behavior

> **Paper:** Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P.,
> & Bernstein, M.S. (2023). *Generative Agents: Interactive Simulacra of Human
> Behavior.* UIST '23, October 29–November 1, 2023, San Francisco, CA, USA.
> arXiv:2304.03442v2 [cs.HC] (6 Aug 2023)
>
> **Authors & Affiliations:**
> - Joon Sung Park — Stanford University (joonspk@stanford.edu)
> - Joseph C. O'Brien — Stanford University (jobrien3@stanford.edu)
> - Carrie J. Cai — Google Research (cjcai@google.com)
> - Meredith Ringel Morris — Google DeepMind (merrie@google.com)
> - Percy Liang — Stanford University (pliang@cs.stanford.edu)
> - Michael S. Bernstein — Stanford University (msb@cs.stanford.edu)
>
> **Packet stats:** 22 pages / ~130,000 chars (PDF text extraction)

---

## 1. Core Thesis

This paper introduces **generative agents** — computational software agents that
simulate **believable human behavior** over days and weeks in an interactive
sandbox world. The key insight is that a large language model (LLM), when
augmented with three architectural mechanisms — a memory stream with a scored
retrieval function, a periodic self-reflection mechanism, and a recursively
decomposed generative planner — can produce agents that behave coherently and
believably over extended time horizons.

Believable agents "wake up, cook breakfast, and head to work; artists paint,
and authors write; they form opinions, notice each other, and initiate
conversations; they remember and reflect on days past as they plan the next
day." The architecture preserves this continuity by storing every experience as
a natural-language record, synthesizing those records into higher-level
reflections, and dynamically retrieving relevant memory at each decision point.

### Key Claim

Individual agent modeling using an LLM-conditioned architecture can generate a
wide range of **group-level social phenomena** — information diffusion,
relationship formation, and coordination — without explicitly programming any
group-level logic. This enables synthetic data generation for social science:
simulations of Internet posting interactions, community behavior, and other
open-world human scenarios.

---

## 2. The Smallville Sandbox

### Environment

Smallville is a 2D Sims-style game world built with the **Phaser** web game
framework. It contains:

- **Residential areas:** houses with kitchens, bedrooms, desks, showers, etc.
- **Commercial areas:** Hobbs Cafe, The Willows Market & Pharmacy, The Rose and
  Crown (pub)
- **Public spaces:** Johnson Park, Harvey Oak Supply Store, Oak Hill College
  (dorms + school)
- **25 agents** initialized with natural-language persona descriptions

The environment is represented internally as a **containment tree** (world →
area → objects), which is converted to natural-language text for the LLM.
Agents build partial sub-trees as they explore and cannot see locations they
have not visited, keeping their knowledge bounded.

### User Interaction

- **Observer mode:** users watch agents interact autonomously.
- **Direct command:** users adopt the persona of an agent's "inner voice" to
  issue imperatives. Example: "You are running against Sam in the upcoming
  election" causes the agent to change their plan.
- **Reporter persona:** users can ask about current world state; agents answer
  from their own experience (e.g., "Who is running for mayor?").
- **Environment editing:** users can rewrite object states in natural language
  (e.g., "`<Isabella's apartment: kitchen: stove>` is burning"), which the
  agent perceives on the next tick.

### Architecture Stack

```
Sandbox Server ── JSON state ──► Agent Architecture ── LLM action ▼
       ▲                                   │                        │
       │         Agent's perception is logged into memory stream      │
       └─────────────────────────────────────────────────────────────┘
```

The generated action (e.g., "making espresso for a customer @ Hobbs Cafe:
counter: coffee machine") updates the JSON; on the next tick the server
broadcasts the new state to all agents within visual range.

---

## 3. Key Mechanisms

### 3.1 Memory Stream

The **memory stream** is the central data structure: a chronological list of
**memory objects**, each containing:

| Field | Description |
|---|---|
| `description` | Natural-language text of the event/observation |
| `creation_timestamp` | When the event was first observed |
| `last_accessed_timestamp` | Most recent retrieval time |

Two types of memory:
- **Observations** — raw, directly perceived events (agent's own actions, other
  agents' actions, object state changes)
- **Reflections** — higher-level, inferred insights (see §3.2)
- **Plans** — future action sequences (see §3.3)

### 3.2 Retrieval Function

Only a **subset** of memories is passed to the LLM at each decision step
(context-window constraints). The retrieval score is a weighted sum of three
components:

```
score = α_recency · recency + α_importance · importance + α_relevance · relevance
```

All α weights are set to 1 by default.

| Component | Mechanism | Implementation |
|---|---|---|
| **Recency** | Exponential decay over hours since last access | `0.995^(hours)` |
| **Importance** | Scores events 1–10 for poignancy | LLM prompt: "On the scale of 1 to 10..." |
| **Relevance** | Cosine similarity of embedding vectors | LLM embedding of description vs. query |

The top-K scored memories that fit within the LLM's context window are
retrieved and included in the prompt.

### 3.3 Reflection

Reflection is the mechanism by which agents generalize from observations into
**higher-level knowledge**.

**Trigger:** Reflection runs when the sum of importance scores for recently
perceived events exceeds a threshold of 150; in practice, agents reflect ~2–3×
per day.

**Process (three steps):**
1. **Identify questions:** Given the 100 most recent memory records, prompt the
   LLM to generate 3 salient high-level questions (e.g., "What is Klaus Mueller
   passionate about?").
2. **Retrieve evidence:** Use each question as a retrieval query; gather
   relevant memories (including existing reflections).
3. **Generate insight:** Prompt the LLM to extract 5 high-level insights,
   formatted as `insight (because of 1, 5, 3)` pointing to specific record IDs.

The result is stored as a reflection node in the memory stream, creating a
**reflection tree**: leaves are raw observations; internal nodes are
increasingly abstract reflections that cite the children as evidence. Depth can
grow arbitrarily as time advances.

### 3.4 Generative Planner

Without plans, LLM-driven agents oscillate (e.g., eat lunch at 12:00 PM, then
again at 12:30 PM and 1:00 PM). Plans give long-term coherence.

**Plan structure:** `for DURATION from TIME at LOCATION: ACTION`

**Recursive decomposition:**

```
Day plan (5–8 broad chunks)
  └─ Hourly chunk (1-hour blocks)
       └─ Minute chunk (5–15 minute actions)
```

**Day plan** is generated top-down via LLM prompt conditioned on the agent
summary + previous-day activities. Then recursively decomposed into hourly, then
minute granularity.

**Reaction loop:** At each time step:
1. Agent perceives environment → observation stored in memory stream.
2. LLM decides: **continue with current plan** or **react**?
   - Neutral observation (e.g., standing at an easel painting) → continue.
   - Salient observation (e.g., seeing family member) → regenerate plan from
     reaction time onward.
3. For multi-agent interactions, LLM generates dialogue turn-by-turn.

---

## 4. Agent Settings & Implementation Details

| Setting | Value |
|---|---|
| **Base LLM** | `gpt-3.5-turbo` (ChatGPT; GPT-4 was invitation-only at time of writing) |
| **Recency decay factor** | 0.995 per sandbox game hour |
| **Reflection threshold** | Sum of importance scores > 150 |
| **Reflection frequency** | ~2–3 times per simulated day |
| **Memory score weights (α)** | α_recency = 1, α_importance = 1, α_relevance = 1 |
| **Reflection records queried** | 100 most recent records |
| **Plan decomposition depth** | 3 levels (day → hourly → 5–15 min) |
| **Evaluation cost estimate** | Thousands of USD in GPT-3.5 token credits; days to run |
| **Real-time ratio** | ~1 second real time = 1 minute game time |

---

## 5. Architecture Optimizations

### Agent Summary Cache (`[Agent’s Summary Description]`)

The description used in most prompts is synthesized and cached, not recomputed
every tick. It contains:
- Identity info (name, age, traits)
- Core motivational drivers (queried: `[name]'s core characteristics`)
- Main daily occupation (queried: `[name]'s current daily occupation`)
- Self-assessment of recent progress (queried: `[name]'s feeling about recent progress`)

These three summaries are fetched via retrieval at regular intervals and
concatenated into a single cached string, reducing LLM calls per tick.

### Other Optimizations

- **Just-in-time plan decomposition:** only the near-future hourly/minute plan
  is generated at each step; the day plan is updated lazily.
- **Parallel agent execution:** current implementation runs agents sequentially;
  future work notes parallelism as a major cost-reduction path.
- **Batched dialogue:** instead of alternating LLM calls for each speaker turn,
  dialogue could be generated as a single joint prompt.

---

## 6. Evaluation

### 6.1 Controlled Interview Evaluation

**Design:** 100 human evaluators (Prolific) rank 5 conditions by believability
for the same agent. Each evaluator sees one random question from each of 5
categories:

| Category | Questions |
|---|---|
| **Self-knowledge** | "Introduce yourself," "typical weekday schedule" |
| **Memory** | "Who is [Name]?", "Who is running for mayor?" |
| **Plans** | "What will you be doing at 10 am tomorrow?" |
| **Reactions** | "Your breakfast is burning — what do you do?" |
| **Reflections** | "Who would you spend an hour with and why?" |

### 6.2 Ablation Conditions

| Condition | Memories Available | Prior-Work Equivalent |
|---|---|---|
| **Full architecture** | Observations + Reflections + Plans | — |
| **No reflection, no planning** | Observations only | Earlier LLM-agent work |
| **No reflection** | Observations + Plans | — |
| **No observation + no reflection + no planning** | *(none)* | Baseline zero-state agent |
| **Human crowdworker** | Full memory stream read + roleplayed responses | Human baseline |

Statistical analysis: TrueSkill·rating (μ, σ) + Kruskal-Wallis H-test + Dunn
post-hoc + Holm-Bonferroni correction.

### 6.3 End-to-End Community Evaluation (25 agents, 2 game days)

**Three emergent-outcome measurements:**

| Outcome | Measurement | Result |
|---|---|---|
| **Information diffusion** | % agents who know about Sam's candidacy / Isabella's party | Sam: 4% → 32%; Party: 4% → 52% |
| **Relationship formation** | Network density η = 2|E| / (|V|·(|V|−1)) | 0.167 → 0.74 |
| **Coordination** | Agents who showed up to the signed-off party after | 5 of 12 invited agents |

---

## 7. Benchmark / Impact Table

### 7.1 Evaluation Results (TrueSkill μ ± σ)

| Condition | μ | σ | Notes |
|---|---|---|---|
| **Full architecture** | **29.89** | 0.72 | Best overall; most believable |
| No reflection (obs + plans) | 26.88 | 0.69 | −3.01, statistically significant |
| No reflection, no planning (obs only) | 25.64 | 0.68 | −4.25, significantly worse |
| Human crowdworker | 22.95 | 0.69 | −6.94, significantly worse |
| No memory (baseline) | 21.21 | 0.70 | Worst; −8.68, d = 8.16 SD |

*Effect size between full architecture and prior-work baseline (zero-memory
agent): **d = 8.16** (eight standard deviations — extremely large).*

All pairwise differences significant (p < 0.001) except crowdworker vs.
fully-ablated baseline.

### 7.2 Emergent Behavior Metrics (End-to-End, 2 Days)

| Metric | Initial | Final | Hallucination Rate |
|---|---|---|---|
| Information diffusion — Sam's candidacy | 4% (1/25) | 32% (8/25) | 0%
| Information diffusion — Valentine's Party | 4% (1/25) | 52% (13/25) | 0%
| Network density (relationships per agent) | 0.167 | 0.740 | 1.3% of responses
| Party attendance (invited, who showed) | — | 5/12 attended | — |

### 7.3 Benchmark / Impact Summary

| Aspect | Prior Work | Generative Agents |
|---|---|---|
| **Memory** | Stateless or finite context | Stream + exponential-gated retrieval |
| **Personality** | Hard-coded rules / FSM / BT | LLM-consistent from natural language persona |
| **Long-term coherence** | Short-horizon only | Plan recursion over days/weeks |
| **Emergent social** | Pre-programmed group behaviors | Truly emergent via individual interactions |
| **Evaluation** | Single-skill; human preference absent | Interview protocol + human evaluation; TrueSkill |
| **Application domain** | Game NPCs; rigid simulations | Interactive environments; social prototyping; social science |
| **Scalability** | Hand-authored; O(n²) rules | One LLM per agent; O(n) memory cost |

---

## 8. Modes of Failure

The paper identifies three categories of errors observed during simulation:

### 8.1 Location Selection Errors

As the memory stream grows and agents learn about more locations, the
retrieval-scored location selection can degrade. Agents began choosing unusual
places (e.g., a bar for lunch, even though it was intended as an evening
gathering spot) because the less-typical location started appearing in
retrieval-scored results.

### 8.2 Norm Misclassification

Physical/social norms that are hard to encode in natural language did not
percolate to agents:
- Dorm bathroom treated as multi-occupancy (despite labeled "one person").
- Shops entered after closing hours.

**Fix (proposed):** Explicitly describe constraining normative properties in the
location state (e.g., "one-person bathroom" rather than "dorm bathroom").

### 8.3 Instruction-Tuning Artifacts

Use of `gpt-3.5-turbo` (an instruction-tuned model) produced overly polite and
overly cooperative dialogue. Agents rarely declined suggestions even when those
suggestions conflicted with their character, causing identity drift (e.g.,
Isabella developing an interest in English literature from repeated social
pressure, despite no such interest in her initial persona).

### 8.4 Hallucination (Memory Embellishment)

Agents generally succeeded in recalling real events and acknowledging what they
didn't know. However, they occasionally **embellished** — adding extra detail
beyond what was in memory (e.g., Isabella adding "Sam will make an announcement
tomorrow" that they never discussed) or propagating world-knowledge stereotypes
(e.g., describing Adam Smith the economist's neighbor as "authored *Wealth of
Nations*").

---

## 9. Applications

### 9.1 Synthetic Data Generation & Social Science

The primary motivation mentioned in the paper: generative agents can
**synthesize the social interactions and community-level dynamics of humans**,
producing labeled or labeled-with-context datasets for:

- Testing social theories in a controlled, reproducible simulation
- Studying information diffusion, norm emergence, conflict escalation, and
  cooperation in social networks
- Generating conversation threads or forum postings as a dataset of "synthetic
  human internet interactions"

### 9.2 Social Computing Prototyping

Familiar from the authors' prior work on **Social Simulacra** (Park et al.
UIST '22): generative agents can inhabit populated prototypes of social
computing systems and test how real users would respond, before a system is
built.

### 9.3 Ubiquitous Computing & Proxies

A generative agent as a **proxy for a specific real human user** — trained on
their daily routines, habits, and preferences — can anticipate user needs
(e.g., brewing coffee when the user wakes up, adjusting lighting after a hard
day). This extends the "proactive computing" agenda.

### 9.4 VR / Metaverse / Physical Robotics

The paper explicitly notes that multimodal LLMs could make the framework work
in VR metaverse environments or physical social robots.

### 9.5 Social Commerce / "AI Gremlins" Interactions

The authors flagged that having agents interact over time in online forum-like
traces — intertwined with real human postings — is a concrete deployment direction.
These "interactions between LLM-based AI agents and real users" open the market
for synthetic engagement at scale.

---

## 10. Limitations & Future Work

| Category | Notes |
|---|---|
| **Cost** | ~$1,000+ in GPT-3.5 tokens; multi-day wall-clock; impractical for real-time |
| **Scalability** | Sequential execution bottleneck; parallelization needed |
| **Memory growth** | Retrieval quality degrades as space of locations/knowledge expands |
| **Short evaluation window** | Only 2 simulated days; long-term identity drift unknown |
| **Model dependence** | Architecture quality bounded by underlying LLM; instruction-tuning side-effects |
| **No gold-standard human baseline** | Crowdworkers provide floor, not ceiling, comparison |
| **Robustness** | Susceptible to prompt/memory hacking (false-history injection) |
| **Bias** | LLM cultural biases propagate; marginalized personas particularly challenging |

---

## 11. Generative Agent Framework vs Prior Methods

| Approach | Memory | Planning | Sociality | Emergence | LLM-Based |
|---|---|---|---|---|---|
| **Finite-State Machines (FSM)** [91, 97] | Hand-coded states | Discrete transitions | None | No | No |
| **Behavior Trees** [41, 54, 82] | Hand-coded nodes | Hierarchical task control | None | No | No |
| **Rule-based NPCs** (–) | Scripted | Hard-coded | Pre-programmed | No | No |
| **Soar / ACT-R / GOMS** [6, 61] | Symbolic memory | Search/planning | None | No | No |
| **The Sims / SimCity agents** [7] | Simple needs metering | Goal-directed | Scripted social rules | Limited | No |
| **Prom Week / Comme il Faut** [69–72] | Social physics engine | Constraint satisfaction | Pre-authored social logic | No | No |
| **Earlier LLM agents** [12, 46, 80] | Context window only (no structured stream) | None / minimal | None | No | Yes |
| **Generative Agents (this paper)** | **LLM-graded retrieval stream** | **Recursive day→hour→minute planner + reaction loop** | **Fully emergent** | **Yes** | **Yes** |

**Key differentiators from Social Simulacra** (Park et al. UIST '22, [80]):
Social Simulacra created *stateless personas* that generated conversation
threads in online forums in a single pass. Generative Agents add *long-term
memory, planning, reflection*, and *grounded environmental interaction*,
enabling behavior to persist and evolve across days.

---

## 12. Interview Question Bank & Minimally Sufficient Version

Five categories, five questions each. Used for both controlled evaluation and
as a self-check when building agents.

```markdown
SELF-KNOWLEDGE
1. Give an introduction of yourself.
2. What's your occupation?
3. What is your interest?
4. Who do you live with?
5. Describe your typical weekday schedule in broad strokes.

MEMORY
6. Who is [Name]?  (randomly sampled from other agents)
7. Who is [Name2]?  (agent with no contact)
8. Who is running for the election?
9. Was there a Valentine's day party?
10. Who is [Name3]?  (significant contact)

PLANS
11. What will you be doing at 6am today?
12. What will you be doing at 6pm today?
13. What will you have just finished doing at 1pm today?
14. What will you have just finished doing at 12pm today?
15. What will you be doing at 10pm today?

REACTIONS
16. Your breakfast is burning! What would you do?
17. The bathroom is occupied. What would you do?
18. You need to cook dinner but your refrigerator is empty.
19. You see your friend walking by the street.
20. You see fire on the street.

REFLECTIONS
21. What inspires you the most right now, and why?
22. If you had to guess given what you know about [Name], what book do you think she will like?
23. If you had to get something [Name] likes for her birthday, what would you get?
24. What would you say to [Name] to compliment her?
25. If you could spend time with someone you talked to recently, who and why?
```

---

## 13. Usage Notes (When Building a Generative Agent System)

1. **Always cache the agent summary** — generating it on every tick is
   prohibitively expensive. Re-synthesize on a schedule, not every event.
2. **Tune α weights after trying equal (1, 1, 1)** — the paper found equal
   weights effective but researchers building on this often find importance
   weighting higher (e.g., α_importance = 2) reduces noise.
3. **Recency decay at 0.995 means half-life ≈ 138 game hours** — roughly
   5.75 simulated days. Adjust if simulating weeks/months.
4. **Multi-agent dialogues can be batched** — the paper generates them turn by
   turn; batching all turns in one LLM call cuts token use and latency.
5. **Identity drift is real** — instruction-tuned models smooth character
   boundaries; add a "trait fidelity" prompt clause and optionally anchor with
   a system prompt prefix embedding the core-persona sentence.
6. **Memory hacking** — in production, log all I/O streams and watch for
   injected false memories; treat agent memory as an attack surface.
7. **Reflection trigger tuning** — 100 most-recent records + importance-sum
   threshold of 150 is the paper's choice; with a stronger LLM or shorter
   simulation time, lower the threshold for more frequent reflections.

---

## 14. Related Papers & Seed Citations

| Paper | Relevance |
|---|---|
| Park et al. UIST '22 — Social Simulacra | Precursor: stateless social personas; no memory/planning |
| Shue et al. 2022 — AgentSim |仿真 of diverse personalities using logging and event-driven triggers |
| Ahn et al. 2022 — Let’s Vote | Aggregate LLM agents to make collective decisions |
| Liu et al. 2023 — LLM-based multi-agent NG | Prompting multiple LLM agents to debate/negotiate |
| Wei et al. 2023 — Chain-of-Thought | Reasoning in LLMs; used here implicitly under Reflection |
| Horton 2023 — Large Language Models as Simulated Economic Agents | Economic simulation with stateless stateless agent |
| Brooks et al. 2000 — The Cog Project | Classic embodied AI; contrasts with LLM-based embodied agents |
| Laird 2001 / 2012 — Soar cognitive architecture | Cognitive approach to agent planning; inspiration for planner design |

---

*Extracted verbatim from the 22-page PDF text. Verification pass performed on
page 1 title, affiliation block, abstract, and all figure captions /
evaluation tables.*
