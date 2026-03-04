# SWE-WorldPRM: Process-Reward Model for Software Engineering Agents

## 1. Motivation
LLM agents struggle with long-horizon, multi-step decisions in software engineering (SWE) because feedback is sparse and environments are non-serializable. Current methods rely on noisy temporal difference (TD) learning or expensive LLM-as-a-judge calls. Small models fail to learn trajectories effectively under these conditions.

**SWE-WorldPRM** aims to solve this by:
1.  **Modeling Environment Dynamics:** Predicting the next state to ground reward signals.
2.  **Dynamic Subtask Tracking:** Using a checklist to monitor progress at each step.

---

## 2. Limitations of Existing Approaches
* **Limitation 1 (Environment Complexity):** No effective PRM or world model exists specifically for SWE; evaluation is usually restricted to outcome-level metrics.
* **Limitation 2 (TD Learning Noise):** Bootstrapping terminal rewards can amplify noise along a trajectory, leading to unreliable signals.
* **Limitation 3 (Disjoint Modeling):** Separating world models (state prediction) and reward models (evaluation) ignores highly complementary signals.
* **Limitation 4 (Lack of Subtask Awareness):** Most PRMs lack explicit progress tracking against subgoals.

---

## 3. Research Questions
* **RQ1:** How can we produce high-quality, stable step-level labels for SWE agents at a manageable cost?
* **RQ2:** Does joint training with world modeling (next-state prediction) improve PRM performance and stability?
* **RQ3:** What is the optimal strategy for combining process-level and outcome-level rewards (e.g., via TD learning)?
* **RQ4:** How should the PRM handle dynamic checklists that evolve as actions reveal new information?

---

## 4. Methodology: SWE-WorldPRM

### 4.1 System Overview


SWE-WorldPRM maintains a dynamic checklist $p_t$ that summarizes subgoals and completion status. For a task specification $x$, history $h_t$, and candidate action $a_t$, the model predicts:

1.  **$\hat{o}_{t+1}$**: Predicted next observation (World Model).
2.  **$p_{t+1}$**: Updated checklist.
3.  **$\Delta_t$**: Progress update grounded in the checklist.
4.  **$R_t$**: Process reward with reasoning.

### 4.2 Formal Definition
The transition dynamics are modeled in the observation space:
$$P(\hat{o}_{t+1} \mid x, h_t, a_t)$$

The reward model estimates action quality:
$$R(x, h_t, p_t, a_t, \hat{o}_{t+1})$$

### 4.3 Key Components
* **World Model Training:** Pre-training/warmup on environment dynamics (next observation prediction) using logged interaction data.
* **Dynamic Checklist:** Online revision of subtasks; new subtasks are added or refined based on the evidence revealed by actions.
* **Hybrid Rewards:** Combining dense checklist-based rewards with sparse outcome rewards via TD learning to anchor the agent to task success.

---

## 5. Implementation Plan

### Phase 1: Prompt-based PRM (Baselines)
* Implement various prompting strategies: Vanilla, Rubric-based (SWE-Search), Observed next-state, and Dynamic Checklist.
* Identify cost-performance trade-offs of using frontier models (Sonnet-4.5, GPT-5) as judges.

### Phase 2: SWE-PRMBench Construction
* Create a benchmark of 50–100 expert trajectories from SWE-Bench-Verified.
* Collect 1,500–3,000 step-level manual annotations (action ranking + feedback).

### Phase 3: Model Finetuning
* Generate 2k trajectories (~50k steps) of training data.
* Finetune Qwen3-14B/30B models using joint objectives (World Modeling + PRM).

---

## 6. Evaluation Framework

### Benchmarks
* **SWE-Bench-Verified:** Execution-based evaluation on real-world GitHub issues.
* **SWE-PRMBench:** Pairwise preference accuracy versus human labels.

### Metrics
* **Solve Rate:** Final task success percentage.
* **Efficiency:** Average steps per trajectory.
* **Trajectory Quality:** Rate of invalid tool calls or redundant edits.
* **Generalization:** Performance across different scaffolds (OpenHands, SWE-Agent).

---

## 7. Related Work Comparison

| Reward Model | Level | Signal | Base Model | Data |
| :--- | :--- | :--- | :--- | :--- |
| **Guided-Search** | Process | TD Learning | LLaMA 3.1-70B | Rejection Sampling |
| **SWE-Search** | Process | LLM-Judge | Qwen-2.5-72B | Prompt-only |
| **SWE-Gym Verifier**| Outcome | F2P Tests | Qwen-2.5-32B | On/Off Policy |
| **SWE-WorldPRM** | **Process** | **Checklist + World** | **Qwen3-30B** | **SWE-Smith/R2E** |

---