## Model Architecture and Training

**Base model.** We use OSCAR (naver/oscar-qwen2-7B), a 7B-parameter document compression model based on Qwen2-7B, which encodes each document into a sequence of 8 memory tokens. The hidden state dimension of OSCAR is 28,672 (= 4,096 × 7, or equivalently the concatenation of the 8 memory token hidden states of dimension 3,584 each).

**Projector.** To map OSCAR's compressed memory representations into a dense retrieval embedding space, we introduce a lightweight *per-token gated projector* (MEMProjectorGated). Given the concatenated hidden states of the 8 memory tokens $\mathbf{m} \in \mathbb{R}^{B \times 28672}$, the projector first reshapes the input into per-token representations $\mathbf{x} \in \mathbb{R}^{B \times 8 \times 3584}$ and applies a shared linear token projection:

\[\mathbf{h}_i = \text{ReLU}(\mathbf{W}_{\text{proj}}\, \mathbf{x}_i + \mathbf{b}), \quad \mathbf{h}_i \in \mathbb{R}^{1024}\]

A scalar gate is computed per token via a linear layer without bias:

\[g_i = \frac{\exp(\mathbf{w}_g^\top \mathbf{h}_i)}{\sum_{j=1}^{8} \exp(\mathbf{w}_g^\top \mathbf{h}_j)}\]

The gated pool aggregates token representations as a weighted sum:

\[\mathbf{p} = \sum_{i=1}^{8} g_i \cdot \mathbf{h}_i\]

A two-layer MLP head then projects $\mathbf{p}$ to the target embedding space:

\[\mathbf{e} = \text{LayerNorm}\!\left(\mathbf{W}_2\,\text{Dropout}\!\left(\text{ReLU}(\mathbf{p})\right)\right) \in \mathbb{R}^{768}\]

The final embedding is L2-normalized. All projector weights are initialized with Xavier uniform; the projector is trained in bfloat16. The projector contains approximately **47M parameters**.

***

## Training Objective

Training proceeds in two stages, separated at step $t_1 = 1{,}500$.

**Stage 1 (steps 0–1,499): MSE anchoring.** The model is trained to align query and document embeddings with pre-computed teacher embeddings from BGE-base-en-v1.5:

\[\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{MSE}}^{q} + \mathcal{L}_{\text{MSE}}^{d}\]

where $\mathcal{L}_{\text{MSE}}^{q} = \|\mathbf{e}_q - \hat{\mathbf{e}}_q\|^2$ and $\mathcal{L}_{\text{MSE}}^{d}$ is the analogous loss for the positive document embedding. This stage ensures geometrically stable initialization of the embedding space before contrastive learning begins.

**Stage 2 (steps 1,500+): Contrastive distillation.** The full loss combines four terms:

\[\mathcal{L}_{\text{stage2}} = \lambda_{\text{KL}}\,\mathcal{L}_{\text{KL}} + \lambda_{\text{NCE}}\,\mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{rank}}\,\mathcal{L}_{\text{margin}} + \lambda_{\text{MSE}}\,\mathcal{L}_{\text{MSE}}^{q}\]

with weights $\lambda_{\text{KL}} = 0.3$, $\lambda_{\text{NCE}} = 0.05$, $\lambda_{\text{rank}} = 0.02$, $\lambda_{\text{MSE}} = 1.0$.

**Listwise KL distillation** ($\mathcal{L}_{\text{KL}}$). For each query $q$ with $K$ candidates, teacher and student score distributions are computed at temperature $\tau = 0.07$:

\[\mathcal{L}_{\text{KL}} = \sum_{k=1}^{K} p^{\text{teacher}}_k \log\frac{p^{\text{teacher}}_k}{p^{\text{student}}_k}\]

**Global in-batch InfoNCE** ($\mathcal{L}_{\text{InfoNCE}}$). All $B \times K$ candidate documents in a batch are used as negatives. The positive for query $i$ is the candidate with the highest teacher score:

\[\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(s_{i,k^*}/\tau)}{\sum_{j=1}^{BK}\exp(s_{i,j}/\tau)}\]

where $k^* = \arg\max_k\, \hat{s}_{i,k}^{\text{teacher}}$.

**Hard negative margin loss** ($\mathcal{L}_{\text{margin}}$). Given the hardest in-batch negative $s_i^-$:

\[\mathcal{L}_{\text{margin}} = \frac{1}{B}\sum_{i=1}^{B}\max\!\left(0,\; m - s_i^+ + s_i^-\right), \quad m = 0.15\]

***

## Training Data

The projector is trained on six IR datasets with teacher embeddings pre-computed offline using BGE-base-en-v1.5. Each dataset provides query–document pairs with BM25-mined hard negatives ($K = 50$ candidates per query). Dataset sampling follows fixed weights during training: MS MARCO (0.85), SciFact (0.03), TREC-COVID (0.03), FiQA (0.03), NFCorpus (0.02), ArguAna (0.02), Quora (0.02). The heavy weighting toward MS MARCO ensures broad generalization, while the domain-specific datasets provide signal on specialized retrieval tasks.

***

## Training Procedure

The projector is trained with AdamW ($\text{lr} = 2 \times 10^{-5}$, $\text{min\_lr} = 10^{-6}$, weight decay $= 0.01$) using a cosine learning rate schedule with 300 warmup steps, for 5 epochs with batch size 64. The OSCAR backbone is frozen throughout; only the projector weights are updated. Validation is performed every 500 steps on a held-out 2% split of the training data, with retrieval quality monitored via a BEIR probe set (120 samples each from FiQA-2018, NFCorpus, and SciFact) using BM25-mined hard negatives. The final model checkpoint is selected by BEIR full ndcg@10 evaluated at epoch end.

