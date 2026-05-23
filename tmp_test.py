import torch
from projected_token.training.recipes.trainer_query_distill import QueryDistillTrainer

class DummyArgs:
    stage2_teacher_listwise_kl_weight = 0.65
    stage2_ranking_weight = 0.1
    stage2_infonce_weight = 0.2
    batch_size = 8
    gradient_accumulation_steps = 1
    lr = 1e-4
    min_lr = 1e-6
    max_steps = 100
    warmup_steps = 10
    scheduler = "cosine"
    val_split = 0.0
    query_mse_weight = 1.0
    doc_mse_weight = 1.0
    negative_mse_weight = 1.0
    ranking_weight = 0.1
    infonce_weight = 0.1
    temperature = 0.05
    margin = 0.15
    beir_probe_config = None
    beir_probe_samples = 10
    beir_probe_samples_per_dataset = None

trainer = QueryDistillTrainer(
    model=torch.nn.Linear(10, 10),
    optimizer=torch.optim.Adam(torch.nn.Linear(10, 10).parameters()),
    train_loader=[],
    val_loader=[],
    device=torch.device("cpu"),
    run_dir=".",
    **{k: v for k, v in DummyArgs.__dict__.items() if not k.startswith("__")}
)

q = torch.randn(8, 10)
p = torch.randn(8, 10)
n = torch.randn(8, 10)
q_t = torch.randn(8, 10)
p_t = torch.randn(8, 10)
n_t = torch.randn(8, 10)

kl = trainer._teacher_listwise_kl(q, p, n, q_t, p_t, n_t)
print("KL shape:", kl.shape)
print("KL value:", kl.item())
