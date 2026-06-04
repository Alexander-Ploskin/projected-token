import re
import sys

def parse_log(filename):
    steps = []
    
    with open(filename, 'r') as f:
        current_step = None
        data = {}
        for line in f:
            if "[val] step=" in line:
                if current_step is not None:
                    steps.append(data)
                current_step = int(line.strip().split("=")[1])
                data = {'step': current_step}
            elif "[val] retrieval" in line and current_step is not None:
                parts = line.strip().split()
                for p in parts[2:]:
                    if "=" in p:
                        k, v = p.split("=")
                        data[k] = float(v)
            elif "[val] losses" in line and current_step is not None:
                parts = line.strip().split()
                for p in parts[2:]:
                    if "=" in p:
                        k, v = p.split("=")
                        data[k] = float(v)
            elif "[val] batch_stats" in line and current_step is not None:
                parts = line.strip().split()
                for p in parts[2:]:
                    if "=" in p:
                        k, v = p.split("=")
                        data[k] = float(v)
            elif "Epoch" in line and current_step is not None:
                # parse grad and lr
                m = re.search(r'grad=([0-9.]+)', line)
                if m:
                    data['grad'] = float(m.group(1))
                m = re.search(r'lr=([0-9.e-]+)', line)
                if m:
                    data['lr'] = float(m.group(1))
        
        if current_step is not None:
            steps.append(data)
            
    return steps

def print_summary(steps):
    print(f"{'Step':>6} | {'ndcg@10':>8} | {'inter_ndcg':>10} | {'recall@10':>9} | {'gap':>7} | {'gap_std':>7} | {'rank_loss':>9} | {'infonce':>7} | {'kl_mean':>7} | {'top1_agr':>8} | {'act_frac':>8} | {'grad':>6} | {'lr':>8}")
    print("-" * 125)
    for d in steps:
        step = d.get('step', 0)
        ndcg = d.get('beir_probe_ndcg@10', 0)
        inter_ndcg = d.get('beir_interquery_ndcg@10', 0)
        recall = d.get('beir_probe_recall@10', 0)
        gap = d.get('gap', 0)
        gap_std = d.get('gap_std', 0)
        rank_loss = d.get('ranking_loss', 0)
        infonce = d.get('infonce_loss', 0)
        
        # kl and top1 agreement are in batch_stats, we have max/min/std but maybe not mean?
        # let's approximate or just print max/min
        kl_max = d.get('teacher_listwise_kl_batch_max', 0)
        kl_min = d.get('teacher_listwise_kl_batch_min', 0)
        kl_mean = (kl_max + kl_min) / 2 # rough
        
        top1_max = d.get('teacher_top1_agreement_batch_max', 0)
        top1_min = d.get('teacher_top1_agreement_batch_min', 0)
        top1_mean = (top1_max + top1_min) / 2 # rough
        
        act_frac = d.get('ranking_active_frac', 0)
        grad = d.get('grad', 0)
        lr = d.get('lr', 0)
        
        print(f"{step:6d} | {ndcg:8.4f} | {inter_ndcg:10.4f} | {recall:9.4f} | {gap:7.4f} | {gap_std:7.4f} | {rank_loss:9.4f} | {infonce:7.4f} | {kl_mean:7.4f} | {top1_mean:8.4f} | {act_frac:8.4f} | {grad:6.4f} | {lr:8.2e}")

if __name__ == "__main__":
    steps = parse_log(sys.argv[1])
    print_summary(steps)
