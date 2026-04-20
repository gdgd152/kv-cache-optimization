# main.py
import matplotlib.pyplot as plt

from model import load_model
from eval_ppl import (
    get_wikitext, get_pg19,
    compute_ppl_wikitext, compute_ppl_pg19,
    compute_ppl_wikitext_with_cache,
    compute_ppl_pg19_with_cache
)
from generate import generate


# ===== 工具函数1：滑动平均 =====
def smooth(data, window=10):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(sum(data[start:i+1]) / (i - start + 1))
    return smoothed


# ===== 工具函数2：多次运行取平均 =====
def run_generation_avg(model, tokenizer, device, prompt,
                       kv_method, kv_params,
                       max_new_tokens=300,
                       repeat=3):

    all_tpot = []
    ttft_list, first_list, tpot_avg_list, th_list = [], [], [], []

    for i in range(repeat):
        print(f"  Run {i+1}/{repeat}...")

        ttft, first_tok, avg_tpot, throughput, tpot = generate(
            model, tokenizer, device, prompt,
            max_new_tokens=max_new_tokens,
            kv_method=kv_method,
            kv_params=kv_params
        )

        all_tpot.append(tpot)
        ttft_list.append(ttft)
        first_list.append(first_tok)
        tpot_avg_list.append(avg_tpot)
        th_list.append(throughput)

    # ===== 对齐长度 =====
    min_len = min(len(t) for t in all_tpot)
    avg_tpot_curve = []

    for i in range(min_len):
        avg_tpot_curve.append(sum(run[i] for run in all_tpot) / repeat)

    # ===== 标量取平均 =====
    avg_ttft = sum(ttft_list) / repeat
    avg_first = sum(first_list) / repeat
    avg_tpot = sum(tpot_avg_list) / repeat
    avg_throughput = sum(th_list) / repeat

    return avg_ttft, avg_first, avg_tpot, avg_throughput, avg_tpot_curve


# ===== 作图函数 =====
def plot_curve(results, title, save_name):
    plt.figure(figsize=(10, 6))

    for name, (ttft, first, tpot, th, curve) in results.items():
        curve_smooth = smooth(curve, window=10)
        steps = list(range(len(curve_smooth)))

        plt.plot(steps, curve_smooth, label=f"{name} ({th:.1f} tok/s)")

    plt.xlabel("Generation Step")
    plt.ylabel("Time per Token (s)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    print(f"Saved: {save_name}")
    plt.show()


# ===== 主函数 =====
def run():
    print("Loading model...")
    model, tokenizer, device = load_model()

    print("Loading datasets...")
    wiki_texts = get_wikitext()
    pg_text = get_pg19()

    # =========================
    # 1. Sliding Window PPL
    # =========================
    print("\n===== Sliding Window PPL =====")
    wiki_subset = wiki_texts[:100]

    ppl_wiki = compute_ppl_wikitext(model, tokenizer, device, wiki_subset)
    print(f"WikiText (baseline): {ppl_wiki:.2f}")

    ppl_pg = compute_ppl_pg19(model, tokenizer, device, pg_text)
    print(f"PG19 (baseline): {ppl_pg:.2f}")

    # =========================
    # 2. KV Cache PPL
    # =========================
    print("\n===== KV Cache PPL =====")

    cache_methods = [
        ("Baseline", None, {}),
        ("Truncate", "truncate", {"max_length": 256}),
        ("Streaming", "streaming", {"sink_size": 4, "window_size": 256}),
    ]

    for name, method, params in cache_methods:
        ppl = compute_ppl_wikitext_with_cache(
            model, tokenizer, device, wiki_subset,
            kv_method=method, kv_params=params
        )
        print(f"WikiText - {name}: {ppl:.2f}")

    for name, method, params in cache_methods:
        ppl = compute_ppl_pg19_with_cache(
            model, tokenizer, device, pg_text,
            kv_method=method, kv_params=params
        )
        print(f"PG19 - {name}: {ppl:.2f}")

    # =========================
    # 3. Generation (PG19)
    # =========================
    print("\n===== PG19 Generation =====")

    prompt_pg = pg_text[:3000]

    pg_results = {}

    for name, method, params in cache_methods:
        print(f"\nRunning {name}...")

        result = run_generation_avg(
            model, tokenizer, device,
            prompt_pg,
            kv_method=method,
            kv_params=params,
            max_new_tokens=300,
            repeat=3
        )

        pg_results[name] = result

        print(f"{name}")
        print(f"TTFT: {result[0]:.4f}s")
        print(f"First token: {result[1]:.4f}s")
        print(f"Avg TPOT: {result[2]:.4f}s")
        print(f"Throughput: {result[3]:.2f}")

    plot_curve(pg_results, "PG19 TPOT Curve (Long Context)", "tpot_pg19.png")

    # =========================
    # 4. Generation (WikiText)
    # =========================
    print("\n===== WikiText Generation =====")

    prompt_wiki = wiki_texts[0][:1000]

    wiki_results = {}

    for name, method, params in cache_methods:
        print(f"\nRunning {name}...")

        result = run_generation_avg(
            model, tokenizer, device,
            prompt_wiki,
            kv_method=method,
            kv_params=params,
            max_new_tokens=300,
            repeat=3
        )

        wiki_results[name] = result

        print(f"{name}")
        print(f"TTFT: {result[0]:.4f}s")
        print(f"First token: {result[1]:.4f}s")
        print(f"Avg TPOT: {result[2]:.4f}s")
        print(f"Throughput: {result[3]:.2f}")

    plot_curve(wiki_results, "WikiText TPOT Curve (Short Context)", "tpot_wikitext.png")


if __name__ == "__main__":
    run()