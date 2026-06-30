import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGS_DIR = "outputs/figures"


class BenchmarkVisualizer:
    def __init__(self, figs_dir: str = FIGS_DIR) -> None:
        self.figs_dir = figs_dir

    def latency_chart(self, data: dict) -> str:
        models = list(data.keys())
        labels = list(next(iter(data.values())).keys())
        x = np.arange(len(labels))
        w = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, m in enumerate(models):
            vals = [data[m][label]["latency_s"] for label in labels]
            ax.bar(x + i * w, vals, w, label=m)
        ax.set_xlabel("Prompt Length")
        ax.set_ylabel("Latency (s)")
        ax.set_title("Inference Latency by Model & Prompt Length")
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        path = f"{self.figs_dir}/latency.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def throughput_chart(self, data: dict) -> str:
        models = list(data.keys())
        labels = list(next(iter(data.values())).keys())
        x = np.arange(len(labels))
        w = 0.2
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, m in enumerate(models):
            vals = []
            for label in labels:
                d = data[m][label]
                tok = d["tokens"]
                sec = d["latency_s"]
                vals.append(tok / sec if sec > 0 else 0)
            ax.bar(x + i * w, vals, w, label=m)
        ax.set_xlabel("Prompt Length")
        ax.set_ylabel("Tokens / Second")
        ax.set_title("Throughput by Model & Prompt Length")
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()
        path = f"{self.figs_dir}/throughput.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def prompt_scale_chart(self, data: dict) -> str:
        models = list(data.keys())
        labels = list(next(iter(data.values())).keys())
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in models:
            vals = [data[m][label]["latency_s"] for label in labels]
            ax.plot(labels, vals, marker="o", label=m)
        ax.set_xlabel("Prompt Length")
        ax.set_ylabel("Latency (s)")
        ax.set_title("Latency Scaling with Prompt Length")
        ax.legend()
        fig.tight_layout()
        path = f"{self.figs_dir}/prompt_scale.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def radar_chart(self, data: dict) -> str:
        models = list(data.keys())
        categories = ["Short", "Medium", "Long"]
        n = len(categories)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        for m in models:
            vals = [data[m][label]["latency_s"] for label in categories]
            vals += vals[:1]
            ax.plot(angles, vals, label=m)
            ax.fill(angles, vals, alpha=0.05)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Model Latency Radar")
        ax.legend(loc="upper right")
        fig.tight_layout()
        path = f"{self.figs_dir}/radar.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def generate_all(self, data: dict) -> dict[str, str]:
        return {
            "latency": self.latency_chart(data),
            "throughput": self.throughput_chart(data),
            "prompt_scale": self.prompt_scale_chart(data),
            "radar": self.radar_chart(data),
        }
