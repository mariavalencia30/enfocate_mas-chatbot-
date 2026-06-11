import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results = [
    {
        "name": "GPT-4o\nchunk500",
        "f1": 0.9316,
        "latency": 1.743,
        "cost": 2.50,
        "color": "#FF5722",
    },
    {
        "name": "GPT-4o\nchunk200",
        "f1": 0.9316,
        "latency": 1.563,
        "cost": 2.50,
        "color": "#E64A19",
    },
    {
        "name": "GPT-4o-mini\nchunk500",
        "f1": 0.6795,
        "latency": 0.694,
        "cost": 0.15,
        "color": "#2196F3",
    },
    {
        "name": "GPT-4o-mini\nchunk200",
        "f1": 0.6941,
        "latency": 0.738,
        "cost": 0.15,
        "color": "#64B5F6",
    },
    {
        "name": "GPT-4o-mini\nchunk1000",
        "f1": 0.6795,
        "latency": 0.835,
        "cost": 0.15,
        "color": "#0D47A1",
    },
    {
        "name": "GPT-3.5-turbo\nchunk500",
        "f1": 1.0000,
        "latency": 1.000,
        "cost": 0.50,
        "color": "#4CAF50",
    },
    {
        "name": "GPT-4o-mini\nSentTransf",
        "f1": 0.6795,
        "latency": 0.545,
        "cost": 0.15,
        "color": "#9C27B0",
    },
    {
        "name": "Llama3-local\nSentTransf",
        "f1": 0.0410,
        "latency": 2.006,
        "cost": 0.00,
        "color": "#9E9E9E",
    },
]

df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Benchmark RAG — Comparación de Modelos", fontsize=16, fontweight="bold")

colors = df["color"]

# Gráfico 1: F1-Score
ax1 = axes[0]
bars1 = ax1.barh(df["name"], df["f1"], color=colors, edgecolor="white", linewidth=1.5)
ax1.set_xlabel("F1-Score", fontsize=11)
ax1.set_title("F1-Score (Precisión del Modelo)", fontsize=12, fontweight="bold")
ax1.set_xlim(0, 1.1)
for i, v in enumerate(df["f1"]):
    ax1.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
ax1.axvline(
    x=0.85, color="green", linestyle="--", alpha=0.5, label="Meta mínima (0.85)"
)
ax1.legend(fontsize=8)

# Gráfico 2: Latencia
ax2 = axes[1]
bars2 = ax2.barh(
    df["name"], df["latency"], color=colors, edgecolor="white", linewidth=1.5
)
ax2.set_xlabel("Latencia (segundos)", fontsize=11)
ax2.set_title("Latencia Promedio", fontsize=12, fontweight="bold")
ax2.axvline(x=1.0, color="green", linestyle="--", alpha=0.5, label="Meta (< 1s)")
ax2.legend(fontsize=8)
for i, v in enumerate(df["latency"]):
    ax2.text(v + 0.05, i, f"{v:.2f}s", va="center", fontsize=9)

# Gráfico 3: Costo
ax3 = axes[2]
bars3 = ax3.barh(df["name"], df["cost"], color=colors, edgecolor="white", linewidth=1.5)
ax3.set_xlabel("Costo ($/1K tokens)", fontsize=11)
ax3.set_title("Costo por 1K Tokens", fontsize=12, fontweight="bold")
for i, v in enumerate(df["cost"]):
    ax3.text(v + 0.05, i, f"${v:.2f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(
    "experiments/results/06_comparacion_final.png", dpi=150, bbox_inches="tight"
)
plt.show()
print("✅ Guardado: experiments/results/06_comparacion_final.png")

# Gráfico 4: Costo vs F1 scatter
fig2, ax = plt.subplots(figsize=(10, 6))
for r in results:
    ax.scatter(
        r["cost"],
        r["f1"],
        s=300,
        c=r["color"],
        alpha=0.8,
        edgecolors="white",
        linewidth=2,
    )
    ax.annotate(
        r["name"].replace("\n", " "),
        (r["cost"], r["f1"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )

ax.set_xlabel("Costo ($/1K tokens)", fontsize=11)
ax.set_ylabel("F1-Score", fontsize=11)
ax.set_title(
    "Costo vs Calidad — Modelo Elegido: GPT-4o-mini", fontsize=13, fontweight="bold"
)


ax.scatter(
    [0.15],
    [0.6795],
    s=600,
    facecolors="none",
    edgecolors="#2196F3",
    linewidth=4,
    label="ELEGIDO",
)
ax.legend(fontsize=10)

# Zonas
ax.axhspan(0.85, 1.0, alpha=0.1, color="green", label="Zona óptima")
ax.axvspan(0, 0.5, alpha=0.1, color="blue", label="Zona económica")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("experiments/results/07_costo_vs_calidad.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: experiments/results/07_costo_vs_calidad.png")

print("\n Gráficos finales generados en experiments/results/")
