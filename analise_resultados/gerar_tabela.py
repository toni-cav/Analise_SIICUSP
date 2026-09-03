import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

PASTA = "analise_resultados"

PRETO = "#0b0b0b"
CINZA = "#52514e"

tabela = pd.read_csv(f"{PASTA}/V6_tabela_modelos.csv")

nomes_desejados = ["Árvore de decisão", "Naive Bayes", "SVM", "Random Forest","XGBoost (Optuna)"]

CONFIG = {
    "pt": {
        "nomes_exibicao": {
            "Árvore de decisão": "Árvore de Decisão",
            "Naive Bayes": "Naive Bayes",
            "SVM": "Support Vector Machine",
            "Random Forest": "Random Forest",
            "XGBoost (Optuna)": "XGBoost",
        },
        "cabecalho": ["Modelo", "F1-score", "Revocação", "Precisão", "AUC-ROC"],
        "titulo": "Tabela 1. Desempenho dos modelos no conjunto de teste.",
        "nota": "Nota: F1-score, Recall e Precisão correspondem à média macro entre as duas classes. Valores em negrito indicam o melhor resultado por coluna.",
        "arquivo": "tabela_cientifica_pt.png",
    },
    "en": {
        "nomes_exibicao": {
            "Árvore de decisão": "Decision Tree",
            "Naive Bayes": "Naive Bayes",
            "SVM": "Support Vector Machine",
            "Random Forest": "Random Forest",
            "XGBoost (Optuna)": "XGBoost",
        },
        "cabecalho": ["Model", "F1-score", "Recall", "Precision", "AUC-ROC"],
        "titulo": "Table 1. Model performance on the test set.",
        "nota": "Note: F1-score, Recall, and Precision are macro-averaged across both classes. Bold values indicate the best result per column.",
        "arquivo": "tabela_cientifica_en.png",
    },
}


def gerar(idioma):
    cfg = CONFIG[idioma]
    sub = tabela[tabela["Modelo"].isin(nomes_desejados)].copy()
    sub["ordem"] = sub["Modelo"].map({n: i for i, n in enumerate(nomes_desejados)})
    sub = sub.sort_values("ordem")

    linhas = []
    for _, r in sub.iterrows():
        linhas.append([
            cfg["nomes_exibicao"][r["Modelo"]],
            f"{r['F1 (macro)']:.3f}",
            f"{r['Recall (macro)']:.3f}",
            f"{r['Precisão (macro)']:.3f}",
            f"{r['ROC-AUC']:.3f}",
        ])

    # acha, pra cada coluna numerica (j=1..4), o indice da linha com maior valor
    melhores = {}
    for j in range(1, 5):
        col_vals = [float(l[j]) for l in linhas]
        melhores[j] = col_vals.index(max(col_vals))

    n_linhas = len(linhas)
    n_cols = 5
    larguras = [1.7, 1.0, 1.0, 1.0, 1.0]
    bordas_x = [0]
    for w in larguras:
        bordas_x.append(bordas_x[-1] + w)
    escala = n_cols / bordas_x[-1]
    bordas_x = [b * escala for b in bordas_x]

    ALTURA_LINHA = 0.75
    TOPO = (n_linhas + 1) * ALTURA_LINHA + 0.3

    fig_w, fig_h = 8.6, TOPO * 0.70
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(-0.15, TOPO)
    ax.axis("off")

    def y_linha(i):
        return TOPO - 0.15 - i * ALTURA_LINHA

    y_cab_topo = y_linha(0)
    y_regua_superior = y_cab_topo
    y_regua_cabecalho = y_cab_topo - ALTURA_LINHA
    y_regua_inferior = y_linha(n_linhas) - ALTURA_LINHA

    # margem direita da ultima coluna (regras/numeros terminam alinhados aqui,
    # como numa tabular do LaTeX)
    margem_direita = bordas_x[-1] - 0.08

    # cabecalho: 1a coluna a esquerda, demais alinhadas a direita (tabular padrao)
    for j, tit in enumerate(cfg["cabecalho"]):
        if j == 0:
            ax.text(bordas_x[j], y_cab_topo - ALTURA_LINHA / 2, tit,
                    ha="left", va="center", fontsize=15.5, color=PRETO)
        else:
            xpos = bordas_x[j + 1] - 0.08
            ax.text(xpos, y_cab_topo - ALTURA_LINHA / 2, tit,
                    ha="right", va="center", fontsize=15.5, color=PRETO)

    # linhas de dados (sem preenchimento, so texto)
    for i, linha in enumerate(linhas):
        yc = y_linha(i + 1) - ALTURA_LINHA / 2
        for j, valor in enumerate(linha):
            if j == 0:
                ax.text(bordas_x[j], yc, valor, ha="left", va="center",
                        fontsize=15, color=PRETO)
            else:
                destaque = (melhores[j] == i)
                xpos = bordas_x[j + 1] - 0.08
                ax.text(xpos, yc, valor, ha="right", va="center",
                        fontsize=15, color=PRETO,
                        fontweight="bold" if destaque else "normal")

    # regras horizontais estilo booktabs: \toprule, \midrule, \bottomrule
    ax.plot([0, n_cols], [y_regua_superior, y_regua_superior], color=PRETO, linewidth=1.4)
    ax.plot([0, n_cols], [y_regua_cabecalho, y_regua_cabecalho], color=PRETO, linewidth=0.7)
    ax.plot([0, n_cols], [y_regua_inferior, y_regua_inferior], color=PRETO, linewidth=1.4)

    fig.tight_layout(pad=0.3)
    fig.savefig(f"{PASTA}/{cfg['arquivo']}", dpi=220, bbox_inches="tight", facecolor="white")
    print("salvo", cfg["arquivo"])


gerar("pt")
gerar("en")
