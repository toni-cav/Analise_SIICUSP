# Predição de depressão com aprendizado de máquina — PNS 2019

Análise da Pesquisa Nacional de Saúde (PNS) 2019 para explorar indicadores de
depressão e testar modelos de classificação supervisionada capazes de prever
o diagnóstico profissional relatado (`Q092`).

## Estrutura do projeto

| Arquivo / pasta | Conteúdo |
|---|---|
| `exploratorio_depressao_e_sem_depressao.ipynb` | Notebook principal. Pipeline completo: preparação e imputação dos dados, comparação de modelos (Árvore de Decisão, Random Forest, Naive Bayes, SVM, XGBoost com Optuna), avaliação (F1, recall, precisão, AUC-ROC), análise de falsos negativos e explicabilidade com SHAP. |
| `analise_simples.ipynb` | Análise exploratória descritiva (não preditiva) da PNS 2019: perfil demográfico/socioeconômico, prevalência de diagnóstico e de PHQ-9 elevado por grupo, dados ausentes. |
| `preparar_dados.py` | Funções auxiliares para preparar variáveis descritivas a partir da base bruta, usadas pelo notebook principal. |
| `analise_resultados/` | Saídas finais geradas a partir do notebook principal: tabela comparativa dos modelos e gráficos SHAP (PT/EN), prontos para uso no resumo/pôster. |
| `pns2019.csv` | Base bruta da PNS 2019 (não versionada — ver `.gitignore`). Precisa ser baixada separadamente e colocada na raiz do projeto. |

## Como reproduzir

1. Coloque o arquivo `pns2019.csv` (microdados da PNS 2019) na raiz do projeto.
2. Crie o ambiente virtual e instale as dependências (`pandas`, `numpy`, `scikit-learn`,
   `xgboost`, `optuna`, `shap`, `matplotlib`).
3. Rode `exploratorio_depressao_e_sem_depressao.ipynb` do início ao fim.
4. Os gráficos e a tabela finais usados no resumo são gerados por
   `analise_resultados/gerar_tabela.py` e pelos scripts de SHAP referenciados
   no notebook, e salvos em `analise_resultados/`.

## Observações

- A análise usa apenas a PNS 2019. Uma comparação com a PNS 2013 foi descartada
  por não ser o foco do trabalho.
- Os pesos amostrais (`V00291`) são normalizados antes dos testes qui-quadrado
  para evitar inflar a significância estatística (ver comentários no notebook).
