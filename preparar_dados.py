import pandas as pd
import numpy as np


def criar_dados_descritivos(dados):
    dados_descritivos = pd.DataFrame(index=dados.index)
    variaveis_preparadas = {}

    def registrar(nome, bloco, origem):
        variaveis_preparadas[nome] = {
            "Bloco": bloco,
            "Origem": origem,
        }

    def sim_nao(serie):
        return serie.map({
            1: "Sim",
            2: "Não",
        })

    def codigos_como_texto(serie):
        resultado = pd.Series(
            pd.NA,
            index=serie.index,
            dtype="object",
        )

        validos = serie.notna()

        resultado.loc[validos] = (
            "Código "
            + serie.loc[validos].astype(int).astype(str)
        )

        return resultado

    # DIAGNÓSTICO
    dados_descritivos["Diagnóstico de depressão"] = dados["Q092"].map({
        1: "Com diagnóstico",
        2: "Sem diagnóstico",
    })
    
    dados_descritivos["Diagnóstico binário"] = dados["Q092"].map({
        1: 1,
        2: 0,
    })

    dados_descritivos["Peso"] = dados["V00291"]

    registrar(
        "Diagnóstico de depressão",
        "Diagnóstico",
        "Q092",
    )

    registrar(
        "Diagnóstico binário",
        "Diagnóstico",
        "Q092",
    )

    # DEMOGRÁFICAS E TERRITORIAIS
    dados_descritivos["Sexo"] = dados["C006"].map({
        1: "Homem",
        2: "Mulher",
    })
    registrar("Sexo", "Demográficas e territoriais", "C006")

    dados_descritivos["Faixa etária"] = pd.cut(
        dados["C008"],
        bins=[17, 29, 39, 49, 59, 69, np.inf],
        labels=[
            "18–29",
            "30–39",
            "40–49",
            "50–59",
            "60–69",
            "70+",
        ],
    )
    registrar("Faixa etária", "Demográficas e territoriais", "C008")

    dados_descritivos["Raça/cor"] = dados["C009"].map({
        1: "Branca",
        2: "Preta",
        3: "Amarela",
        4: "Parda",
        5: "Indígena",
    })
    registrar("Raça/cor", "Demográficas e territoriais", "C009")

    dados_descritivos["Vive com cônjuge"] = sim_nao(
        dados["C01001"]
    )
    registrar(
        "Vive com cônjuge",
        "Demográficas e territoriais",
        "C01001",
    )

    dados_descritivos["Estado civil"] = dados["C011"].map({
        1: "Casada(o)",
        2: "Divorciada(o) ou separada(o)",
        3: "Viúva(o)",
        4: "Solteira(o)",
    })
    registrar(
        "Estado civil",
        "Demográficas e territoriais",
        "C011",
    )

    dados_descritivos["Área"] = dados["V0026"].map({
        1: "Urbana",
        2: "Rural",
    })
    registrar("Área", "Demográficas e territoriais", "V0026")

    nomes_uf = {
        11: "Rondônia",
        12: "Acre",
        13: "Amazonas",
        14: "Roraima",
        15: "Pará",
        16: "Amapá",
        17: "Tocantins",
        21: "Maranhão",
        22: "Piauí",
        23: "Ceará",
        24: "Rio Grande do Norte",
        25: "Paraíba",
        26: "Pernambuco",
        27: "Alagoas",
        28: "Sergipe",
        29: "Bahia",
        31: "Minas Gerais",
        32: "Espírito Santo",
        33: "Rio de Janeiro",
        35: "São Paulo",
        41: "Paraná",
        42: "Santa Catarina",
        43: "Rio Grande do Sul",
        50: "Mato Grosso do Sul",
        51: "Mato Grosso",
        52: "Goiás",
        53: "Distrito Federal",
    }

    dados_descritivos["UF"] = dados["V0001"].map(nomes_uf)

    dados_descritivos["Região"] = pd.Series(
        pd.NA,
        index=dados.index,
        dtype="object",
    )

    dados_descritivos.loc[
        dados["V0001"].between(11, 17),
        "Região",
    ] = "Norte"

    dados_descritivos.loc[
        dados["V0001"].between(21, 29),
        "Região",
    ] = "Nordeste"

    dados_descritivos.loc[
        dados["V0001"].between(31, 35),
        "Região",
    ] = "Sudeste"

    dados_descritivos.loc[
        dados["V0001"].between(41, 43),
        "Região",
    ] = "Sul"

    dados_descritivos.loc[
        dados["V0001"].between(50, 53),
        "Região",
    ] = "Centro-Oeste"

    registrar("UF", "Demográficas e territoriais", "V0001")
    registrar("Região", "Demográficas e territoriais", "V0001")

    # SOCIOECONÔMICAS E TRABALHO
    dados_descritivos["Escolaridade"] = dados["VDD004A"].map({
        1: "Sem instrução / Fundamental incompleto",
        2: "Sem instrução / Fundamental incompleto",
        3: "Fundamental completo / Médio incompleto",
        4: "Fundamental completo / Médio incompleto",
        5: "Médio completo",
        6: "Superior",
        7: "Superior",
    })
    registrar(
        "Escolaridade",
        "Socioeconômicas e trabalho",
        "VDD004A",
    )

    dados_descritivos["Renda per capita"] = dados["VDF003"].where(
        dados["VDF003"] >= 0
    )
    registrar(
        "Renda per capita",
        "Socioeconômicas e trabalho",
        "VDF003",
    )

    filtro = dados_descritivos["Renda per capita"].notna()

    postos = dados_descritivos.loc[
        filtro,
        "Renda per capita",
    ].rank(method="first")

    dados_descritivos["Quintil de renda"] = pd.Series(
        pd.NA,
        index=dados.index,
        dtype="object",
    )

    dados_descritivos.loc[
        filtro,
        "Quintil de renda",
    ] = pd.qcut(
        postos,
        q=5,
        labels=[
            "Q1 — menor renda",
            "Q2",
            "Q3",
            "Q4",
            "Q5 — maior renda",
        ],
    ).astype(str)

    registrar(
        "Quintil de renda",
        "Socioeconômicas e trabalho",
        "VDF003",
    )

    dados_descritivos[
        "Faixa de renda — códigos"
    ] = codigos_como_texto(dados["VDF004"])

    registrar(
        "Faixa de renda — códigos",
        "Socioeconômicas e trabalho",
        "VDF004",
    )

    dados_descritivos["Força de trabalho"] = dados["VDE001"].map({
        1: "Na força de trabalho",
        2: "Fora da força de trabalho",
    })

    registrar(
        "Força de trabalho",
        "Socioeconômicas e trabalho",
        "VDE001",
    )

    dados_descritivos["Condição de ocupação"] = dados["VDE002"].map({
        1: "Ocupada",
        2: "Desocupada",
    })

    registrar(
        "Condição de ocupação",
        "Socioeconômicas e trabalho",
        "VDE002",
    )

    dados_descritivos[
        "Trabalhou em atividade remunerada"
    ] = sim_nao(dados["E001"])

    registrar(
        "Trabalhou em atividade remunerada",
        "Socioeconômicas e trabalho",
        "E001",
    )

    dados_descritivos["Faixa de horas trabalhadas"] = pd.cut(
        dados["E017"],
        bins=[-0.1, 19, 39, 44, 60, np.inf],
        labels=[
            "Até 19 h",
            "20–39 h",
            "40–44 h",
            "45–60 h",
            "Mais de 60 h",
        ],
    )

    registrar(
        "Faixa de horas trabalhadas",
        "Socioeconômicas e trabalho",
        "E017",
    )

    dados_descritivos["Faixa de horas domésticas"] = pd.cut(
        dados["E033"],
        bins=[-0.1, 4, 9, 19, 39, np.inf],
        labels=[
            "Até 4 h",
            "5–9 h",
            "10–19 h",
            "20–39 h",
            "40 h ou mais",
        ],
    )

    registrar(
        "Faixa de horas domésticas",
        "Socioeconômicas e trabalho",
        "E033",
    )

    # DOMICÍLIO
    dados_descritivos["Quantidade de banheiros"] = pd.cut(
        dados["A01401"],
        bins=[-0.1, 0, 1, 2, np.inf],
        labels=[
            "Nenhum",
            "1 banheiro",
            "2 banheiros",
            "3 ou mais",
        ],
    )

    registrar(
        "Quantidade de banheiros",
        "Domicílio",
        "A01401",
    )

    dados_descritivos[
        "Esgoto — códigos"
    ] = codigos_como_texto(dados["A01501"])

    registrar(
        "Esgoto — códigos",
        "Domicílio",
        "A01501",
    )

    dados_descritivos[
        "Destino do lixo — códigos"
    ] = codigos_como_texto(dados["A016010"])

    registrar(
        "Destino do lixo — códigos",
        "Domicílio",
        "A016010",
    )

    dados_descritivos[
        "Internet no domicílio"
    ] = sim_nao(dados["A01901"])

    registrar(
        "Internet no domicílio",
        "Domicílio",
        "A01901",
    )

    # APOIO SOCIAL
    apoio_quantidade = {
        "M01401": "Faixa de familiares com quem pode contar",
        "M01501": "Faixa de amigos com quem pode contar",
    }

    for original, nova in apoio_quantidade.items():
        dados_descritivos[nova] = pd.cut(
            dados[original],
            bins=[-0.1, 0, 1, 2, 4, np.inf],
            labels=[
                "Nenhum",
                "1 pessoa",
                "2 pessoas",
                "3–4 pessoas",
                "5 ou mais",
            ],
        )

        registrar(nova, "Apoio social", original)

    apoio_codigos = {
        "M01601": "Atividade em grupo — códigos",
        "M01701": "Participação comunitária — códigos",
        "M01801": "Trabalho voluntário — códigos",
        "M01901": "Atividade religiosa — códigos",
    }

    for original, nova in apoio_codigos.items():
        dados_descritivos[nova] = codigos_como_texto(
            dados[original]
        )

        registrar(nova, "Apoio social", original)

    # ACESSO À SAÚDE
    dados_descritivos[
        "Cadastro na Saúde da Família — códigos"
    ] = codigos_como_texto(dados["B001"])

    registrar(
        "Cadastro na Saúde da Família — códigos",
        "Acesso à saúde",
        "B001",
    )

    dados_descritivos[
        "Visita da Saúde da Família — códigos"
    ] = codigos_como_texto(dados["B003"])

    registrar(
        "Visita da Saúde da Família — códigos",
        "Acesso à saúde",
        "B003",
    )

    acesso_saude = {
        "I00102": "Possui plano de saúde",
        "J001": "Procura o mesmo serviço de saúde",
        "J037": "Internação nos últimos 12 meses",
    }

    for original, nova in acesso_saude.items():
        dados_descritivos[nova] = sim_nao(dados[original])
        registrar(nova, "Acesso à saúde", original)

    # ÁLCOOL, ATIVIDADE FÍSICA E TABACO
    dados_descritivos["Faixa de doses de álcool"] = pd.cut(
        dados["P029"],
        bins=[-0.1, 0, 2, 4, 7, np.inf],
        labels=[
            "Nenhuma",
            "1–2 doses",
            "3–4 doses",
            "5–7 doses",
            "8 ou mais",
        ],
    )

    registrar(
        "Faixa de doses de álcool",
        "Álcool, atividade física e tabaco",
        "P029",
    )

    dados_descritivos["Pratica exercício"] = sim_nao(
        dados["P034"]
    )

    registrar(
        "Pratica exercício",
        "Álcool, atividade física e tabaco",
        "P034",
    )

    dados_descritivos["Minutos de exercício por semana"] = (
        dados["P035"]
        * (dados["P036"] * 60 + dados["P03701"])
    )

    registrar(
        "Minutos de exercício por semana",
        "Álcool, atividade física e tabaco",
        "P035/P036/P03701",
    )

    dados_descritivos["Faixa de exercício semanal"] = pd.cut(
        dados_descritivos["Minutos de exercício por semana"],
        bins=[-0.1, 0, 74, 149, 299, np.inf],
        labels=[
            "Nenhum",
            "1–74 min",
            "75–149 min",
            "150–299 min",
            "300 min ou mais",
        ],
    )

    registrar(
        "Faixa de exercício semanal",
        "Álcool, atividade física e tabaco",
        "P035/P036/P03701",
    )

    dados_descritivos[
        "Uso de tabaco — códigos"
    ] = codigos_como_texto(dados["P050"])

    registrar(
        "Uso de tabaco — códigos",
        "Álcool, atividade física e tabaco",
        "P050",
    )

    # VIOLÊNCIA
    blocos_violencia = {
        "Violência psicológica": [
            "V00201",
            "V00202",
            "V00203",
            "V00204",
            "V00205",
        ],
        "Violência física": [
            "V01401",
            "V01402",
            "V01403",
            "V01404",
            "V01405",
        ],
        "Violência sexual": [
            "V02701",
            "V02702",
        ],
    }

    for tipo_violencia, colunas in blocos_violencia.items():
        for coluna in colunas:
            nome_item = f"{coluna} — item de {tipo_violencia.lower()}"

            dados_descritivos[nome_item] = sim_nao(
                dados[coluna]
            )

            registrar(
                nome_item,
                tipo_violencia,
                coluna,
            )

        teve_violencia = (
            dados[colunas] == 1
        ).any(axis=1)

        respondeu = dados[colunas].isin(
            [1, 2]
        ).any(axis=1)

        dados_descritivos[tipo_violencia] = pd.Series(
            pd.NA,
            index=dados.index,
            dtype="object",
        )

        dados_descritivos.loc[
            respondeu & teve_violencia,
            tipo_violencia,
        ] = "Sim"

        dados_descritivos.loc[
            respondeu & ~teve_violencia,
            tipo_violencia,
        ] = "Não"

        registrar(
            tipo_violencia,
            tipo_violencia,
            "Indicador derivado",
        )

    # SAÚDE E COMORBIDADES
    dados_descritivos["Estado de saúde"] = dados["N001"].map({
        1: "Muito bom",
        2: "Bom",
        3: "Regular",
        4: "Ruim",
        5: "Muito ruim",
    })

    registrar(
        "Estado de saúde",
        "Saúde geral e comorbidades",
        "N001",
    )

    comorbidades = {
        "Q00201": "Hipertensão",
        "Q03001": "Diabetes",
        "Q060": "Colesterol alto",
        "Q06306": "Doença do coração",
        "Q068": "AVC",
        "Q074": "Asma",
        "Q079": "Artrite ou reumatismo",
        "Q084": "Problema crônico de coluna",
        "Q088": "DORT ou doença do trabalho",
        "Q11006": "Outra doença mental",
    }

    for codigo, nome in comorbidades.items():
        dados_descritivos[nome] = sim_nao(dados[codigo])

        registrar(
            nome,
            "Saúde geral e comorbidades",
            codigo,
        )

    colunas_comorbidades = list(comorbidades.keys())

    respostas_validas_comorbidades = dados[
        colunas_comorbidades
    ].isin([1, 2]).any(axis=1)

    numero_comorbidades = (
        dados[colunas_comorbidades] == 1
    ).sum(axis=1)

    dados_descritivos["Número de comorbidades"] = (
        numero_comorbidades.where(
            respostas_validas_comorbidades
        )
    )

    registrar(
        "Número de comorbidades",
        "Saúde geral e comorbidades",
        "Indicador derivado",
    )

    dados_descritivos["Faixa de comorbidades"] = pd.cut(
        dados_descritivos["Número de comorbidades"],
        bins=[-1, 0, 1, 2, np.inf],
        labels=[
            "Nenhuma",
            "1 condição",
            "2 condições",
            "3 ou mais",
        ],
    )

    registrar(
        "Faixa de comorbidades",
        "Saúde geral e comorbidades",
        "Indicador derivado",
    )

    # Mantém somente adultos com resposta válida para Q092.
    dados_descritivos = dados_descritivos.dropna(
        subset=["Diagnóstico de depressão"]
    )

    return dados_descritivos, variaveis_preparadas
