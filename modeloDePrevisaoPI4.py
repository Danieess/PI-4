"""
PI IV - Previsão do consumo médio de energia
Regiões Metropolitanas de Sorocaba e Campinas

Objetivo:
- estimar consumo médio anual por consumidor (kWh/consumidor/ano)
- separar residencial e comercial
- gerar estimativas regionais para Sorocaba e Campinas
- usar OSN como variável de contexto/cenário, sem forçar uma relação causal
- estimar o custo aproximado do consumo usando a tarifa informada manualmente
  para cada região
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------
# 1. CONFIGURAÇÃO
# ------------------------------------------------------------

basePath = Path(".")
municipalFile = basePath / "consumo_energia_sp_2020_2024_municipios.csv"
osnFile = basePath / "OSN_CARGA_ENERGIA_LIMPO.csv"

outputDir = basePath / "saida_modelo"
outputDir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# COMPONENTES TARIFÁRIOS - PREENCHido PELO USUÁRIO
# ------------------------------------------------------------
# Campinas -> CPFL Paulista
# Sorocaba -> CPFL Piratininga
#
# Preencha SOMENTE os valores dos componentes abaixo.
# O programa calcula automaticamente kwMin e kwMax.
#
# CPFL Paulista
cpflPaulistaTUSDForaPonta = 164.16
cpflPaulistaTEForaPonta = 272.82

cpflPaulistaTUSDPonta = 1351.57
cpflPaulistaTEPonta = 431.88

# CPFL Piratininga
cpflPiratiningaTUSDForaPonta = 128.99
cpflPiratiningaTEForaPonta = 322.77

cpflPiratiningaTUSDPonta = 943.90
cpflPiratiningaTEPonta = 513.45


def calcularKwMin(tusdForaPonta, teForaPonta):
    if tusdForaPonta is None or teForaPonta is None:
        return None

    base = tusdForaPonta + teForaPonta
    return (base / 1000) * 0.3 + (base / 1000)


def calcularKwMax(tusdPonta, tePonta):
    if tusdPonta is None or tePonta is None:
        return None

    base = tusdPonta + tePonta
    return (base / 1000) * 0.3 + (base / 1000)


# Valores calculados automaticamente.
cpflPaulistaKwMin = calcularKwMin(
    cpflPaulistaTUSDForaPonta,
    cpflPaulistaTEForaPonta,
)
cpflPaulistaKwMax = calcularKwMax(
    cpflPaulistaTUSDPonta,
    cpflPaulistaTEPonta,
)

cpflPiratiningaKwMin = calcularKwMin(
    cpflPiratiningaTUSDForaPonta,
    cpflPiratiningaTEForaPonta,
)
cpflPiratiningaKwMax = calcularKwMax(
    cpflPiratiningaTUSDPonta,
    cpflPiratiningaTEPonta,
)

municipiosSorocaba = [
    "Alambari", "Alumínio", "Araçariguama", "Araçoiaba da Serra",
    "Boituva", "Capela do Alto", "Cerquilho", "Cesário Lange",
    "Ibiúna", "Iperó", "Itapetininga", "Itu", "Jumirim", "Mairinque",
    "Piedade", "Pilar do Sul", "Porto Feliz", "Salto",
    "Salto de Pirapora", "São Miguel Arcanjo", "São Roque", "Sarapuí",
    "Sorocaba", "Tapiraí", "Tatuí", "Tietê", "Votorantim"
]

municipiosCampinas = [
    "Americana", "Campinas", "Cosmópolis", "Elias Fausto", "Holambra",
    "Hortolândia", "Indaiatuba", "Jaguariúna", "Monte Mor",
    "Nova Odessa", "Paulínia", "Pedreira"
]

# Os nomes abaixo representam o esquema original do CSV de entrada.
# Internamente, eles serão convertidos para lower camel case.
perfis = {
    "residencial": ("num_cons_residencial", "kWh_residencial"),
    "comercial": ("num_cons_comercial", "kWh_comercial"),
    "rural": ("num_cons_rural", "kWh_rural"),
    "industrial": ("num_cons_industrial", "kWh_industrial"),
}

# Mapeamento das colunas externas do CSV para o padrão lower camel case.
colunasMunicipais = {
    "municipio": "municipio",
    "ano": "ano",
    "num_cons_residencial": "numConsResidencial",
    "kWh_residencial": "kwhResidencial",
    "num_cons_comercial": "numConsComercial",
    "kWh_comercial": "kwhComercial",
    "num_cons_rural": "numConsRural",
    "kWh_rural": "kwhRural",
    "num_cons_industrial": "numConsIndustrial",
    "kWh_industrial": "kwhIndustrial",
}

colunasOsn = {
    "ano": "ano",
    "nom_subsistema": "nomSubsistema",
    "val_cargaenergiamwmed": "valCargaEnergiaMwmed",
    "din_instante": "dinInstante",
}

# ------------------------------------------------------------
# 2. LEITURA E VALIDAÇÃO
# ------------------------------------------------------------

def lerDados():
    municipal = pd.read_csv(municipalFile, encoding="utf-8-sig")
    osn = pd.read_csv(osnFile, encoding="utf-8-sig")

    obrigatoriasMunicipal = {"municipio", "ano"}
    obrigatoriasOsn = {"ano", "nom_subsistema", "val_cargaenergiamwmed"}

    faltantes = obrigatoriasMunicipal - set(municipal.columns)
    if faltantes:
        raise ValueError(
            f"Colunas ausentes no CSV municipal: {sorted(faltantes)}"
        )

    faltantes = obrigatoriasOsn - set(osn.columns)
    if faltantes:
        raise ValueError(
            f"Colunas ausentes no CSV OSN: {sorted(faltantes)}"
        )

    municipal = municipal.rename(columns=colunasMunicipais)
    osn = osn.rename(columns=colunasOsn)

    municipal["ano"] = municipal["ano"].astype(int)
    osn["ano"] = osn["ano"].astype(int)

    return municipal, osn


def classificarRegiao(municipio):
    if municipio in municipiosSorocaba:
        return "Sorocaba"
    if municipio in municipiosCampinas:
        return "Campinas"
    return np.nan


def obterKwMinRegiao(regiao):
    if regiao == "Campinas":
        return cpflPaulistaKwMin
    if regiao == "Sorocaba":
        return cpflPiratiningaKwMin
    return np.nan


def obterKwMaxRegiao(regiao):
    if regiao == "Campinas":
        return cpflPaulistaKwMax
    if regiao == "Sorocaba":
        return cpflPiratiningaKwMax
    return np.nan


def obterDistribuidora(regiao):
    if regiao == "Campinas":
        return "CPFL Paulista"
    if regiao == "Sorocaba":
        return "CPFL Piratininga"
    return np.nan

# ------------------------------------------------------------
# 3. TRANSFORMAÇÃO WIDE -> LONG
# ------------------------------------------------------------

def prepararMunicipal(df):
    df = df.copy()
    df["regiao"] = df["municipio"].map(classificarRegiao)

    partes = []

    for perfil, (colNcOriginal, colKwhOriginal) in perfis.items():
        colNc = colunasMunicipais[colNcOriginal]
        colKwh = colunasMunicipais[colKwhOriginal]

        temp = df[
            ["municipio", "regiao", "ano", colNc, colKwh]
        ].copy()

        temp["perfil"] = perfil
        temp = temp.rename(
            columns={
                colNc: "numConsumidores",
                colKwh: "consumoKwh",
            }
        )

        temp["numConsumidores"] = pd.to_numeric(
            temp["numConsumidores"], errors="coerce"
        )
        temp["consumoKwh"] = pd.to_numeric(
            temp["consumoKwh"], errors="coerce"
        )

        temp["consumoMedioKwh"] = np.where(
            temp["numConsumidores"] > 0,
            temp["consumoKwh"] / temp["numConsumidores"],
            np.nan,
        )

        partes.append(temp)

    longDf = pd.concat(partes, ignore_index=True)
    longDf = longDf.dropna(subset=["regiao", "consumoMedioKwh"])
    longDf = longDf.sort_values(["perfil", "municipio", "ano"])

    grupo = longDf.groupby(["perfil", "municipio"], group_keys=False)

    longDf["lag1"] = grupo["consumoMedioKwh"].shift(1)
    longDf["lag2"] = grupo["consumoMedioKwh"].shift(2)
    longDf["media2Anos"] = grupo["consumoMedioKwh"].transform(
        lambda serie: serie.shift(1).rolling(2).mean()
    )

    longDf["crescimento1Ano"] = grupo["consumoMedioKwh"].pct_change()
    longDf["crescimento1Ano"] = longDf["crescimento1Ano"].replace(
        [np.inf, -np.inf], np.nan
    )

    return longDf

# ------------------------------------------------------------
# 4. OSN: CARGA DO SISTEMA COMO CONTEXTO
# ------------------------------------------------------------

def prepararOsn(osn):
    osn = osn.copy()
    osn["dinInstante"] = pd.to_datetime(
        osn["dinInstante"], errors="coerce"
    )
    osn["valCargaEnergiaMwmed"] = pd.to_numeric(
        osn["valCargaEnergiaMwmed"], errors="coerce"
    )

    diaria = (
        osn.dropna(subset=["dinInstante", "valCargaEnergiaMwmed"])
        .groupby("dinInstante", as_index=False)["valCargaEnergiaMwmed"]
        .sum()
        .rename(columns={"valCargaEnergiaMwmed": "cargaTotalMwmed"})
    )
    diaria["ano"] = diaria["dinInstante"].dt.year

    anual = (
        diaria.groupby("ano")["cargaTotalMwmed"]
        .agg(
            osnCargaMediaMwmed="mean",
            osnCargaMaxMwmed="max",
            osnCargaMinMwmed="min",
            osnCargaStdMwmed="std",
        )
        .reset_index()
    )

    anual["osnVariacaoAnual"] = anual["osnCargaMediaMwmed"].pct_change()

    if 2024 in set(anual["ano"]):
        base2024 = anual.loc[
            anual["ano"] == 2024, "osnCargaMediaMwmed"
        ].iloc[0]
        anual["osnIndice2024"] = (
            anual["osnCargaMediaMwmed"] / base2024
        ) * 100
    else:
        anual["osnIndice2024"] = np.nan

    return anual

# ------------------------------------------------------------
# 5. CONJUNTO SUPERVISIONADO
# ------------------------------------------------------------

featuresNumericas = [
    "ano",
    "numConsumidores",
    "lag1",
    "lag2",
    "media2Anos",
    "crescimento1Ano",
]

featuresCategoricas = [
    "regiao",
    "perfil",
]

target = "consumoMedioKwh"


def construirPipeline(modelo):
    preprocessador = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                featuresNumericas,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                featuresCategoricas,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessamento", preprocessador),
            ("modelo", modelo),
        ]
    )

# ------------------------------------------------------------
# 6. AVALIAÇÃO TEMPORAL
# ------------------------------------------------------------

def mapeSeguro(yTrue, yPred):
    yTrue = np.asarray(yTrue)
    yPred = np.asarray(yPred)
    mascara = yTrue != 0

    if not mascara.any():
        return np.nan

    return np.mean(
        np.abs((yTrue[mascara] - yPred[mascara]) / yTrue[mascara])
    ) * 100


def avaliarModelo(nome, pipeline, treino, teste):
    xTreino = treino[featuresNumericas + featuresCategoricas]
    yTreino = treino[target]

    xTeste = teste[featuresNumericas + featuresCategoricas]
    yTeste = teste[target]

    pipeline.fit(xTreino, yTreino)
    previsao = pipeline.predict(xTeste)

    resultado = {
        "modelo": nome,
        "mae": mean_absolute_error(yTeste, previsao),
        "rmse": np.sqrt(mean_squared_error(yTeste, previsao)),
        "r2": r2_score(yTeste, previsao),
        "mapePercentual": mapeSeguro(yTeste, previsao),
    }

    return pipeline, resultado, previsao


def executarValidacaoTemporal(longDf):
    resultados = []
    comparacoes = []

    for perfil in perfis:
        treino = longDf[
            (longDf["ano"] <= 2023) & (longDf["perfil"] == perfil)
        ].dropna(subset=["lag1", "lag2"]).copy()

        teste = longDf[
            (longDf["ano"] == 2024) & (longDf["perfil"] == perfil)
        ].dropna(subset=["lag1", "lag2"]).copy()

        modelos = {
            "baselineMedia": DummyRegressor(strategy="mean"),
            "randomForest": RandomForestRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
            "gradientBoosting": GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
            ),
        }

        for nome, modelo in modelos.items():
            pipeline = construirPipeline(modelo)
            pipeline, metricas, previsao = avaliarModelo(
                nome, pipeline, treino, teste
            )
            metricas["perfil"] = perfil
            resultados.append(metricas)

            if nome == "gradientBoosting":
                comparacao = teste[
                    ["municipio", "regiao", "ano", "perfil", target]
                ].copy()
                comparacao["previsao"] = previsao
                comparacao["erroAbsoluto"] = np.abs(
                    comparacao[target] - comparacao["previsao"]
                )
                comparacoes.append(comparacao)

    resultadosDf = pd.DataFrame(resultados)
    resultadosDf = resultadosDf[
        [
            "perfil",
            "modelo",
            "mae",
            "rmse",
            "r2",
            "mapePercentual",
        ]
    ]
    resultadosDf.to_csv(
        outputDir / "metricasModelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparacaoDf = pd.concat(comparacoes, ignore_index=True)
    comparacaoDf.to_csv(
        outputDir / "validacao2024.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return resultadosDf

# ------------------------------------------------------------
# 7. PREVISÃO DO PRÓXIMO ANO
# ------------------------------------------------------------

def prepararLinhaFutura(historico, anoFuturo):
    linhas = []

    for (perfil, municipio), grupo in historico.groupby(
        ["perfil", "municipio"]
    ):
        grupo = grupo.sort_values("ano")
        regiao = grupo["regiao"].iloc[-1]

        ultimo = grupo.iloc[-1]
        anterior = grupo.iloc[-2] if len(grupo) >= 2 else None

        lag1 = ultimo[target]
        lag2 = anterior[target] if anterior is not None else np.nan

        if anterior is not None:
            crescimento = (
                (ultimo[target] / anterior[target]) - 1
                if anterior[target] != 0
                else np.nan
            )
        else:
            crescimento = np.nan

        media2 = (
            grupo[target].tail(2).mean()
            if len(grupo) >= 2
            else ultimo[target]
        )

        linhas.append(
            {
                "municipio": municipio,
                "regiao": regiao,
                "ano": anoFuturo,
                "perfil": perfil,
                "numConsumidores": ultimo["numConsumidores"],
                "lag1": lag1,
                "lag2": lag2,
                "media2Anos": media2,
                "crescimento1Ano": crescimento,
            }
        )

    return pd.DataFrame(linhas)


def preverAnoFuturo(longDf, anoFuturo=2026):
    previsoes = []

    for perfil in perfis:
        modelo = construirPipeline(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
            )
        )

        treino = longDf[
            (longDf["ano"] <= 2024) & (longDf["perfil"] == perfil)
        ].dropna(subset=["lag1", "lag2"]).copy()

        xTreino = treino[featuresNumericas + featuresCategoricas]
        yTreino = treino[target]
        modelo.fit(xTreino, yTreino)

        futuro = prepararLinhaFutura(longDf, anoFuturo)
        futuro = futuro[futuro["perfil"] == perfil].copy()

        futuro["previsaoKwhConsumidorAno"] = modelo.predict(
            futuro[featuresNumericas + featuresCategoricas]
        )
        futuro["previsaoKwhConsumidorMes"] = (
            futuro["previsaoKwhConsumidorAno"] / 12
        )
        previsoes.append(futuro)

    futuroDf = pd.concat(previsoes, ignore_index=True)

    validacao = pd.read_csv(
        outputDir / "validacao2024.csv", encoding="utf-8-sig"
    )
    erroPorPerfil = (
        validacao.groupby("perfil")["erroAbsoluto"].mean().to_dict()
    )
    futuroDf["erroMedioValidacaoKwh"] = futuroDf["perfil"].map(erroPorPerfil)
    futuroDf["limiteInferiorReferencia"] = (
        futuroDf["previsaoKwhConsumidorAno"]
        - futuroDf["erroMedioValidacaoKwh"]
    ).clip(lower=0)
    futuroDf["limiteSuperiorReferencia"] = (
        futuroDf["previsaoKwhConsumidorAno"]
        + futuroDf["erroMedioValidacaoKwh"]
    )

    futuroDf.to_csv(
        outputDir / f"previsoesMunicipais{anoFuturo}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return futuroDf

# ------------------------------------------------------------
# 8. AGREGAR RESULTADO POR REGIÃO + CUSTO
# ------------------------------------------------------------

def agregarRegioes(futuroDf):
    def ponderada(grupo):
        pesos = grupo["numConsumidores"].clip(lower=0)
        valores = grupo["previsaoKwhConsumidorAno"]

        if pesos.sum() == 0:
            return valores.mean()

        return np.average(valores, weights=pesos)

    regionalDf = (
        futuroDf.groupby(["regiao", "perfil"])
        .apply(
            lambda grupo: pd.Series(
                {
                    "estimativaKwhConsumidorAno": ponderada(grupo),
                    "estimativaKwhConsumidorMes": ponderada(grupo) / 12,
                    "municipiosConsiderados": grupo["municipio"].nunique(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    regionalDf["distribuidora"] = regionalDf["regiao"].map(
        obterDistribuidora
    )

    # Valores de custo por kWh calculados a partir dos componentes tarifários.
    regionalDf["kwMin"] = regionalDf["regiao"].map(obterKwMinRegiao)
    regionalDf["kwMax"] = regionalDf["regiao"].map(obterKwMaxRegiao)

    # Custo estimado usando a faixa tarifária calculada.
    regionalDf["custoEstimadoAnualMin"] = (
        regionalDf["estimativaKwhConsumidorAno"]
        * regionalDf["kwMin"]
    )
    regionalDf["custoEstimadoAnualMax"] = (
        regionalDf["estimativaKwhConsumidorAno"]
        * regionalDf["kwMax"]
    )
    regionalDf["custoEstimadoMensalMin"] = (
        regionalDf["estimativaKwhConsumidorMes"]
        * regionalDf["kwMin"]
    )
    regionalDf["custoEstimadoMensalMax"] = (
        regionalDf["estimativaKwhConsumidorMes"]
        * regionalDf["kwMax"]
    )

    # Erro médio da validação, associado pela chave regiao + perfil.
    erroRegional = (
        futuroDf.groupby(["regiao", "perfil"], as_index=False)[
            "erroMedioValidacaoKwh"
        ]
        .first()
        .rename(columns={"erroMedioValidacaoKwh": "erroRegionalKwh"})
    )

    regionalDf = regionalDf.merge(
        erroRegional,
        on=["regiao", "perfil"],
        how="left",
    )

    regionalDf["limiteInferiorKwh"] = (
        regionalDf["estimativaKwhConsumidorAno"]
        - regionalDf["erroRegionalKwh"]
    ).clip(lower=0)
    regionalDf["limiteSuperiorKwh"] = (
        regionalDf["estimativaKwhConsumidorAno"]
        + regionalDf["erroRegionalKwh"]
    )

    # Faixa de custo de referência: menor consumo combinado com kwMin
    # e maior consumo combinado com kwMax.
    regionalDf["custoMinimoReferencia"] = (
        regionalDf["limiteInferiorKwh"] * regionalDf["kwMin"]
    )
    regionalDf["custoMaximoReferencia"] = (
        regionalDf["limiteSuperiorKwh"] * regionalDf["kwMax"]
    )

    regionalDf.to_csv(
        outputDir / "estimativaRegional2026.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return regionalDf

# ------------------------------------------------------------
# 9. CENÁRIO COM OSN
# ------------------------------------------------------------

def gerarCenarioOsn(osnAnual, regionalDf, anoReferencia=2024):
    osn = osnAnual.copy()

    base = osn.loc[
        osn["ano"] == anoReferencia, "osnCargaMediaMwmed"
    ]

    regionalDf = regionalDf.copy()

    if base.empty:
        regionalDf["osnIndiceCenario"] = np.nan
        return regionalDf

    base = base.iloc[0]

    osnAno = osn[["ano", "osnCargaMediaMwmed", "osnIndice2024"]]

    regionalDf["osnCargaBase2024Mwmed"] = base

    regionalDf.to_csv(
        outputDir / "estimativaRegionalComContextoOsn.csv",
        index=False,
        encoding="utf-8-sig",
    )

    osnAno.to_csv(
        outputDir / "contextoOsnAnual.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return regionalDf

# ------------------------------------------------------------
# 10. EXECUÇÃO
# ------------------------------------------------------------

def main():
    print("Lendo arquivos...")
    municipal, osn = lerDados()

    print("Preparando dados municipais...")
    longDf = prepararMunicipal(municipal)

    longDf.to_csv(
        outputDir / "baseModelagemLong.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Preparando dados OSN...")
    osnAnual = prepararOsn(osn)

    print("Validando modelos com 2024 como teste temporal...")
    metricas = executarValidacaoTemporal(longDf)

    print("\nMÉTRICAS")
    print(metricas.to_string(index=False))

    print("\nGerando previsão para 2026...")
    futuroDf = preverAnoFuturo(longDf, anoFuturo=2026)

    regionalDf = agregarRegioes(futuroDf)
    regionalDf = gerarCenarioOsn(osnAnual, regionalDf)

    print("\nESTIMATIVA REGIONAL")
    print(regionalDf.to_string(index=False))

    componentesTarifarios = [
        cpflPaulistaTUSDForaPonta,
        cpflPaulistaTEForaPonta,
        cpflPaulistaTUSDPonta,
        cpflPaulistaTEPonta,
        cpflPiratiningaTUSDForaPonta,
        cpflPiratiningaTEForaPonta,
        cpflPiratiningaTUSDPonta,
        cpflPiratiningaTEPonta,
    ]

    if any(valor is None for valor in componentesTarifarios):
        print(
            "\nATENÇÃO: preencha os componentes tarifários "
            "da CPFL Paulista e da CPFL Piratininga no início do script "
            "para calcular kwMin, kwMax e os custos estimados."
        )
    else:
        print("\nTARIFAS CALCULADAS")
        print(f"CPFL Paulista - kwMin: {cpflPaulistaKwMin:.6f}")
        print(f"CPFL Paulista - kwMax: {cpflPaulistaKwMax:.6f}")
        print(f"CPFL Piratininga - kwMin: {cpflPiratiningaKwMin:.6f}")
        print(f"CPFL Piratininga - kwMax: {cpflPiratiningaKwMax:.6f}")

    print("\nArquivos gerados em:", outputDir.resolve())


if __name__ == "__main__":
    main()
