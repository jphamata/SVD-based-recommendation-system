# Assay — a bancada de pesquisa em IA: sinal ou ruído

> Desde 0.14.0. Código: `lib/vapor/assay.ex`, `lib/vapor/assay/` (`stats`, `data`, `scaling`).
> Testes: `test/vapor/assay_test.exs`. Qualidade: `mix vapor.quality --only round14`.

O ensaio (*assay*) é a prova que diz quanto ouro há de fato no metal. A dor que ele ataca é a
mais comum na pesquisa e na indústria de IA: **uma diferença de avaliação que é ruído**, um
*leaderboard* que ordena empates, um benchmark contaminado, uma lei de escala extrapolada sem
checagem. Cada ferramenta recebe um CSV (ou texto) e responde com o número, o intervalo e a
frase que diz se é sinal.

| ferramenta | entrada | resposta | o controle |
|---|---|---|---|
| `compare` | colunas a, b por item | diferença pareada, IC *bootstrap*, permutação por troca de sinal (exata até n = 16), McNemar exato, d_z, **efeito mínimo detectável** e quantos itens seriam necessários | taxa de erro tipo I medida em 60 conjuntos nulos: 0,00 (≤ 0,15) |
| `leaderboard` | uma coluna por sistema | postos com *bootstrap* de postos, quem empata com o líder, Holm e BH | dois sistemas iguais não são separados |
| `calibration` | p, correct | ECE de massa igual, **o piso de ECE de um modelo perfeitamente calibrado**, p-valor, Brier, NLL, Platt ajustado numa metade e julgado na outra | o calibrado não é acusado (p = 0,45); o confiante demais é (p = 0,002) |
| `agreement` | uma coluna por anotador | α de Krippendorff (nominal, com faltantes), κ de Fleiss e de Cohen, IC | Krippendorff (2011): 0,743; rótulos ao acaso: α ≈ 0 |
| `judge` | ordem AB e BA | viés de posição de um juiz-LLM (binomial exato) | — |
| `contamination` | treino / teste | sobreposição de 13-gramas por item, o subconjunto limpo | — |
| `dedup` | documentos | MinHash LSH (128 hashes, 16 × 8 bandas) **verificado por Jaccard exato** | erro da estimativa |
| `scaling` | N, D, L | L = E + A/N^α + B/D^β (Hoffmann, abordagem 3: Huber em log, grade + Nelder–Mead), IC por *bootstrap*, **previsão dos maiores sem eles**, N\*(C) ótimo | perdas embaralhadas: a previsão erra 33 % e o certificado diz |

O controle da lei de escala achou um defeito real nesta rodada: num ajuste sem estrutura os
parâmetros em log subiam até estourar `exp`. Agora tudo fica finito e o *holdout* diz que a lei
não prevê nada — que é a resposta certa.

```
vapor assay compare resultados.csv          # sai 0 se a diferença é real, 1 se é ruído
vapor assay scaling corridas.csv --json | jq .holdout
vapor assay dedup corpus.jsonl --keep > limpo.jsonl
vapor assay contamination treino.txt teste.txt
```
