defmodule Vapor.Quality.Report do
  @moduledoc "The quality benchmark's report: Markdown for people, JSON for machines."
  alias Vapor.Quality.Suite

  @doc "A JSON-safe copy of a report (tuples → lists, non-finite → strings)."
  def plain(%{__struct__: _} = s), do: s |> Map.from_struct() |> plain()
  def plain(m) when is_map(m), do: Map.new(m, fn {k, v} -> {to_string(k), plain(v)} end)
  def plain(l) when is_list(l), do: Enum.map(l, &plain/1)
  def plain(t) when is_tuple(t), do: t |> Tuple.to_list() |> plain()
  def plain(b) when is_binary(b), do: if(String.valid?(b), do: b, else: Base.encode64(b))
  def plain(x), do: x

  def json(report), do: Vapor.JSON.encode(plain(report))

  @doc "The Markdown report."
  def markdown(r) do
    checks = Suite.checks(r)
    passed = Enum.count(checks, & &1.pass)

    """
    # Qualidade das saídas — vapor

    Gerado por `mix vapor.quality` (substrato: **#{r.substrate}**, #{r.timings_ms.total} ms no total).
    **#{passed} de #{length(checks)} verificações passaram.** Cada verificação compara uma saída
    com a verdade conhecida **e** com um controle que um gerador de ruído alcançaria — passar
    significa "sinal", e uma falha diz a distância. Metodologia: [docs/QUALIDADE.md](../QUALIDADE.md).

    ## 1. Os portões de ruído, calibrados antes de julgar

    Cada portão é ajustado a controles negativos (ruído, sinal embaralhado, repetição) e
    positivos (sinal real retido) e **recusa existir** se eles se sobrepõem. A margem é a
    folga entre o pior negativo e o pior positivo.

    | portão | escore | pior negativo | pior positivo | margem | controles |
    |---|---|---|---|---|---|
    #{Enum.map_join(r.gates, "\n", fn {name, g} -> "| #{name} | #{dir(g.direction)} | #{f(g.t_noise)} | #{f(g.t_natural)} | #{f(g.margin)} | #{controls(g.controls)} |" end)}

    ## 2. Texto: um modelo plantado pela pilha inteira

    Um bigrama contado na prosa em português de `docs/` embutido **exatamente** nos pesos de um
    decoder Llama (`Vapor.Quality.Planted`), admitido pela eclusa, compilado e executado.

    | medida | valor |
    |---|---|
    | max \\|logit − log P\\| contra a tabela analítica | #{e(r.text.max_table_error)} |
    | bits/caractere do modelo (retido) | #{f(r.text.bits_per_char)} |
    | bits/caractere da tabela | #{f(r.text.table_bits)} |
    | linha de base unigrama | #{f(r.text.unigram_bits)} |
    | linha de base uniforme | #{f(r.text.uniform_bits)} |
    | veredito do portão, gerações do modelo plantado | #{inspect(r.text.planted_verdicts)} |
    | veredito do portão, mesmo modelo com pesos aleatórios | #{inspect(r.text.random_verdicts)} |

    Amostra (temperatura 1): `#{String.replace(r.text.sample, "\n", "⏎")}`

    É texto com estatística de português, não português: o portão diz `:structured` (mais
    estrutura que todo controle de ruído, menos que todo texto real), e é isso que deve dizer.

    ## 3. Any-to-any: todas as rotas do hub, em entradas retidas

    Pares de cores nunca vistos em nenhum ajuste, variantes novas (jitter, iluminação, ruído de
    sensor), fases e amplitudes novas, ruído a 10 dB.

    | rota | o que mede | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|---|---|
    #{Enum.map_join(r.any_to_any, "\n", fn rt -> Enum.map_join(rt.checks, "\n", fn c -> "| #{rt.route} | #{rt.measures} | #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end) end)}

    ## 4. Fusão de modelos

    Um especialista em português (A) e um em inglês (B), bigramas plantados; bits/caractere em
    texto retido de cada domínio (menor é melhor).
    Especialistas: A@pt #{f(r.merge.specialists.pt.pt)}, A@en #{f(r.merge.specialists.pt.en)}, B@pt #{f(r.merge.specialists.en.pt)}, B@en #{f(r.merge.specialists.en.en)}.

    | método | pt | en | média | ms | recibo |
    |---|---|---|---|---|---|
    #{Enum.map_join(r.merge.methods, "\n", fn m -> "| #{m.method} | #{f(m.pt)} | #{f(m.en)} | #{f(m.mean)} | #{m.ms} | #{ok(m.receipt_ok)} |" end)}

    #{Enum.map_join(r.merge.checks, "\n", fn c -> "- #{ok(c.pass)} #{c.name}" end)}

    TIES e DARE pressupõem vetores de tarefa esparsos (diferenças pequenas de um fine-tune);
    as diferenças entre tabelas de log-probabilidade de dois domínios são densas, e a poda as
    distorce — medido aqui, não presumido.

    #{merge_real(Map.get(r, :merge_real))}

    #{real(Map.get(r, :real))}

    ## 5. Substratos

    #{if r.substrates.programs == [], do: "Sem worker nativo nesta execução (oráculo apenas).", else: Enum.map_join(r.substrates.programs, "\n", fn p -> "- #{ok(p.bit_identical)} #{p.program}: nativo = oráculo bit a bit" end)}

    #{round06(Map.get(r, :round06))}

    #{round07(Map.get(r, :round07))}

    #{round08(Map.get(r, :round08))}
    #{round09(Map.get(r, :round09))}
    #{round10(Map.get(r, :round10))}
    #{round11(Map.get(r, :round11))}
    #{round12(Map.get(r, :round12))}
    #{round13(Map.get(r, :round13))}
    #{round14(Map.get(r, :round14))}
    #{round15(Map.get(r, :round15))}

    ## 6. Custo da eclusa

    | adaptador | família | contrato | admitir (µs) | construir (µs) | reduzir (µs) | nós |
    |---|---|---|---|---|---|---|
    #{Enum.map_join(r.lock, "\n", fn l -> "| #{l.adapter} | #{l.family} | #{l.interface} | #{l.admit_us} | #{l.build_us} | #{l.lower_us} | #{l.nodes} |" end)}

    ## Tempos (ms)

    #{r.timings_ms |> Enum.sort() |> Enum.map_join(" · ", fn {k, t} -> "#{k} #{t}" end)}
    """
    |> String.replace(~r/\n {4}/, "\n")
  end

  defp round06(nil), do: ""

  defp round06(x) do
    """
    ## 5b. Rodada 0.6 — especialistas, cache latente, janelas, ÷, convolução, registro, estado, árvore

    Cada recurso da rodada 0.6 contra a sua verdade **e** contra um controle que mostra que a
    verificação discrimina (a forma ingênua, a forma errada, a entrada adulterada).

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round07(nil), do: ""

  defp round07(x) do
    """
    ## 5c. Rodada 0.7 — o escaneado de escritório, padrões na saída, fusão em disco

    O caminho inteiro de uma página escaneada (PDF → CCITT → ordem de leitura → leitor → modelo de
    língua), os formatos e padrões da saída restrita e a fusão em *streaming*, cada um contra o
    controle que a forma ingênua produziria.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round08(nil), do: ""

  defp round08(x) do
    """
    ## 5d. Rodada 0.8 — tabelas, JBIG2, GPU residente, especialistas esparsos, Mamba-2, dossiê, cluster

    Estrutura e texto de tabelas escaneadas (contra a leitura da 0.7 e a leitura livre), o decodificador
    JBIG2 contra o jbig2dec, a sessão residente na GPU contra a CPU, os especialistas de 4 bits
    predicados contra os densos, o Mamba-2 contra os logits do próprio transformers, o dossiê de
    auditoria contra adulteração e os *shards* entre nós BEAM contra um nó só.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round15(nil), do: ""

  defp round15(x) do
    """
    ## 5k. Rodada 0.15 — o Opus: Amálgama, Copela, Rebis, Aludel, Tábua

    Cada peça da rodada decide alguma coisa, então cada controle pergunta: *teria dito o mesmo se a
    afirmação fosse falsa?* A soma exata contra a soma da esquerda para a direita; o bit trocado
    contra o bit abaixo do envelope (dito, não fingido); somadores conformes que não podem ser
    acusados contra o falsificador que conhece a semente; o cavalo de Troia contra a simulação
    aleatória que não o vê; o multiplicador contra o produto parcial errado; o AES-GCM contra o
    OpenSSL e a etiqueta adulterada; o GHZ contra Hadamards independentes; Motzkin + 1/1000 contra
    Motzkin que toca o zero; a barreira contra o campo instável; o contrato contra as suas
    precedências; a fusão alinhada contra a ingênua.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round14(nil), do: ""

  defp round14(x) do
    """
    ## 5j. Rodada 0.14 — a bancada aberta: Alembic, Athanor, Touchstone, Crucible, Assay

    A bancada aceita entrada arbitrária, então cada controle faz a pergunta que importa para entrada
    aberta: *a mesma resposta teria saído se não houvesse nada a encontrar?* A fornalha contra a
    busca aleatória com o mesmo orçamento; a enumeração contra uma força bruta independente e um
    certificado forjado; a prova de R(3,3) contra o pentágono; o momento plantado contra o passeio
    aleatório no conjunto de validação; a mesma semente contra outra; a bomba de memória contra um
    programa pequeno; hamiltonianos aleatórios redescobertos contra sistemas dissipativos sem lei;
    o oscilador contra a caixa pequena demais; Yoshida contra RK4; o H₂ contra o seu estiramento
    (a falha conhecida do RHF, exibida); a filogenia e a regressão contra o embaralhado; o teste
    pareado no nulo contra o efeito plantado; o modelo calibrado contra o confiante demais; o α de
    Krippendorff contra rótulos ao acaso; a lei de escala contra perdas embaralhadas (que expôs um
    estouro de `exp`, corrigido); o jogo resolvido contra o acaso; o ruído do navegador contra o do
    servidor.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round13(nil), do: ""

  defp round13(x) do
    """
    ## 5i. Rodada 0.13 — finanças, a mesa de operações e as pendências fechadas

    Cada resposta contra a sua referência e contra o seu controle: o calendário pelo cômputo contra
    só os feriados fixos; o rateio exato contra o arredondamento binário que perde um centavo; o DI1
    em dias úteis contra os dias corridos; a paridade e as gregas contra um preço abaixo do
    intrínseco; a americana contra a europeia; o sorriso calmo contra a fatia de Vogt; os bits do
    oráculo e de duas threads contra o termo de Itô esquecido; o VaR histórico contra o normal em
    caudas grossas (e o tamanho do teste); o sinal plantado contra o melhor de trinta ruídos e a
    espiada no amanhã; o certificado de Farkas contra a proposta errada; a borboleta que paga contra
    os preços de estado; o motor contra o diário forjado; a recusa com causa contra a recusa sem
    causa; o Hawkes contra o Poisson; a sessão contra outra semente; a assinatura contra o
    manifesto reescrito; a leitura afim contra a unidade composta.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round12(nil), do: ""

  defp round12(x) do
    """
    ## 5h. Rodada 0.12 — bancada, engenharia, lógica, tabuleiros, proteínas, render, direção de cena

    Cada solucionador contra a forma fechada, o valor publicado ou o oráculo, e contra o seu controle:
    Dormand–Prince contra RK4 de passo fixo; Crank–Nicolson (ordem 2) contra Euler implícito (ordem 1);
    o sistema com unidades coerentes contra o mesmo com uma força no lugar de uma velocidade; a lata
    ótima com limites contra a divergência sem eles; trapézios contra Euler no RC; Newton contra
    Gauss–Seidel; dez elementos contra um; QM6 contra o Q4 que trava; invariantes contra uma espécie
    que não se conserva; Fenske contra o refluxo abaixo do mínimo; a prova DRUP inteira contra a
    truncada; Knuth–Bendix contra os axiomas só orientados; Tales contra a variante falsa; a proposta
    certa contra a trocada; perft contra os números publicados; o CFR+ contra o jogo uniforme; o
    alinhamento planejado contra o embaralhado; a fornalha contra o estimador viciado; e a direção
    com nomes contra a frase sem ninguém.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round11(nil), do: ""

  defp round11(x) do
    """
    ## 5g. Rodada 0.11 — descoberta, matemática, ciência, autojogo, cena viva, esboço, arquivos

    Redes de ordenação contra redes sorteadas e podadas; o truque de bits contra a fórmula ingênua
    que estoura; o posto 7 contra o posto 6 impossível; teoremas contra seus gêmeos falsos;
    conjecturas contra as trincas triviais; a homologia sobre ℚ contra a de GF(2); o laço contra a
    mancha; cada experimento de ciência contra a sua forma fechada ou o valor publicado e o seu
    controle; o agente treinado contra a mesma busca sem treino; a política de muitos mundos contra a
    de um; a cena, o esqueleto, a direção, o esboço e a planta contra a verdade com que foram
    desenhados; e o arquivo contra um byte trocado.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round10(nil), do: ""

  defp round10(x) do
    """
    ## 5f. Rodada 0.10 — eclusa de substratos, treino, contexto sem fim, física, redes, figuras, fórmulas, CJK, árabe

    A eclusa contra um dispositivo que arredonda operandos a bf16; o modelo que o vapor treinou contra
    Witten–Bell e contra o próprio texto embaralhado; o fluxo sem fim contra posições crescentes; o
    simulador contra o período exato do pêndulo e contra si mesmo em outro substrato; a identificação
    do gêmeo contra medidas embaralhadas no tempo; a política contra a política nula; o alarme contra
    a ausência de falha; as redes contra seus nulos; os gráficos contra rótulos permutados (que devem
    ser recusados); as fórmulas contra a leitura plana; o CJK contra caracteres aleatórios; o árabe
    contra o leitor latino.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp round09(nil), do: ""

  defp round09(x) do
    """
    ## 5e. Rodada 0.9 — estúdio: difusão, determinismo e cache, ampliação, RL, 3D, áudio, MCP

    O pipeline de Stable Diffusion contra o do próprio diffusers (checkpoint minúsculo incluído), o
    estúdio contra si mesmo (raiz de Merkle estável, cache que só recalcula o que mudou), o ampliador
    consistente contra Lanczos com a mesma projeção, as políticas contra controles, a malha contra a
    esfera analítica, a reamostragem contra a decimação e o servidor MCP contra uma raiz falsa.

    | verificação | valor | controle | limiar | ok |
    |---|---|---|---|---|
    #{Enum.map_join(x.checks, "\n", fn c -> "| #{c.name} | #{v(c.value)} | #{v(c.control)} | #{c.threshold} | #{ok(c.pass)} |" end)}
    """
  end

  defp real(nil), do: ""

  defp real(x) do
    o = x.ocr
    tess = fn nil -> "—"; v -> f(v) end

    """
    ## 4c. Dados reais — leitura, fala, caligrafia

    Modelos admitidos pela eclusa (`priv/ocr`, `priv/speech`, `priv/digits`), dados nunca vistos no treino.

    | rota | medida | vapor | controle | Tesseract |
    |---|---|---|---|---|
    | OCR, 5 fontes fora do treino (#{o.lines} linhas) | CER | #{f(o.cer)} | #{f(o.control_cer)} (texto fluente errado) | #{tess.(o.tesseract_cer)} |
    | OCR, foto real de página | CER | #{f(o.page.cer)} | — | #{tess.(o.page.tesseract_cer)} |
    | fala, voz nunca ouvida (#{x.speech.clips} gravações) | acerto | #{f(x.speech.accuracy)} | 0.100 (acaso); invertida no tempo: #{f(x.speech.reversed_accuracy)} | — |
    | caligrafia → dígito (#{x.digits.held_out} retidos) | acerto | #{f(x.digits.accuracy)} | 0.100 | — |
    | dígito → caligrafia (#{x.digits.generated} gerados), lidos de volta | acerto | #{f(x.digits.read_back_accuracy)} | 0.100 | — |
    | distância ao treino mais próximo (mediana) | níveis de cinza | #{f(x.digits.nearest_train.generated)} | reais retidos #{f(x.digits.nearest_train.held_out_real)}; memorização #{f(x.digits.nearest_train.memorising_control)} | — |
    | voz → texto → desenho → leitura | acerto | #{f(x.chain.accuracy)} | 0.100 | — |

    CER por fonte: #{o.per_font |> Enum.sort() |> Enum.map_join(", ", fn {k, v} -> "#{k} #{f(v)}" end)}.

    Fala: a voz retida (escolhida antes do treino) é a mais fácil das seis do conjunto; retendo cada voz por vez (`test/python/train_speech.py --loso`) o acerto médio é 0,72 ([ANY_TO_ANY.md §6](../ANY_TO_ANY.md)). Inverter a gravação no tempo quase não muda a leitura: um dígito falado se reconhece pelo timbre das vogais — por isso o controle é o acaso.

    Página real lida pelo vapor:

    ```
    #{o.page.text}
    ```

    #{Enum.map_join(x.checks, "\n", fn c -> "- #{ok(c.pass)} #{c.name}" end)}
    """
  end

  defp merge_real(nil), do: ""

  defp merge_real(m) do
    sec = fn title, part ->
      """
      **#{title}** — regime diagnosticado: `#{part.diag.regime}`; escolhido na validação: **#{part.chosen}**.

      #{Enum.map_join(part.diag.advice, "\n", &("> " <> &1 <> "  "))}

      | método | validação | pt (teste) | en (teste) | média (teste) | ms |
      |---|---|---|---|---|---|
      #{Enum.map_join(part.rows, "\n", fn x -> "| #{x.method} | #{f(x.val)} | #{f(x.pt)} | #{f(x.en)} | #{f(x.mean)} | #{x.ms} |" end)}
      """
    end

    """
    ### 4b. Fusão de transformers treinados (não plantados)

    Decoders Llama de caracteres treinados pelo PyTorch (`priv/quality/merge`,
    `test/python/train_merge_models.py`). Bits/caractere, teste disjunto da validação.

    | modelo | pt | en | média |
    |---|---|---|---|
    #{m.models |> Enum.sort() |> Enum.map_join("\n", fn {k, s} -> "| #{k} | #{f(s.pt)} | #{f(s.en)} | #{f(s.mean)} |" end)}

    #{sec.("Fine-tunes de uma base (ft_pt + ft_en)", m.fine_tune)}

    #{sec.("Treinados separadamente (solo_pt + solo_en, sem base comum)", m.independent)}

    #{Enum.map_join(m.checks, "\n", fn c -> "- #{ok(c.pass)} #{c.name}" end)}
    """
  end

  defp ok(true), do: "✅"
  defp ok(false), do: "❌"
  defp dir(:up), do: "↑ mais estrutura"
  defp dir(:down), do: "↓ mais estrutura"
  defp f(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 3)
  defp f(x), do: inspect(x)
  defp e(x) when is_float(x), do: :io_lib.format("~.2e", [x]) |> to_string()
  # tiny nonzero measurements in scientific notation (0.000 would hide them)
  defp v(x) when is_float(x) and x != 0.0 and abs(x) < 1.0e-3, do: e(x)
  # and huge ones (an integrator that blew up) too
  defp v(x) when is_float(x) and abs(x) >= 1.0e9, do: e(x)
  defp v(x) when is_float(x), do: f(x)
  defp v(x) when is_binary(x), do: x
  defp v(nil), do: "—"
  defp v(x) when is_list(x), do: x |> Enum.frequencies() |> Enum.map_join(", ", fn {k, n} -> "#{n}× #{k}" end)
  defp v(x) when is_tuple(x), do: x |> Tuple.to_list() |> Enum.map_join(" / ", &v/1)
  defp v(x), do: inspect(x)

  defp controls(cs), do: Enum.map_join(cs, "; ", fn {k, s} -> "#{k}: #{f(s[:min])}…#{f(s[:max])}" end)
end
