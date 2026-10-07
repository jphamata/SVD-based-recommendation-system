defmodule Vapor.Embed do
  @moduledoc """
  Text embeddings from any supported decoder: the final normalised hidden
  state, pooled, as an L2-normalised binary32 vector.

  Decoder-based embedding models (Qwen3-Embedding, gte-Qwen2, e5-mistral)
  are exactly the architectures vapor runs; with `pooling: :last` (their
  convention, the last token after the EOS the tokenizer appends —
  `append_eos: true`) or `:mean`. The program is the model without its
  vocabulary head (built through the model airlock with `head: false,
  hidden: true` — any adapter whose spec lists the `:hidden` feature),
  so the hidden states are the certified program's bits: an embedding is the
  same vector on every substrate — which is what makes a vector index
  reproducible (`Vapor.RAG`). Pooling and normalisation happen in the BEAM
  in binary64 with `+ − × ÷ √` only (correctly rounded everywhere), then
  rounded once to binary32.
  """
  alias Vapor.{Tensor, Tokenizer}
  alias Vapor.Lock.Spec
  alias Vapor.Runtime.{Native, Substrates, Worker}

  defstruct [:cfg, :spec, :comp, :zeros, :worker, :tk, :max_seq, :pooling, :append_eos, :isa, :digest]

  @doc """
  Open an embedding model. Options: `:model` (path) or `:config` +
  `:weights` (+ `:tokenizer`), `:max_seq` (512), `:pooling` (`:last` |
  `:mean`, default `:last`), `:append_eos` (default `true` with `:last`),
  `:worker` (an existing `Vapor.Runtime.Worker`), `:isa`.
  """
  def open(opts) do
    with {:ok, cfg, ws, tk} <- source(opts),
         spec = Vapor.Lock.spec(cfg),
         :ok <- embeddable(spec) do
      s = Keyword.get(opts, :max_seq, min(spec.max_pos, 512))
      pooling = Keyword.get(opts, :pooling, :last)

      with {:ok, prog} <- Vapor.Lock.build(spec, ws, max_seq: s, hidden: true, head: false),
           {:ok, comp} <- Vapor.Compile.Lower.lower(prog),
           {:ok, w} <- worker(opts) do
        {:ok, %__MODULE__{cfg: cfg, spec: spec, comp: comp, zeros: Vapor.Lock.zero_state(prog), worker: w, tk: tk, max_seq: s,
                          pooling: pooling, append_eos: Keyword.get(opts, :append_eos, pooling == :last),
                          isa: Keyword.get(opts, :isa, Substrates.host_isa()),
                          digest: Vapor.Canonical.hex_digest({:embed, spec.digest, pooling, s})}}
      end
    end
  end

  defp embeddable(%Spec{interface: :causal_lm} = spec) do
    if Spec.supports?(spec, :hidden), do: :ok,
      else: {:error, Vapor.Rejection.new({:embed, spec.family}, "a builder with the :hidden feature", "use a decoder adapter that exposes hidden states")}
  end

  defp embeddable(spec), do: {:error, Vapor.Rejection.new({:embed, spec.family}, "the :causal_lm contract", "embed rows with Vapor.Modal")}

  defp source(opts) do
    case Keyword.fetch(opts, :model) do
      {:ok, path} -> with {:ok, m} <- Vapor.Lock.open(path), do: {:ok, m.spec, m.weights, m.tokenizer}
      :error -> {:ok, Keyword.fetch!(opts, :config), Keyword.fetch!(opts, :weights), Keyword.get(opts, :tokenizer)}
    end
  end

  defp worker(opts) do
    case Keyword.fetch(opts, :worker) do
      {:ok, w} -> {:ok, w}
      :error -> Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")])
    end
  end

  @doc "Embed texts (or token-id lists): `{:ok, [[float]]}`."
  def embed(%__MODULE__{} = e, inputs) when is_list(inputs) do
    Enum.reduce_while(inputs, {:ok, []}, fn input, {:ok, acc} ->
      case embed_one(e, input) do
        {:ok, v} -> {:cont, {:ok, [v | acc]}}
        err -> {:halt, err}
      end
    end)
    |> case do
      {:ok, vs} -> {:ok, Enum.reverse(vs)}
      err -> err
    end
  end

  @doc "Token ids for an input, as the embedder sees it (without a tokenizer: its bytes)."
  def ids(%__MODULE__{tk: nil} = e, text) when is_binary(text), do: text |> :binary.bin_to_list() |> Enum.take(-e.max_seq)

  def ids(%__MODULE__{} = e, text) when is_binary(text) do
    ids = Tokenizer.encode(e.tk, text, add_bos: false)
    ids = if e.append_eos and e.tk.eos, do: ids ++ [e.tk.eos], else: ids
    Enum.take(ids, -e.max_seq)
  end

  def ids(_e, ids) when is_list(ids), do: ids

  defp embed_one(e, input) do
    ids = ids(e, input)
    n = length(ids)

    if n == 0 do
      {:error, :empty_input}
    else
      env = Map.merge(e.zeros,
                      %{tok: Tensor.from_list(:s32, [n], ids), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))})

      with {:ok, got} <- Native.run(e.worker, e.comp, env, isa: e.isa, mode: :native) do
        rows = got.outputs.hidden |> Tensor.to_floats() |> Enum.chunk_every(e.spec.width)
        {:ok, pool(rows, e.pooling) |> normalize()}
      end
    end
  end

  defp pool(rows, :last), do: List.last(rows)

  defp pool(rows, :mean) do
    n = length(rows)
    rows |> Enum.zip_with(& &1) |> Enum.map(fn col -> Enum.reduce(col, 0.0, &(&2 + &1)) / n end)
  end

  @doc "L2 normalisation in binary64, then rounded once to binary32."
  def normalize(v) do
    norm = :math.sqrt(Enum.reduce(v, 0.0, fn x, s -> s + x * x end))
    Enum.map(v, fn x -> Vapor.CR.to_f32(if norm > 0, do: x / norm, else: 0.0) end)
  end
end
