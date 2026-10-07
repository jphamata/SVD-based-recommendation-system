defmodule Vapor.Recurrent do
  @moduledoc """
  Generation for **recurrent** causal language models (specs with the
  `:recurrent` feature: `Vapor.Lock.Adapters.Mamba`) — models whose memory
  is a state of fixed size rather than a KV cache.

  The program is one step; a prompt is fed one token per step and the
  state stays in the worker's session between steps (`Vapor.Runtime.Session`
  copies `s ← s_next` inside the worker), so a step moves one id in and one
  row of logits out at any context length — memory and time per token are
  constant, which is the point of a state-space model. Without a native
  worker the oracle runs the same program (`worker: nil`).

  Decoding is greedy at temperature 0 and otherwise `Vapor.Sampler` — a pure
  function of the logits, the seed and the step, as in `Vapor.Engine`.
  """
  alias Vapor.{Sampler, Tensor}
  alias Vapor.Compile.Lower
  alias Vapor.Lock.Spec
  alias Vapor.Runtime.{Oracle, Session, Substrates}

  defstruct [:spec, :program, :session, :state, steps: 0]

  @doc """
  Open a generator. Options: `:worker` (a native `Vapor.Runtime.Worker`;
  nil runs the oracle), `:isa` (the host's).
  """
  def open(%Spec{} = spec, weights, opts \\ []) do
    with :ok <- recurrent(spec),
         {:ok, p} <- Vapor.Lock.build(spec, weights, []) do
      case Keyword.get(opts, :worker) do
        nil ->
          {:ok, %__MODULE__{spec: spec, program: p, state: spec.adapter.empty_state(spec.config)}}

        w ->
          with {:ok, comp} <- Lower.lower(p),
               {:ok, s} <- Session.open(w, comp, isa: Keyword.get(opts, :isa, Substrates.host_isa())) do
            {:ok, %__MODULE__{spec: spec, program: p, session: s}}
          end
      end
    end
  end

  defp recurrent(%Spec{} = spec) do
    if :recurrent in spec.features,
      do: :ok,
      else: {:error, Vapor.Rejection.new({:recurrent, spec.family}, "a spec with the :recurrent feature", "serve attention models with Vapor.Engine")}
  end

  @doc "Feed one token; returns `{logits_row_binary, generator}`."
  def feed(%__MODULE__{session: nil} = g, tok) do
    out = Oracle.eval_program(g.program, Map.put(g.state, :tok, Tensor.from_list(:s32, [1], [tok])))
    state = Map.new(g.program.state, fn {i, o} -> {i, out[o]} end)
    {out.logits.data, %{g | state: state, steps: g.steps + 1}}
  end

  def feed(%__MODULE__{session: s} = g, tok) do
    {:ok, %{logits: l}, _} = Session.step(s, %{tok: Tensor.from_list(:s32, [1], [tok])}, [:logits])
    {l.data, %{g | steps: g.steps + 1}}
  end

  @doc "Feed every token of `toks`; returns `{last_logits_row, generator, all_rows}`."
  def prefill(g, toks) do
    {rows, g} = Enum.map_reduce(toks, g, fn t, g -> feed(g, t) end)
    {List.last(rows), g, rows}
  end

  @doc """
  Generate `n` tokens after `prompt`. Options: those of `Vapor.Sampler.params/1`
  (`temperature`, `top_k`, `top_p`, `seed`), `:stop_ids`. Returns
  `{:ok, ids, generator}`.
  """
  def generate(%__MODULE__{} = g, prompt, n, opts \\ []) when prompt != [] do
    params = Sampler.params(opts)
    stop = Keyword.get(opts, :stop_ids, [])
    {row, g, _} = prefill(g, prompt)

    {ids, g} =
      Enum.reduce_while(0..(n - 1)//1, {[], row, g}, fn i, {acc, row, g} ->
        id = Sampler.sample(row, params, i)

        cond do
          id in stop -> {:halt, {acc, row, g}}
          i == n - 1 -> {:halt, {[id | acc], row, g}}
          true -> {row, g} = feed(g, id); {:cont, {[id | acc], row, g}}
        end
      end)
      |> then(fn {acc, _row, g} -> {Enum.reverse(acc), g} end)

    {:ok, ids, g}
  end

  @doc "Close the session (no-op on the oracle)."
  def close(%__MODULE__{session: nil}), do: :ok
  def close(%__MODULE__{session: s}), do: Session.close(s)
end
