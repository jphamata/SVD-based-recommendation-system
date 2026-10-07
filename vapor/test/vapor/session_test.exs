defmodule Vapor.SessionTest do
  @moduledoc """
  Phase P5, resident programs: weights mapped once, KV caches kept in the
  worker, per-step traffic of a few ids in and the requested rows out —
  with results bit-identical to one-shot execution.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Rejection, Tensor}
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Native, Session, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native

  defp model do
    {:ok, c} = Config.from_map(tiny_config("qwen2"))
    {:ok, p} = Llama.program(c, tiny_weights(c), max_seq: 16)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {c, comp}
  end

  defp ids(xs), do: Tensor.from_list(:s32, [length(xs)], xs)

  test "chunked prefill + single-token steps in a session = one-shot prefill, bit for bit" do
    {c, comp} = model()
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    toks = [3, 1, 4, 1, 5, 9, 2]
    n = length(toks)
    env = Map.merge(Llama.empty_caches(c, 16), %{tok: ids(toks), pos: ids(Enum.to_list(0..(n - 1)))})
    {:ok, ref} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native)

    {:ok, s} = Session.open(w, comp, isa: Substrates.host_isa())
    {:ok, %{logits: first}, _} = Session.step(s, %{tok: ids(Enum.take(toks, 3)), pos: ids([0, 1, 2])}, [:logits])

    rest =
      for i <- 3..(n - 1) do
        {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids([Enum.at(toks, i)]), pos: ids([i])}, [:logits])
        assert l.shape == [1, 96]
        l.data
      end

    got = Enum.map(0..2, &Tensor.row(first, &1)) ++ rest
    assert got == Enum.map(0..(n - 1), &Tensor.row(ref.outputs.logits, &1))

    # the caches live in the worker; asking for them returns the same state
    {:ok, %{k1_next: k1}, _} = Session.step(s, %{tok: ids([0]), pos: ids([99])}, [:k1_next])
    assert k1 == ref.outputs.k1_next

    # the RVV interpreter runs sessions too
    {:ok, e} = Session.open(w, comp, isa: :riscv64, mode: :emulate, vlen: 256, poison: true)
    {:ok, %{logits: le}, _} = Session.step(e, %{tok: ids(toks), pos: ids(Enum.to_list(0..(n - 1)))}, [:logits])
    assert le == ref.outputs.logits
  end

  test "a session is owned by its worker: reopening invalidates the old handle; dynamic inputs must be given" do
    {_c, comp} = model()
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, s1} = Session.open(w, comp, isa: Substrates.host_isa())
    {:ok, s2} = Session.open(w, comp, isa: Substrates.host_isa())
    assert {:error, :session_lost} = Session.step(s1, %{tok: ids([1]), pos: ids([0])}, [:logits])
    assert {:ok, _, _} = Session.step(s2, %{tok: ids([1]), pos: ids([0])}, [:logits])
    assert {:error, %Rejection{node: {:input, :pos}}} = Session.step(s2, %{tok: ids([1])}, [:logits])
    assert :ok == elem(Session.close(s2), 0) or match?({:ok, _}, Session.close(s2))
  end
end
