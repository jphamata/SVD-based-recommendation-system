defmodule Vapor.SlidingWindowTest do
  @moduledoc """
  Sliding-window attention executed exactly (it used to be refused once the
  window could bind, which capped Mistral at its window and Gemma 3 at
  its local window). A row at position `p` attends to `max(0, p − w + 1) … p`
  in the canonical order — only the range changes — so a windowed row is
  bit for bit the unwindowed attention over the same keys moved to the
  front of a cache. Checked on every substrate, contiguous and paged;
  against transformers in `Vapor.FrontierHFTest` (mistral-window,
  gemma3-window), with the unwindowed program as the control that must
  fail there.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Model.{Config, Llama}
  alias Vapor.Runtime.{Dispatch, Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @h 4
  @hkv 2
  @dh 16
  @s 40
  @pos [0, 3, 17, 39, 77]

  defp prog(win) do
    q = T.input(:q, :f32, [T.dyn(:t, 8), @h * @dh])
    k = T.input(:k, :f32, [@s, @hkv * @dh])
    v = T.input(:v, :f32, [@s, @hkv * @dh])
    Program.new(o: T.attention(q, k, v, T.input(:pos, :s32, [T.dyn(:t, 8)]), @h, @hkv, nil, win))
  end

  defp env do
    %{q: Tensor.random(:f32, [length(@pos), @h * @dh], 1), k: Tensor.random(:f32, [@s, @hkv * @dh], 2),
      v: Tensor.random(:f32, [@s, @hkv * @dh], 3), pos: Tensor.from_list(:s32, [length(@pos)], @pos)}
  end

  test "a windowed row = unwindowed attention over its window moved to the front (oracle, bit for bit)" do
    e = env()
    rw = @hkv * @dh * 4

    for win <- [1, 5, 16, 17, 100] do
      %{o: o} = Oracle.eval_program(prog(win), e)

      shifted =
        for {p, i} <- Enum.with_index(@pos), into: <<>> do
          pc = min(p, @s - 1)
          s0 = max(0, pc - win + 1)
          front = fn t -> Tensor.new(:f32, [@s, @hkv * @dh], binary_part(t.data, s0 * rw, (@s - s0) * rw) <> :binary.copy(<<0>>, s0 * rw)) end
          one = %{q: Tensor.new(:f32, [1, @h * @dh], binary_part(e.q.data, i * @h * @dh * 4, @h * @dh * 4)), k: front.(e.k), v: front.(e.v),
                  pos: Tensor.from_list(:s32, [1], [pc - s0])}
          Oracle.eval_program(prog(nil), one).o.data
        end

      assert o.data == shifted, "window #{win}"
    end

    # a window at least the cache is no window at all
    assert Oracle.eval_program(prog(@s), e) == Oracle.eval_program(prog(nil), e)
  end

  @tag :native
  test "windowed attention: host ISAs, the poisoned RVV interpreter and the fabric = the oracle" do
    {:ok, w} = Worker.start_link(exec: worker_exec(:host), threads: 2)
    fabric = Enum.find(Substrates.list(), &(&1.kind == :fabric))
    e = env()

    for win <- [1, 5, 17] do
      {:ok, c} = Lower.lower(prog(win))
      {:ok, ref} = Native.run_oracle(c, e)
      for isa <- Substrates.host_isas(), do: assert(elem(Native.run(w, c, e, isa: isa, mode: :native), 1).outputs == ref.outputs)
      {:ok, emu} = Native.run(w, c, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 128)
      assert emu.outputs == ref.outputs
      if fabric, do: assert(elem(Dispatch.run_on(fabric, c, e, []), 1).outputs == ref.outputs)
    end
  end

  @tag :native
  test "a model whose window binds: paged = contiguous, prefill = cached decoding, and the window matters" do
    {:ok, c} = Config.from_map(tiny_config("mistral", %{"sliding_window" => 4}))
    ws = tiny_weights(c, 5)
    toks = [5, 17, 3, 90, 33, 2, 64, 8, 11, 70]
    n = length(toks)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    run = fn p, e -> {:ok, comp} = Lower.lower(p); {:ok, got} = Native.run(w, comp, e, isa: Substrates.host_isa(), mode: :native); got.outputs end
    base = %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(0..(n - 1)))}

    {:ok, contig} = Llama.program(c, ws, max_seq: 16)
    ref = run.(contig, Map.merge(Llama.empty_caches(c, 16), base))

    # paged: 2 sequences of 16 rows in pages of 4, this one in sequence 1
    {:ok, paged} = Llama.program(c, ws, max_seq: 16, kv: {:paged, 4, 8, 2})
    pool = Tensor.new(:f32, [32, c.kv_heads * c.head_dim], :binary.copy(<<0::32>>, 32 * c.kv_heads * c.head_dim))
    pe = Map.merge(base, %{table: Tensor.from_list(:s32, [2, 4], [0, 1, 2, 3, 7, 5, 6, 4]), slot: Tensor.from_list(:s32, [n], List.duplicate(1, n))})
    pe = Map.merge(pe, for(l <- 0..(c.layers - 1), name <- [:"k#{l}", :"v#{l}"], into: %{}, do: {name, pool}))
    assert run.(paged, pe).logits == ref.logits

    {rows, _} =
      Enum.map_reduce(Enum.with_index(toks), Llama.empty_caches(c, 16), fn {t, i}, caches ->
        out = run.(contig, Map.merge(caches, %{tok: Tensor.from_list(:s32, [1], [t]), pos: Tensor.from_list(:s32, [1], [i])}))
        {out.logits.data, Map.new(caches, fn {k, _} -> {k, out[:"#{k}_next"]} end)}
      end)

    assert IO.iodata_to_binary(rows) == ref.logits.data

    # control: dropping the window changes the rows past it, not those before
    {:ok, full} = Llama.program(%{c | sliding_window: nil}, ws, max_seq: 16)
    other = run.(full, Map.merge(Llama.empty_caches(c, 16), base))
    v = c.vocab
    assert binary_part(other.logits.data, 0, 4 * v * 4) == binary_part(ref.logits.data, 0, 4 * v * 4)
    refute binary_part(other.logits.data, 4 * v * 4, (n - 4) * v * 4) == binary_part(ref.logits.data, 4 * v * 4, (n - 4) * v * 4)
  end

  # prefill of `toks` in chunks of `n` through the paged program, the block
  # table mapping logical page j to physical page `ring.(j)`: all-row logits
  defp paged_prefill(c, ws, toks, n, page, ring) do
    s = 64
    mp = div(s, page)
    {:ok, p} = Llama.program(c, ws, max_seq: s, max_tokens: n, kv: {:paged, page, mp, 1})
    kvw = c.kv_heads * c.head_dim
    ids = &Tensor.from_list(:s32, [length(&1)], &1)
    table = Tensor.from_list(:s32, [1, mp], Enum.map(0..(mp - 1), ring))
    st0 = for {i, _} <- p.state, into: %{}, do: {i, Tensor.new(:f32, [s, kvw], :binary.copy(<<0::32>>, s * kvw))}

    toks
    |> Enum.with_index()
    |> Enum.chunk_every(n)
    |> Enum.map_reduce(st0, fn chunk, st ->
      env = Map.merge(st, %{tok: ids.(Enum.map(chunk, &elem(&1, 0))), pos: ids.(Enum.map(chunk, &elem(&1, 1))),
                            slot: ids.(Enum.map(chunk, fn _ -> 0 end)), table: table})
      out = Oracle.eval_program(p, env)
      {out.logits.data, for({i, o} <- p.state, into: %{}, do: {i, out[o]})}
    end)
    |> elem(0)
  end

  test "circular cache bound: a ring of ⌈(w + n − 1)/page⌉ pages gives the uncapped bits; one page fewer does not" do
    {:ok, c} = Config.from_map(tiny_config("mistral", %{"vocab_size" => 128, "sliding_window" => 8, "max_position_embeddings" => 64}))
    ws = tiny_weights(c, 5)
    toks = Enum.map(1..23, &rem(&1 * 21, 128))

    for {page, n} <- [{8, 9}, {4, 5}, {4, 1}] do
      r = div(8 + n - 1 + page - 1, page)
      full = paged_prefill(c, ws, toks, n, page, & &1)
      assert paged_prefill(c, ws, toks, n, page, &rem(&1, r)) == full, "page #{page}, chunk #{n}, ring #{r}"
      # the control: one page fewer overwrites a position some query still reads
      if r > 1, do: refute(paged_prefill(c, ws, toks, n, page, &rem(&1, r - 1)) == full, "page #{page}, chunk #{n}, ring #{r - 1}")
    end
  end
end
