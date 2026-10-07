defmodule Vapor.Speculative.Tree do
  @moduledoc """
  **Tree** speculative decoding over a paged KV cache: several draft
  continuations (branches) verified by the target in **one** step, each in
  its own slot, all of them reading the accepted context's pages — no copy.

  The construction:

    * the accepted context lives in a list of pages (the *main* pages);
    * branch 0 writes straight into them (a rejected row is stale beyond
      the accepted length and is overwritten before anything reads it — the
      linear `Vapor.Speculative` argument);
    * branch `b ≥ 1` is a slot whose block table lists the main pages up to
      the last full page (shared, read-only: no branch writes there) and
      then pages of its own; it re-feeds the tokens of the partial last
      page (fewer than `page`) before its own tokens, so every position it
      reads is either shared or recomputed by itself;
    * the target's logits for a row depend only on that row's token,
      position and the KV it reads — batch- and slot-invariance of every
      kernel — so each branch's agreement with the target's greedy choice
      is exactly what sequential decoding would decide;
    * the longest-agreeing branch wins; if it is a fork, its pages simply
      *become* the main pages after the shared ones (ownership moves, no
      copy); the rest return to the pool.

  So the output is the target's greedy decoding token for token, whatever
  the drafts (tested); drafts change only how many target steps it takes.

  The default draft is **prompt lookup**: the continuations that followed
  earlier occurrences of the context's last n-gram (n = 3, 2, 1), the most
  recent first, one branch per distinct next token, copied with overlap
  (a loop the text has entered is proposed for the whole depth) — no draft model, and
  strongest exactly where generation copies its input (retrieval-augmented
  answers, code edits, summaries that quote). Any function
  `draft.(context, branches, depth) → [[token]]` can replace it (a small
  model, a grammar's forced tokens…).
  """
  alias Vapor.{Sampler, Tensor}
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.{Session, Substrates}

  defstruct [:session, :vocab, :page, :pages, :ns, :mp, :tmax]

  @doc """
  Open a target for tree verification on `worker`. Options: `:max_seq`
  (512), `:page` (8), `:branches` (slots, 4), `:max_tokens` (rows per step,
  64), `:pages` (default `branches · max_seq / page`), `:isa`.
  """
  def open(spec_or_config, weights, worker, opts \\ []) do
    spec = Vapor.Lock.spec(spec_or_config)
    s = Keyword.get(opts, :max_seq, 512)
    page = Keyword.get(opts, :page, 8)
    ns = Keyword.get(opts, :branches, 4)
    tmax = Keyword.get(opts, :max_tokens, 64)
    pages = Keyword.get(opts, :pages, div(ns * s, page))

    with {:ok, p} <- Vapor.Lock.build(spec, weights, max_seq: s, max_tokens: tmax, kv: {:paged, page, pages, ns}),
         {:ok, comp} <- Lower.lower(p),
         {:ok, session} <- Session.open(worker, comp, isa: Keyword.get(opts, :isa, Substrates.host_isa())) do
      {:ok, %__MODULE__{session: session, vocab: spec.vocab, page: page, pages: pages, ns: ns, mp: div(s, page), tmax: tmax}}
    end
  end

  @doc """
  Greedy-decode `n` tokens after `prompt`. Options: `:branches` (≤ the
  slots), `:depth` (draft tokens per branch, 4), `:draft` (`:lookup` or a
  function), `:eos` (ids). Returns `{ids, stats}` with `target_steps`,
  `proposed`, `accepted`, `rows` (rows the target computed, re-fed ones
  included) and `fork_wins` (rounds a forked branch won).
  """
  def generate(%__MODULE__{} = tr, prompt, n, opts \\ []) when prompt != [] do
    if length(prompt) + n > tr.mp * tr.page,
      do: raise(ArgumentError, "prompt + #{n} tokens exceed the context of #{tr.mp * tr.page}")

    b = min(Keyword.get(opts, :branches, tr.ns), tr.ns)
    k = Keyword.get(opts, :depth, 4)
    draft = case Keyword.get(opts, :draft, :lookup) do
      :lookup -> &lookup/3
      f when is_function(f, 3) -> f
    end

    eos = Keyword.get(opts, :eos, [])
    free = Enum.to_list(0..(tr.pages - 1))

    # prefill the prompt into the main pages (in chunks of at most tmax rows)
    {main, free} = Enum.split(free, pages_for(length(prompt), tr.page))
    rows_of = fn toks, p0 -> Enum.with_index(toks, p0) |> Enum.map(fn {t, p} -> {t, p, 0} end) end

    last =
      prompt
      |> Enum.chunk_every(tr.tmax)
      |> Enum.reduce({0, nil}, fn chunk, {p0, _} -> {p0 + length(chunk), List.last(step!(tr, rows_of.(chunk, p0), %{0 => main}))} end)
      |> elem(1)

    first = Sampler.argmax(last)
    st = %{target_steps: div(length(prompt) + tr.tmax - 1, tr.tmax), proposed: 0, accepted: 0, rows: length(prompt), fork_wins: 0}
    round(tr, prompt ++ [first], length(prompt), main, free, [first], n, b, k, draft, eos, st)
  end

  # ctx: every token so far; l: positions in the main pages (ctx minus its
  # last, pending token); out: generated tokens
  defp round(_tr, _ctx, _l, _main, _free, out, n, _b, _k, _draft, _eos, st) when length(out) >= n,
    do: {Enum.take(out, n), st}

  defp round(tr, ctx, l, main, free, out, n, b, k, draft, eos, st) do
    pending = List.last(ctx)

    if pending in eos do
      {out, st}
    else
      boundary = div(l, tr.page) * tr.page
      tail = Enum.slice(ctx, boundary, l - boundary)
      depth = min(k, n - length(out) - 1)

      # branches within the step's row budget: branch 0 costs 1 + d rows,
      # a fork |tail| + 1 + d
      branches =
        ctx
        |> draft.(b, max(depth, 0))
        |> Enum.map(&Enum.take(&1, max(depth, 0)))
        |> Enum.uniq()
        |> then(&if(&1 == [], do: [[]], else: &1))
        |> fit(length(tail), tr.tmax, b)

      # pages: branch 0 extends the main pages; a fork owns pages from the
      # boundary on — a fork the pool cannot hold is dropped (never run
      # with a short table: its writes would be skipped)
      need0 = max(pages_for(l + length(hd(branches)) + 1, tr.page) - length(main), 0)
      if need0 > length(free), do: raise(ArgumentError, "the page pool (#{tr.pages}) cannot hold the context: open with more :pages")
      {extra, free} = Enum.split(free, need0)
      shared = Enum.take(main, div(boundary, tr.page))

      {forks, free} =
        branches
        |> tl()
        |> Enum.reduce({[], free}, fn br, {acc, free} ->
          need = pages_for(l + length(br) + 1, tr.page) - length(shared)
          if need <= length(free), do: (fn {own, rest} -> {acc ++ [{br, shared ++ own}], rest} end).(Enum.split(free, need)), else: {acc, free}
        end)

      branches = [hd(branches) | Enum.map(forks, &elem(&1, 0))]
      tables = Map.new(Enum.with_index([main ++ extra | Enum.map(forks, &elem(&1, 1))], fn t, i -> {i, t} end))

      rows =
        branches
        |> Enum.with_index()
        |> Enum.flat_map(fn {br, i} ->
          toks = if i == 0, do: [pending | br], else: tail ++ [pending | br]
          p0 = if i == 0, do: l, else: boundary
          Enum.with_index(toks, p0) |> Enum.map(fn {t, p} -> {t, p, i} end)
        end)

      logits = step!(tr, rows, tables)

      # each branch's verdict, from its own rows (from the pending token on)
      {_, verdicts} =
        branches
        |> Enum.with_index()
        |> Enum.reduce({0, []}, fn {br, i}, {at, acc} ->
          skip = if i == 0, do: 0, else: length(tail)
          mine = Enum.slice(logits, at + skip, length(br) + 1)
          greedy = Enum.map(mine, &Sampler.argmax/1)
          agree = Enum.zip(br, greedy) |> Enum.take_while(fn {a, g} -> a == g end) |> length()
          kept = Enum.take(br, agree) ++ [Enum.at(greedy, agree)]
          {at + skip + length(br) + 1, acc ++ [{i, agree, kept}]}
        end)

      {win, agree, kept} = Enum.max_by(verdicts, fn {i, a, _} -> {a, -i} end)
      kept = cut_eos(kept, eos)
      l2 = l + 1 + agree
      keep = pages_for(l2 + 1, tr.page)

      # the winner's pages become the main pages; everything else is freed
      chosen = tables[win]
      {main2, spare} = Enum.split(chosen, keep)
      released = (Map.values(tables) |> List.flatten() |> Enum.uniq()) -- main2
      free = Enum.uniq(spare ++ released ++ free) -- main2

      st = %{st | target_steps: st.target_steps + 1, proposed: st.proposed + Enum.sum(Enum.map(branches, &length/1)),
                  accepted: st.accepted + agree, rows: st.rows + length(rows), fork_wins: st.fork_wins + if(win > 0, do: 1, else: 0)}
      round(tr, ctx ++ kept, l2, main2, free, out ++ kept, n, b, k, draft, eos, st)
    end
  end

  # as many branches as the row budget allows (branch 0 always)
  defp fit([b0 | rest], tail, tmax, b) do
    {kept, _} =
      Enum.reduce(Enum.take(rest, b - 1), {[b0], tmax - 1 - length(b0)}, fn br, {acc, left} ->
        cost = tail + 1 + length(br)
        if cost <= left, do: {acc ++ [br], left - cost}, else: {acc, left}
      end)

    kept
  end

  defp cut_eos(kept, []), do: kept

  defp cut_eos(kept, eos) do
    case Enum.find_index(kept, &(&1 in eos)) do
      nil -> kept
      i -> Enum.take(kept, i + 1)
    end
  end

  defp pages_for(n, page), do: div(n + page - 1, page)

  defp step!(tr, rows, tables) do
    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end
    oob = tr.pages
    table = for slot <- 0..(tr.ns - 1), do: Enum.take(Map.get(tables, slot, []) ++ List.duplicate(oob, tr.mp), tr.mp)

    inputs = %{tok: ids.(Enum.map(rows, &elem(&1, 0))), pos: ids.(Enum.map(rows, &elem(&1, 1))),
               slot: ids.(Enum.map(rows, &elem(&1, 2))), table: Tensor.from_list(:s32, [tr.ns, tr.mp], List.flatten(table))}

    {:ok, %{logits: l}, _} = Session.step(tr.session, inputs, [:logits])
    v = tr.vocab
    for i <- 0..(length(rows) - 1), do: binary_part(l.data, i * v * 4, v * 4)
  end

  # ------------------------------------------------------- prompt lookup --

  @doc """
  The prompt-lookup draft: for n = 3, 2, 1, every earlier occurrence of the
  context's last n tokens proposes what followed it (up to `depth` tokens),
  most recent first; one branch per distinct first token, at most
  `branches`.
  """
  def lookup(ctx, branches, depth) when depth > 0 do
    arr = List.to_tuple(ctx)
    len = tuple_size(arr)

    3..1//-1
    |> Enum.flat_map(fn ng ->
      if len <= ng, do: [], else: (
        pat = Enum.slice(ctx, len - ng, ng)

        for i <- (len - ng - 1)..0//-1, Enum.slice(ctx, i, ng) == pat do
          copy(arr, len, i + ng, depth)
        end)
    end)
    |> Enum.reject(&(&1 == []))
    |> Enum.uniq_by(&hd/1)
    |> Enum.take(branches)
  end

  def lookup(_ctx, _branches, _depth), do: []

  # what followed position `from`, copied with overlap (LZ77-style): past
  # the end of the context the copy continues from its own proposals, so
  # a loop the context has entered is proposed for the whole depth
  defp copy(arr, len, from, depth) do
    Enum.reduce(0..(depth - 1), [], fn j, acc ->
      src = from + j
      [if(src < len, do: elem(arr, src), else: Enum.at(Enum.reverse(acc), src - len)) | acc]
    end)
    |> Enum.reverse()
  end
end
