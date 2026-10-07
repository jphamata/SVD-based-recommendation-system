defmodule Vapor.Engine do
  @moduledoc """
  Continuous batching over a paged KV cache, on one resident session.

  The model is compiled once with `kv: {:paged, page, pages, sequences}` and
  `logits: :last`, and opened as a `Vapor.Runtime.Session` on a worker
  (optionally with a thread pool). Every step packs rows from *all* active
  sequences into one call — one row per decoding sequence plus a chunk of
  prompt from sequences still prefilling, up to `:step_tokens` — so the
  weights are read once for the whole batch. New requests join at the next
  step; finished ones leave and free their pages at once.

  Invariants (tested): a sequence's tokens are the same whether it runs
  alone or with others, however its prompt is chunked, wherever its pages
  are, and with any thread count — rows are independent in every kernel,
  attention reads only the sequence's own pages in logical order, and the
  sampler depends only on the row's logits, the seed and the step.

  Pages are reserved at admission for the prompt plus `max_tokens`, so a
  running sequence can never stall for memory (no deadlock, no preemption);
  a request that does not fit waits in the queue, one that can never fit is
  refused.

  **Circular cache.** When a sliding window `w` binds on every layer
  (the adapter says, `Vapor.Lock.ring_window/2`), no query ever reads a position
  older than `w`, so a sequence reserves only the pages its live range can
  span — `R = ⌈(w + step_tokens − 1)/page⌉`, not the whole context — and
  its block table maps logical page `j` to `mine[j mod R]`: position `x`
  is overwritten by position `x + R·page`, which a step writes only after
  every query that could read `x` has run (a step writes `p0…p0+n−1`, then
  reads back to `p0−w+1`). Attention
  walks the table from the window's first position (the paged kernel never
  touches older entries), so the bits are those of the uncapped cache —
  tested — while a 32 k context with a 4 k window holds ≈ 8× more
  sequences in the same memory.

  Messages to the requesting process: `{:vapor, ref, {:token, id, bytes}}`
  per generated token, then `{:vapor, ref, {:done, reason, usage}}` with
  `reason` in `:stop | :length | :eos | :error | :cancelled` and `usage =
  %{prompt_tokens, completion_tokens}`. `:error` means the worker died under
  the step: its caches are gone, so the sequences it held end there (the
  worker restarts, the session is reopened, and the engine keeps serving).

  A request lives as long as its receiver: the engine monitors it, and when
  the receiver exits — an HTTP client hung up, a LiveView closed, a job was
  killed — its sequences leave the batch and their slots and pages go back
  to the pool at the next step boundary. `cancel/2` does the same on
  purpose.

  With `replicas: n > 1`, `start_link/1` starts a `Vapor.Engine.Pool` of `n`
  engines instead — data parallelism over requests, behind the same API.
  """
  use GenServer
  alias Vapor.{Sampler, Tensor, Tokenizer}
  alias Vapor.Lock.Spec
  alias Vapor.Runtime.{Session, Substrates, Worker}

  defstruct [:session, :worker, :comp, :isa, :cfg, :spec, :tok, :page, :pages, :ns, :mp, :step_tokens, :eos, :name, :native, :window,
             free_slots: [], free_pages: [], seqs: %{}, queue: :queue.new(), stepping: false,
             stats: %{steps: 0, rows: 0, tokens: 0, busy_ns: 0, failures: 0, cancelled: 0}]

  # ---------------------------------------------------------------- API --

  @doc """
  Start an engine. Options: `:config` + `:weights` (or `:model`, a path
  for `Vapor.Model.open/1`),
  `:tokenizer` (a `Vapor.Tokenizer`, optional), `:max_seq` (context per
  sequence, default 512), `:page` (16), `:sequences` (8), `:pages`
  (default `sequences · max_seq / page`), `:step_tokens` (64), `:threads`
  (1), `:quantize`, `:name`, `:isa` (the host's widest, `Substrates.best_isa/0`;
  `:spirv` serves on the GPU — a resident session on a Vulkan daemon, the
  given `:fabric` or a new one; `staging: true` forces the discrete-GPU memory path;
  `:msl` the same on Metal — `vapor-metal`, or the given `:fabric`),
  `:replicas` (1), `:storage` (`:f32` | `:bf16`, see the adapter's builder).

  The engine knows models only through the model airlock (`Vapor.Lock`):
  `:config` may be any admitted configuration or a `Vapor.Lock.Spec`, and
  the spec must declare the `:causal_lm` contract with the `:paged`,
  `:sample` and `:last` features.
  """
  def start_link(opts) do
    if Keyword.get(opts, :replicas, 1) > 1,
      do: Vapor.Engine.Pool.start_link(opts),
      else: GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))
  end

  @doc """
  Submit a prompt (token ids). Options: `:max_tokens` (default 128),
  `:temperature`, `:top_k`, `:top_p`, `:seed`, `:stop` (strings),
  `:stop_ids` (token ids ending the turn, like EOS), `:constraint` (a
  `Vapor.Grammar.Constraint`: only the tokens it allows are chosen), `:to`
  (receiver, default the caller). Returns `{:ok, ref}` or `{:error, why}`.
  """
  def generate(engine, prompt, opts \\ []) when is_list(prompt),
    do: GenServer.call(engine, {:generate, prompt, Keyword.put_new(opts, :to, self())})

  @doc """
  Withdraw a request (queued or running); its receiver gets
  `{:done, :cancelled, usage}`. Unknown or finished refs are ignored.
  """
  def cancel(engine, ref), do: GenServer.cast(engine, {:cancel, ref})

  @doc "Submit and wait: `{:ok, ids, reason, usage}`."
  def complete(engine, prompt, opts \\ [], timeout \\ 600_000) do
    with {:ok, ref} <- generate(engine, prompt, opts), do: collect(ref, [], timeout)
  end

  @doc false
  def collect(ref, acc, timeout) do
    receive do
      {:vapor, ^ref, {:token, id, _}} -> collect(ref, [id | acc], timeout)
      {:vapor, ^ref, {:done, reason, usage}} -> {:ok, Enum.reverse(acc), reason, usage}
    after
      timeout -> {:error, :timeout}
    end
  end

  @doc "Model facts and counters."
  def info(engine), do: GenServer.call(engine, :info)

  # ------------------------------------------------------------- server --

  @impl true
  def init(opts) do
    isa = Keyword.get(opts, :isa, Substrates.best_isa())

    with {:ok, prep} <- Keyword.get_lazy(opts, :prepared, fn -> prepare(opts) end) |> wrap(),
         {:ok, w} <- substrate(isa, opts),
         {:ok, session} <- Session.open(w, prep.comp, isa: isa, consts: prep.consts, staging: Keyword.get(opts, :staging, false)) do
      tok = Keyword.get(opts, :tokenizer)

      {:ok,
       %__MODULE__{session: session, worker: w, comp: prep.comp, isa: isa, cfg: prep.cfg, spec: prep.spec, tok: tok, page: prep.page,
                   pages: prep.pages, ns: prep.ns, mp: prep.mp, step_tokens: prep.step_tokens, eos: eos_ids(prep.spec, tok),
                   name: Keyword.get(opts, :model_name, prep.spec.family), native: prep.native, window: Map.get(prep, :window),
                   free_slots: Enum.to_list(0..(prep.ns - 1)), free_pages: Enum.to_list(0..(prep.pages - 1))}}
    else
      {:error, why} -> {:stop, why}
    end
  end

  # `isa: :spirv` serves on the GPU: a resident session on a Vulkan daemon
  # (`:fabric`, an existing `Vapor.Runtime.Fabric`, or a fresh one)
  defp substrate(:spirv, opts) do
    case Keyword.fetch(opts, :fabric) do
      {:ok, f} -> {:ok, f}
      :error ->
        case Substrates.binary("vapor-fabric", "native") do
          nil -> {:error, :no_fabric}
          path -> Vapor.Runtime.Fabric.start_link(exec: [path])
        end
    end
  end

  # `isa: :msl` serves on Metal: the same session protocol on `vapor-metal`
  defp substrate(:msl, opts) do
    case Keyword.fetch(opts, :fabric) do
      {:ok, f} -> {:ok, f}
      :error ->
        case Substrates.binary("vapor-metal", "native") do
          nil -> {:error, :no_metal}
          path -> Vapor.Runtime.Fabric.start_link(exec: [path])
        end
    end
  end

  defp substrate(_isa, opts),
    do: Worker.start_link(exec: [Substrates.binary("vapor-worker", "native")], threads: Keyword.get(opts, :threads, 1))

  defp wrap({:ok, _} = ok), do: ok
  defp wrap({:error, _} = e), do: e
  defp wrap(%{} = prep), do: {:ok, prep}

  @doc """
  Everything an engine needs that does not depend on its worker: the model
  compiled for the paged, sampling, batched step, and the constant buffers
  (weights in content-addressed shared memory). A `Vapor.Engine.Pool`
  prepares once and starts every replica from it, so `n` replicas compile
  once and map the same weight pages.
  """
  def prepare(opts) do
    with {:ok, cfg, weights} <- model(opts),
         spec = Vapor.Lock.spec(cfg),
         :ok <- servable(spec) do
      s = Keyword.get(opts, :max_seq, min(spec.max_pos, 512))
      page = Keyword.get(opts, :page, 16)
      ns = Keyword.get(opts, :sequences, 8)
      pages = Keyword.get(opts, :pages, div(ns * s, page))
      step_tokens = Keyword.get(opts, :step_tokens, 64)

      with {:ok, prog} <- Vapor.Lock.build(spec, weights, max_seq: s, max_tokens: step_tokens, logits: :last, sample: true,
                                           kv: {:paged, page, pages, ns}, quantize: Keyword.get(opts, :quantize),
                                           storage: Keyword.get(opts, :storage, :f32)),
           {:ok, comp} <- Vapor.Compile.Lower.lower(prog) do
        {:ok, %{cfg: cfg, spec: spec, comp: comp, consts: Session.const_buffers(comp), native: Enum.any?(prog.outputs, &(elem(&1, 0) == :next)),
                page: page, ns: ns, pages: pages, mp: div(s, page), step_tokens: step_tokens, window: ring_window(spec, s)}}
      end
    end
  end

  defp ring_window(%Spec{} = spec, s), do: Vapor.Lock.ring_window(spec, s)

  # the engine drives a contract, not a family
  defp servable(%Spec{interface: :causal_lm} = spec) do
    case Enum.reject([:paged, :sample, :last], &Spec.supports?(spec, &1)) do
      [] -> :ok
      miss -> {:error, Vapor.Rejection.new({:engine, spec.family}, "a causal LM whose builder supports #{inspect(miss)}", "serve it with a builder that does")}
    end
  end

  defp servable(%Spec{} = spec),
    do: {:error, Vapor.Rejection.new({:engine, spec.family}, "the :causal_lm contract (got #{inspect(spec.interface)})", "use Vapor.Modal for encoders and codecs")}

  defp model(opts) do
    case Keyword.fetch(opts, :model) do
      {:ok, path} ->
        keep = if Keyword.get(opts, :storage) == :bf16, do: [bf16: :keep], else: []
        with {:ok, m} <- Vapor.Lock.open(path, keep), do: {:ok, m.spec, m.weights}

      :error ->
        {:ok, Keyword.fetch!(opts, :config), Keyword.fetch!(opts, :weights)}
    end
  end

  defp eos_ids(%Spec{eos: eos}, tok) do
    (eos ++ [tok && tok.eos]) |> List.flatten() |> Enum.reject(&is_nil/1) |> Enum.uniq()
  end

  # crash reports show the engine's shape, not its weights
  @impl true
  def format_status(_reason, [_pdict, st]),
    do: [data: [{~c"State", %{model: st.name, active: map_size(st.seqs), queued: :queue.len(st.queue), stats: st.stats}}]]

  @impl true
  def handle_call(:info, _from, st) do
    {:reply, %{model: st.name, config: st.cfg, substrate: Session.info(st.session), sequences: st.ns, pages: st.pages, page: st.page,
               max_seq: st.mp * st.page, window: st.window, ring_pages: st.window && ring_pages(st), active: map_size(st.seqs), queued: :queue.len(st.queue), stats: st.stats},
     st}
  end

  def handle_call({:generate, prompt, opts}, _from, st) do
    max_new = Keyword.get(opts, :max_tokens, 128)
    need = pages_for(length(prompt) + max_new, st.page)
    held = if st.window, do: min(need, ring_pages(st)), else: need

    cond do
      prompt == [] ->
        {:reply, {:error, :empty_prompt}, st}

      Enum.any?(prompt, &(not is_integer(&1) or &1 < 0 or &1 >= st.spec.vocab)) ->
        {:reply, {:error, :token_out_of_range}, st}

      need > st.mp ->
        {:reply, {:error, {:context_length, length(prompt) + max_new, st.mp * st.page}}, st}

      # more pages than the whole pool: it would wait in the queue forever
      held > st.pages ->
        {:reply, {:error, {:kv_pages, held, st.pages}}, st}

      true ->
        ref = make_ref()
        to = Keyword.fetch!(opts, :to)

        req = %{ref: ref, to: to, mon: Process.monitor(to), prompt: prompt, max_new: max_new, need: held,
                params: Sampler.params(opts), stop: List.wrap(Keyword.get(opts, :stop, [])),
                stop_ids: Keyword.get(opts, :stop_ids, []), constraint: Keyword.get(opts, :constraint)}

        st = %{st | queue: :queue.in(req, st.queue)} |> admit() |> kick()
        {:reply, {:ok, ref}, st}
    end
  end

  @impl true
  def handle_cast({:cancel, ref}, st), do: {:noreply, withdraw(st, &(&1.ref == ref), true)}

  @impl true
  # (every sequence may have been withdrawn since the step was scheduled)
  def handle_info(:step, st) when map_size(st.seqs) == 0, do: {:noreply, %{st | stepping: false} |> admit() |> kick()}
  def handle_info(:step, st), do: {:noreply, %{st | stepping: false} |> step() |> admit() |> kick()}

  # the receiver is gone: nobody will read these tokens
  def handle_info({:DOWN, mon, :process, _, _}, st), do: {:noreply, withdraw(st, &(&1.mon == mon), false)}

  # steps run inside handle_info, so this is always between two steps
  defp withdraw(st, match, notify?) do
    {gone, keep} = Enum.split_with(st.seqs, fn {_, seq} -> match.(seq) end)
    {qgone, q} = st.queue |> :queue.to_list() |> Enum.split_with(match)

    for r <- Enum.map(gone, &elem(&1, 1)) ++ qgone do
      Process.demonitor(r.mon, [:flush])
      if notify?, do: send(r.to, {:vapor, r.ref, {:done, :cancelled, %{prompt_tokens: length(r.prompt), completion_tokens: length(Map.get(r, :out, []))}}})
    end

    freed = Enum.flat_map(gone, fn {_, seq} -> seq.pages end)
    n = length(gone) + length(qgone)

    %{st | seqs: Map.new(keep), queue: :queue.from_list(q), free_slots: Enum.map(gone, &elem(&1, 0)) ++ st.free_slots,
           free_pages: freed ++ st.free_pages, stats: %{st.stats | cancelled: st.stats.cancelled + n}}
    |> admit()
    |> kick()
  end

  # a step is pending whenever there is work
  defp kick(%{stepping: false} = st) when map_size(st.seqs) > 0 do
    send(self(), :step)
    %{st | stepping: true}
  end

  defp kick(st), do: st

  defp pages_for(n, page), do: div(n + page - 1, page)

  # the ring under a window binding on every layer: position x lives in the
  # slot that position x + R·page overwrites. A step writes p0..p0+n−1
  # (n ≤ step_tokens) and then its queries read back to p0−w+1, so nothing
  # still readable is overwritten iff R·page ≥ w + n − 1
  defp ring_pages(st), do: pages_for(st.window + st.step_tokens - 1, st.page)

  # admit queued requests in order while a slot and their pages are free
  defp admit(st) do
    case :queue.peek(st.queue) do
      {:value, req} when st.free_slots != [] and length(st.free_pages) >= req.need ->
        {_, q} = :queue.out(st.queue)
        [slot | slots] = st.free_slots
        {mine, rest} = Enum.split(st.free_pages, req.need)

        seq = Map.merge(req, %{slot: slot, pages: mine, pending: req.prompt, pos: 0, out: [], text: "",
                               last: nil, step: 0})

        admit(%{st | queue: q, free_slots: slots, free_pages: rest, seqs: Map.put(st.seqs, slot, seq)})

      _ ->
        st
    end
  end

  # ------------------------------------------------------------ one step --

  defp step(st) do
    # decoding sequences first (one row each), then prompt chunks
    {decoding, prefilling} = st.seqs |> Enum.sort() |> Enum.split_with(fn {_, seq} -> seq.pending == [] end)

    {rows, wants, _budget} =
      (decoding ++ prefilling)
      |> Enum.reduce({[], [], st.step_tokens}, fn
        {_slot, _seq}, {rows, wants, 0} ->
          {rows, wants, 0}

        {slot, %{pending: []} = seq}, {rows, wants, budget} ->
          # decoding: the last sampled token at the next position
          {rows ++ [{seq.last, seq.pos, slot}], wants ++ [{slot, length(rows)}], budget - 1}

        {slot, seq}, {rows, wants, budget} ->
          chunk = Enum.take(seq.pending, budget)
          new_rows = chunk |> Enum.with_index(seq.pos) |> Enum.map(fn {t, p} -> {t, p, slot} end)
          done? = length(chunk) == length(seq.pending)
          wants = if done?, do: wants ++ [{slot, length(rows) + length(chunk) - 1}], else: wants
          {rows ++ new_rows, wants, budget - length(chunk)}
      end)

    ids = fn xs -> Tensor.from_list(:s32, [length(xs)], xs) end

    inputs = %{
      tok: ids.(Enum.map(rows, &elem(&1, 0))),
      pos: ids.(Enum.map(rows, &elem(&1, 1))),
      slot: ids.(Enum.map(rows, &elem(&1, 2))),
      table: table(st),
      # a step may produce no next token (every prompt chunk is partial):
      # one dummy row keeps the shape non-empty, its logits are ignored
      last: ids.(if(wants == [], do: [0], else: Enum.map(wants, &elem(&1, 1))))
    }

    # the substrate chooses the next token of every greedy or plain-temperature
    # row; logits cross back only when a top-k/top-p row needs them
    wanted = if wants == [], do: [{nil, 0}], else: wants
    beam? = not st.native or Enum.any?(wants, fn {slot, _} -> seq = st.seqs[slot]; seq.constraint != nil or not Sampler.native?(seq.params) end)

    inputs =
      if st.native do
        rows = Enum.flat_map(wanted, fn
          {nil, _} -> [0.0, 0.0]
          {slot, _} -> seq = st.seqs[slot]; Tuple.to_list(Sampler.native(seq.params, seq.step))
        end)

        Map.put(inputs, :sampling, Tensor.from_list(:f32, [length(wanted), 2], rows))
      else
        inputs
      end

    outs = (if st.native, do: [:next], else: []) ++ (if beam?, do: [:logits], else: [])
    case Session.step(st.session, inputs, outs) do
      {:ok, got, %{elapsed_ns: ns}} -> advance(st, rows, wants, wanted, got, ns, beam?)
      {:error, why} -> recover(st, why)
    end
  end

  # the worker died under the step: its caches are gone, so every sequence
  # it held ends with :error; the worker has restarted — reopen and go on
  defp recover(st, _why) do
    for {_, seq} <- st.seqs do
      Process.demonitor(seq.mon, [:flush])
      send(seq.to, {:vapor, seq.ref, {:done, :error, %{prompt_tokens: length(seq.prompt), completion_tokens: length(seq.out)}}})
    end

    st = %{st | seqs: %{}, free_slots: Enum.to_list(0..(st.ns - 1)), free_pages: Enum.to_list(0..(st.pages - 1)),
                stats: %{st.stats | failures: st.stats.failures + 1}}

    # constant buffers are re-placed: a pruned shared-memory file is rewritten
    case Session.open(st.worker, st.comp, isa: st.isa, consts: Session.const_buffers(st.comp), staging: st.session.info[:staged] == true) do
      {:ok, session} -> %{st | session: session}
      {:error, why} -> exit({:session_lost, why})
    end
  end

  defp advance(st, rows, wants, wanted, got, ns, beam?) do
    t = length(rows)
    v = st.spec.vocab
    nexts = if st.native, do: Tensor.to_list(got.next), else: List.duplicate(nil, length(wanted))

    # rows stay binary: greedy scans them in place, sampling decodes one row
    rows_out =
      for {i, nx} <- Enum.with_index(nexts, fn nx, i -> {i, nx} end), i < length(wants) do
        {nx, if(beam?, do: binary_part(got.logits.data, i * v * 4, v * 4))}
      end

    # advance every sequence by the rows it consumed
    consumed = Enum.frequencies_by(rows, &elem(&1, 2))

    seqs =
      Map.new(st.seqs, fn {slot, seq} ->
        n = Map.get(consumed, slot, 0)
        pending = if seq.pending == [], do: [], else: Enum.drop(seq.pending, n)
        {slot, %{seq | pos: seq.pos + n, pending: pending}}
      end)

    st = %{st | seqs: seqs, stats: %{st.stats | steps: st.stats.steps + 1, rows: st.stats.rows + t,
                                                 tokens: st.stats.tokens + length(wants), busy_ns: st.stats.busy_ns + ns}}

    wants
    |> Enum.zip(rows_out)
    |> Enum.reduce(st, fn {{slot, _}, row}, st -> emit(st, slot, row) end)
  end

  # block table: owned pages, then `pages` (out of range: writes skipped);
  # under a window binding on every layer, the owned pages in a ring
  defp table(st) do
    rows =
      for slot <- 0..(st.ns - 1) do
        case st.seqs do
          %{^slot => seq} when st.window != nil ->
            ring = List.to_tuple(seq.pages)
            for j <- 0..(st.mp - 1), do: elem(ring, rem(j, tuple_size(ring)))

          %{^slot => seq} ->
            Enum.take(seq.pages ++ List.duplicate(st.pages, st.mp), st.mp)

          _ ->
            List.duplicate(st.pages, st.mp)
        end
      end

    Tensor.from_list(:s32, [st.ns, st.mp], List.flatten(rows))
  end

  # sample, stream, and finish if a stop condition holds
  defp emit(st, slot, {native_id, row}) do
    seq = st.seqs[slot]
    {id, seq} = choose(st, seq, native_id, row)
    emit_id(st, slot, seq, id)
  end

  # a constrained sequence chooses among the ids its grammar allows (in the
  # BEAM, from the row's logits); the others take the substrate's choice or
  # sample here
  defp choose(st, %{constraint: nil} = seq, native_id, row),
    do: {if(st.native and Sampler.native?(seq.params), do: native_id, else: Sampler.sample(row, seq.params, seq.step)), seq}

  defp choose(_st, %{constraint: c} = seq, _native_id, row) do
    {allowed, c} = Vapor.Grammar.Constraint.allowed(c)
    allowed = case allowed do
      {:only, ids} -> ids
      :all -> :all
    end

    {Sampler.sample_masked(row, allowed, seq.params, seq.step), %{seq | constraint: c}}
  end

  # nothing allowed (the grammar has no continuation the vocabulary can spell)
  defp emit_id(st, slot, seq, nil) do
    Process.demonitor(seq.mon, [:flush])
    send(seq.to, {:vapor, seq.ref, {:done, :stop, %{prompt_tokens: length(seq.prompt), completion_tokens: length(seq.out)}}})
    %{st | seqs: Map.delete(st.seqs, slot), free_slots: [slot | st.free_slots], free_pages: seq.pages ++ st.free_pages}
  end

  defp emit_id(st, slot, seq, id) do
    bytes = if st.tok, do: Tokenizer.surface(st.tok, id), else: ""

    seq =
      case seq.constraint do
        nil -> seq
        c ->
          case Vapor.Grammar.Constraint.advance(c, id, bytes) do
            {:ok, c} -> %{seq | constraint: c}
            {:error, _} -> %{seq | constraint: nil}
          end
      end

    out = [id | seq.out]
    text = seq.text <> bytes
    n = length(out)

    {reason, text, visible} =
      cond do
        id in st.eos or id in seq.stop_ids -> {:eos, seq.text, ""}
        (hit = stop_hit(text, seq.stop)) != nil -> {:stop, binary_part(text, 0, hit), binary_part(text, byte_size(seq.text), max(hit - byte_size(seq.text), 0))}
        n >= seq.max_new -> {:length, text, bytes}
        true -> {nil, text, bytes}
      end

    if reason != :eos, do: send(seq.to, {:vapor, seq.ref, {:token, id, visible}})
    seq = %{seq | out: out, text: text, last: id, step: seq.step + 1}

    if reason do
      usage = %{prompt_tokens: length(seq.prompt), completion_tokens: if(reason == :eos, do: n - 1, else: n)}
      Process.demonitor(seq.mon, [:flush])
      send(seq.to, {:vapor, seq.ref, {:done, reason, usage}})
      %{st | seqs: Map.delete(st.seqs, slot), free_slots: [slot | st.free_slots], free_pages: seq.pages ++ st.free_pages}
    else
      %{st | seqs: Map.put(st.seqs, slot, seq)}
    end
  end

  defp stop_hit(_text, []), do: nil

  defp stop_hit(text, stops) do
    stops
    |> Enum.flat_map(fn s -> case :binary.match(text, s) do
      {at, _} -> [at]
      :nomatch -> []
    end end)
    |> Enum.min(fn -> nil end)
  end
end
