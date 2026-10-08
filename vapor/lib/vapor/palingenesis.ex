defmodule Vapor.Palingenesis do
  @moduledoc """
  **Palingenesis** (παλιγγενεσία, the alchemists' rebirth of a form from its
  own ashes): a model that is renewed plank by plank, as Theseus' ship was,
  without ever being retrained whole and without losing what it was
  (docs/PALINGENESIS.md).

  A model here is a **hull**: its tensors grouped into **planks** (an
  attention block, an MLP block, one expert of a mixture, a norm, the
  embeddings), each named by the Merkle root of its tensors, the hull named
  by the Merkle root of its planks. A plank is replaced only through a
  **touchstone of gates**, and a replacement that passes becomes a new
  **generation**, published by an atomic swap:

  | gate | question | measure |
  |---|---|---|
  | contract | is it the same part? | the same tensor names, shapes and dtypes |
  | drift (the brake) | does the ship still behave as it did where it must? | Fisher–Rao distance `2·arccos Σ√(pᵢqᵢ)` between the old and new next-token distributions at every position of the **anchor** sequences: the maximum must stay under `ε` |
  | target | does the new plank do what it was brought for? | bits per token on the **target** sequences, old against new, per sequence; the paired sign-flip test of the mean difference (exact up to 16 sequences) must reject "no difference" at `alpha`, with the mean improvement positive |
  | invariants | does it still keep its promises? | caller-supplied checks run on the candidate model |

  The brake and the target pull in opposite directions; a plank is
  admitted only when it improves what it claims **within** the stated
  drift budget. The anchors are what the ship must not forget, and they
  are the only place the brake looks: drift elsewhere is unmeasured, and
  this module says so instead of promising "no forgetting".

  **Generations and readers (read-copy-update).** The current generation of
  a hull is published with `:persistent_term`, so a reader takes it with
  one lock-free, zero-copy lookup. A generation is an immutable value: a
  reader that holds generation `N` keeps exactly `N`'s weights for as long
  as it holds it, while generation `N + 1` is published beside it.
  Generation `N` is reclaimed by the garbage collector when its last reader
  lets go. This is RCU, and on the BEAM it costs nothing beyond immutability. Writers are serialised
  per hull (`:global.trans/3`), so two proposals cannot both build on `N`.
  An engine serving from a worker's shared memory sees a new generation
  when it opens its next session, not in the middle of one.

  **Lineage is identity.** Each generation carries a record: its number,
  its root, its parent's root, the plank replaced, every gate's
  measurement and the hash of the previous record. `verify/2` re-derives
  the chain and the current root from the weights. The ship's planks may
  all be new; the chain of records says it is the same ship, and how it
  came to be this one. Records may be signed (Ed25519, `key:`).

  **Alignment, stated precisely.** Replacing a whole MLP block with one
  whose hidden units are permuted changes *nothing*: each hidden unit is
  computed and consumed inside the block, so a permutation is invisible to
  the rest of the network (and is measured as such). Permutation symmetry
  matters only when a plank is **blended** with the old one
  (`mode: {:blend, t}`); then the new plank is first aligned to the old one
  by the exact Hungarian method (`Vapor.Merge.Align`), and the test shows
  what happens without it.
  """
  alias Vapor.{Certificate, Merge, Merkle, Tensor}
  alias Vapor.Assay.Stats
  alias Vapor.Merge.Align

  defmodule Generation do
    @moduledoc "An immutable generation of a hull: what a reader holds."
    defstruct [:name, :spec, :gen, :weights, :planks, :root, :records]
  end

  @parents ~w(self_attn mlp attention feed_forward block_sparse_moe shared_experts shared_expert)

  # ------------------------------------------------------------- the planks

  @doc """
  Group tensor names into planks: `%{plank => [names]}`. A parameter's
  suffix (`.weight`, `.bias`) is dropped, and so is its projection when
  its parent is a block (`…self_attn.q_proj` → `…self_attn`) or an expert
  (`…experts.3.gate_proj` → `…experts.3`).
  """
  def planks(weights) do
    weights
    |> Map.keys()
    |> Enum.filter(&is_binary/1)
    |> Enum.group_by(&plank_of/1)
    |> Map.new(fn {k, v} -> {k, Enum.sort(v)} end)
  end

  defp plank_of(name) do
    base = String.replace(name, ~r/\.(weight|bias)$/, "")
    rev = base |> String.split(".") |> Enum.reverse()

    case rev do
      [_leaf, parent | _] when parent in @parents -> rev |> tl() |> Enum.reverse() |> Enum.join(".")
      [_leaf, n, "experts" | _] -> if n =~ ~r/^\d+$/, do: rev |> tl() |> Enum.reverse() |> Enum.join("."), else: base
      _ -> base
    end
  end

  @doc "A plank's root: the Merkle root of its tensors (`Vapor.Merge.weights_root/1`'s leaves)."
  def plank_root(weights, names), do: Merge.weights_root(Map.take(weights, names))

  @doc "A hull's root: the Merkle root of `(plank, plank root)` leaves, sorted by plank (hex)."
  def root(weights, planks) do
    planks
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map(fn {id, names} -> Merkle.leaf(Vapor.Canonical.encode({id, plank_root(weights, names)})) end)
    |> Merkle.root()
    |> Base.encode16(case: :lower)
  end

  # ---------------------------------------------------------- publishing

  @doc """
  Launch a hull from an admitted model `%{spec, weights}`: generation 0,
  published under `name` (any term). Options: `key:` signs the record.
  """
  def launch(name, %{spec: spec, weights: weights}, opts \\ []) do
    ps = planks(weights)
    r = root(weights, ps)
    record = sign(%{gen: 0, root: r, parent: nil, prev: nil, plank: nil, kind: "launch", planks: map_size(ps)}, opts[:key])
    g = %Generation{name: name, spec: spec, gen: 0, weights: weights, planks: ps, root: r, records: [record]}
    :persistent_term.put(key(name), g)
    {:ok, g}
  end

  @doc "The current generation of a hull: one lock-free lookup; the value is immutable."
  def checkout(name) do
    case :persistent_term.get(key(name), nil) do
      nil -> {:error, "no hull named #{inspect(name)}"}
      g -> {:ok, g}
    end
  end

  @doc "Take a hull out of service (readers already holding a generation keep it)."
  def retire(name), do: (:persistent_term.erase(key(name)); :ok)

  defp key(name), do: {__MODULE__, name}

  # --------------------------------------------------------------- proposing

  @doc """
  Propose `tensors` (`%{name => %Tensor{}}`, exactly the plank's tensors) as
  the new `plank` of the hull. Gates run against the current generation;
  if all pass, generation `N + 1` is published atomically and returned.

  Options:

    * `anchors` (required): token-id sequences the ship must keep behaving
      on; `epsilon` (default 0.05): the largest Fisher–Rao distance allowed
      at any anchor position;
    * `targets`: sequences the plank is meant to improve; `alpha` (default
      0.05) for the paired sign-flip test; without targets the plank must still pass
      the brake, and the record says no improvement was claimed;
    * `invariants`: `[{name, fn %{spec, weights} -> :ok | {:error, why} end}]`;
    * `mode`: `:replace` (default) or `{:blend, t}` with `0 < t ≤ 1`
      (aligned to the old plank first when the plank is a SwiGLU block;
      `align: false` blends without aligning, the control);
    * `key`: sign the record; `worker`: run the measurements on a native
      worker (the same bits as the oracle).

  Returns `{:ok, generation, report}` or `{:error, report}`; the report
  holds every gate's measurement, the failing gate named.
  """
  def propose(name, plank, tensors, opts \\ []) do
    :global.trans({__MODULE__, name}, fn -> do_propose(name, plank, tensors, opts) end, [node()])
  end

  defp do_propose(name, plank, tensors, opts) do
    with {:ok, g} <- checkout(name),
         {:ok, names} <- Map.fetch(g.planks, plank) |> then(&if(&1 == :error, do: {:error, %{gate: "contract", reason: "no plank #{plank}"}}, else: &1)),
         :ok <- contract(g.weights, names, tensors),
         {:ok, new_plank, align} <- shape_plank(g.weights, names, tensors, Keyword.get(opts, :mode, :replace), Keyword.get(opts, :align, true)) do
      weights = Map.merge(g.weights, new_plank)
      r = root(weights, g.planks)

      if r == g.root do
        {:error, %{gate: "contract", reason: "the proposed plank is identical to the current one"}}
      else
        gates(g, weights, opts)
        |> finish(g, plank, weights, r, align, opts)
      end
    else
      {:error, %{} = report} -> {:error, report}
      {:error, why} -> {:error, %{gate: "contract", reason: why}}
    end
  end

  defp contract(weights, names, tensors) do
    given = tensors |> Map.keys() |> Enum.sort()

    cond do
      given != names -> {:error, %{gate: "contract", reason: "the plank is #{Enum.join(names, ", ")}; given #{Enum.join(given, ", ")}"}}
      (bad = Enum.find(names, fn n -> {tensors[n].shape, tensors[n].dtype} != {weights[n].shape, weights[n].dtype} end)) != nil ->
        {:error, %{gate: "contract", reason: "#{bad}: shape/dtype #{inspect({tensors[bad].shape, tensors[bad].dtype})}, the hull has #{inspect({weights[bad].shape, weights[bad].dtype})}"}}
      true -> :ok
    end
  end

  defp shape_plank(_weights, _names, tensors, :replace, _align), do: {:ok, tensors, nil}

  defp shape_plank(weights, names, tensors, {:blend, t}, align?) when is_number(t) and t > 0 and t <= 1 do
    old = Map.take(weights, names)

    {aligned, report} =
      case align? && Align.align(%{weights: old}, %{weights: tensors}) do
        {:ok, %{weights: a}, rep} -> {a, %{moved: Enum.map(rep.blocks, & &1.moved), digest: rep.digest}}
        _ -> {tensors, nil}
      end

    blended = Map.new(names, fn n -> {n, lerp(old[n], aligned[n], t)} end)
    {:ok, blended, report}
  end

  defp shape_plank(_, _, _, mode, _), do: {:error, %{gate: "contract", reason: "mode is :replace or {:blend, t} with 0 < t ≤ 1, not #{inspect(mode)}"}}

  defp lerp(%Tensor{} = a, %Tensor{} = b, t) do
    out = Enum.zip_with(Tensor.to_floats(a), Tensor.to_floats(b), fn x, y -> x + t * (y - x) end)
    f = Tensor.from_list(:f32, a.shape, out)
    if a.dtype == :bf16, do: Tensor.to_bf16(f), else: f
  end

  # --------------------------------------------------------------- the gates

  defp gates(g, weights, opts) do
    anchors = Keyword.get(opts, :anchors, [])
    eps = Keyword.get(opts, :epsilon, 0.05)
    ro = Keyword.take(opts, [:worker])
    cand = %{spec: g.spec, weights: weights}

    with true <- (anchors != [] and Enum.all?(anchors, &(is_list(&1) and length(&1) >= 1))) ||
                   {:error, %{gate: "drift", reason: "anchors are required: the sequences the ship must not forget"}},
         drift = drift(g, weights, anchors, ro),
         true <- drift.max <= eps || {:error, %{gate: "drift", drift: Map.put(drift, :epsilon, eps), reason: "max Fisher–Rao drift #{fmt(drift.max)} > ε = #{fmt(eps)}"}},
         {:ok, target} <- target(g, weights, Keyword.get(opts, :targets, []), Keyword.get(opts, :alpha, 0.05), ro, Map.put(drift, :epsilon, eps)),
         {:ok, inv} <- invariants(cand, Keyword.get(opts, :invariants, []), Map.put(drift, :epsilon, eps), target) do
      {:ok, %{drift: Map.put(drift, :epsilon, eps), target: target, invariants: inv}}
    end
  end

  defp drift(g, weights, anchors, ro) do
    ds =
      Enum.flat_map(anchors, fn ids ->
        old = Vapor.Modal.Text.logits(g.spec, g.weights, ids, ro)
        new = Vapor.Modal.Text.logits(g.spec, weights, ids, ro)
        Enum.zip_with(old, new, fn a, b -> Vapor.InfoGeom.fisher_rao(softmax(a), softmax(b)) end)
      end)

    %{max: Enum.max(ds), mean: Enum.sum(ds) / length(ds), positions: length(ds), anchors: length(anchors)}
  end

  defp target(_g, _weights, [], _alpha, _ro, _drift), do: {:ok, nil}

  defp target(g, weights, targets, alpha, ro, drift) do
    old = Enum.map(targets, &bits(g.spec, g.weights, &1, ro))
    new = Enum.map(targets, &bits(g.spec, weights, &1, ro))
    d = Enum.zip_with(old, new, &(&1 - &2))
    better = Enum.count(d, &(&1 > 0))
    # the claim is about the mean (the bits a reader pays): the paired sign-flip test, exact up to 16 sequences
    p = Stats.sign_flip(d, 20_000, 17)
    m = Stats.mean(d)
    t = %{sequences: length(targets), bits_old: Stats.mean(old), bits_new: Stats.mean(new), improved: better, p_value: p, alpha: alpha}

    if p <= alpha and m > 0,
      do: {:ok, t},
      else: {:error, %{gate: "target", drift: drift, target: t, reason: "no significant improvement: mean Δ #{fmt(m)} bits/token, sign-flip p = #{fmt(p)} (alpha #{fmt(alpha)}); #{better} of #{length(d)} sequences better"}}
  end

  defp invariants(_cand, [], _drift, _target), do: {:ok, []}

  defp invariants(cand, checks, drift, target) do
    Enum.reduce_while(checks, {:ok, []}, fn {nm, f}, {:ok, acc} ->
      case f.(cand) do
        :ok -> {:cont, {:ok, acc ++ [to_string(nm)]}}
        {:error, why} -> {:halt, {:error, %{gate: "invariants", drift: drift, target: target, reason: "#{nm}: #{why}"}}}
      end
    end)
  end

  # bits per token of a sequence under teacher forcing
  defp bits(spec, weights, ids, ro) when length(ids) >= 2 do
    rows = Vapor.Modal.Text.logits(spec, weights, ids, ro)

    rows
    |> Enum.zip(tl(ids))
    |> Enum.map(fn {row, next} -> -:math.log2(Enum.at(softmax(row), next)) end)
    |> then(&(Enum.sum(&1) / length(&1)))
  end

  defp softmax(row) do
    m = Enum.max(row)
    es = Enum.map(row, &:math.exp(&1 - m))
    s = Enum.sum(es)
    Enum.map(es, &(&1 / s))
  end

  # ---------------------------------------------------------------- publish

  defp finish({:error, report}, _g, _plank, _weights, _r, _align, _opts), do: {:error, report}

  defp finish({:ok, report}, g, plank, weights, r, align, opts) do
    prev = hd(g.records)

    rec =
      sign(%{gen: g.gen + 1, root: r, parent: g.root, prev: record_hash(prev), plank: plank,
             kind: if(match?({:blend, _}, Keyword.get(opts, :mode, :replace)), do: "blend", else: "replace"),
             blend: case Keyword.get(opts, :mode) do {:blend, t} -> t * 1.0; _ -> nil end,
             alignment: align, drift: report.drift, target: report.target, invariants: report.invariants}, opts[:key])

    next = %Generation{g | gen: g.gen + 1, weights: weights, root: r, records: [rec | g.records]}
    # the swap: one put; readers holding the previous generation keep it
    :persistent_term.put(key(g.name), next)
    {:ok, next, Map.put(report, :record, rec)}
  end

  defp sign(payload, nil), do: %Certificate{payload: payload}
  defp sign(payload, key), do: Certificate.sign(%Certificate{payload: payload}, key)

  @doc "A record's identity: SHA-256 of its canonical payload (signatures excluded), hex."
  def record_hash(%Certificate{} = c), do: Base.encode16(:crypto.hash(:sha256, Certificate.canonical(c)), case: :lower)

  # ------------------------------------------------------------------ verify

  @doc """
  Verify a generation's lineage: records numbered `0..N` from launch, each
  naming the hash of the one before and its parent's root, and the
  current root re-derived from the weights. Options: `trusted:` public keys
  whose signature every record must carry.
  `:ok` or `{:error, why}`.
  """
  def verify(%Generation{} = g, opts \\ []) do
    chain = Enum.reverse(g.records)
    trusted = Keyword.get(opts, :trusted)

    links =
      chain
      |> Enum.with_index()
      |> Enum.reduce_while({:ok, nil}, fn {rec, i}, {:ok, before} ->
        p = rec.payload

        cond do
          p.gen != i -> {:halt, {:error, "record #{i} says generation #{p.gen}"}}
          before != nil and p.prev != record_hash(before) -> {:halt, {:error, "record #{i} does not name the record before it"}}
          before != nil and p.parent != before.payload.root -> {:halt, {:error, "record #{i}'s parent is not the root before it"}}
          trusted != nil and Certificate.verify(rec, trusted) != :ok -> {:halt, {:error, "record #{i} is not signed by a trusted key"}}
          true -> {:cont, {:ok, rec}}
        end
      end)

    with {:ok, last} <- links do
      cond do
        last.payload.root != g.root -> {:error, "the last record's root is not the generation's root"}
        root(g.weights, planks(g.weights)) != g.root -> {:error, "the weights do not hash to the recorded root"}
        true -> :ok
      end
    end
  end

  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, [{:decimals, 4}, :compact])
  defp fmt(x), do: to_string(x)
end
