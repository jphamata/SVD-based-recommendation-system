defmodule Mix.Tasks.Vapor.Merge do
  @shortdoc "Fuse checkpoints (linear, task arithmetic, SLERP, TIES, DARE, RegMean), diagnose, select by measurement"
  @moduledoc """
      mix vapor.merge --out DIR [--method linear|task_arithmetic|slerp|ties|dare_linear|dare_ties|regmean]
                      [--weights 0.5,0.5] [--base DIR] [--t 0.5] [--density 0.5] [--lambda 1.0]
                      [--seed 0] [--alpha 0.9] [--calibrate A.txt,B.txt] [--dtype f32|bf16] MODEL_A MODEL_B …

      mix vapor.merge --stream --out DIR [--method …] [--base DIR] [--max-shard BYTES] [--dtype f32|bf16] MODEL_A MODEL_B …

      mix vapor.merge --diagnose [--base DIR] [--density 0.2] MODEL_A MODEL_B …

      mix vapor.merge --out DIR --try "linear;slerp;ties:density=0.2;task_arithmetic:lambda=0.5"
                      --eval held_out.txt [--base DIR] MODEL_A MODEL_B …

  Every input passes the model airlock; they must share the adapter, the
  contract, every tensor's name and shape and the configuration digest
  (`Vapor.Merge`). The fused checkpoint is written with the inputs'
  configuration and the first input's tokenizer files, plus `merge.receipt`
  (canonical CBOR, a `Vapor.Certificate` signed with a fresh key whose
  public half is in `merge.pub`) — anyone can recompute the output's Merkle
  root and check it.

  `--stream` fuses from disk to disk, one tensor at a time
  (`Merge.stream/3`): the memory of the largest tensor, not of the models —
  the same files and roots as without it. Element-wise methods only.

  `--diagnose` measures the regime before anything is fused (relative
  deltas, energy concentration, sign conflicts, whether the models share an
  ancestor) and says which methods' assumptions hold (`Merge.diagnose/2`).

  `--method regmean` needs `--calibrate`: one text per model, in that
  model's domain; its matrices' input Grams are measured on it
  (`Merge.calibrate/3`).

  `--try` builds every candidate (`method[:key=value,…]`, separated by
  `;`), scores each by held-out bits per byte of `--eval` through the
  substrate (`Vapor.Quality.Model.bits_per_byte/4`) and keeps the best; the
  receipt (`vapor.merge.select/1`) records every score and the digest of
  the evaluation text.
  """
  use Mix.Task

  @switches [out: :string, method: :string, weights: :string, base: :string, t: :float, density: :float,
             lambda: :float, seed: :integer, dtype: :string, alpha: :float, calibrate: :string,
             diagnose: :boolean, try: :string, eval: :string, stream: :boolean, max_shard: :integer]

  @impl true
  def run(argv) do
    {o, paths, _} = OptionParser.parse(argv, strict: @switches)
    if length(paths) < 1, do: Mix.raise("give the checkpoints to fuse")
    Mix.Task.run("app.start")

    if o[:stream], do: stream(paths, o), else: load(paths, o)
  end

  defp load(paths, o) do
    models = Enum.map(paths, &ok!(Vapor.Lock.open(&1)))
    base = o[:base] && ok!(Vapor.Lock.open(o[:base]))
    mm = Enum.map(models, &%{spec: &1.spec, weights: &1.weights})
    bm = base && %{spec: base.spec, weights: base.weights}

    if o[:diagnose], do: diagnose(mm, bm, o), else: fuse(models, mm, bm, paths, o)
  end

  defp diagnose(mm, bm, o) do
    d = Vapor.Merge.diagnose(mm, base: bm, density: o[:density] || 0.2)
    pct = fn x -> "#{Float.round(x * 100, 1)} %" end
    Mix.shell().info("regime: #{d.regime}  (#{d.tensors} float tensors)")

    for {m, i} <- Enum.with_index(d.models) do
      Mix.shell().info("  model #{i}: " <> Enum.map_join(m, ", ", fn {k, v} -> "#{k} #{if is_float(v) and v <= 2, do: pct.(v), else: v}" end))
    end

    for p <- d.pairs do
      Mix.shell().info("  pair #{inspect(p.models)}: " <> Enum.map_join(Map.delete(p, :models), ", ", fn {k, v} -> "#{k} #{Float.round(v, 4)}" end))
    end

    Enum.each(d.advice, &Mix.shell().info("→ " <> &1))
  end

  defp fuse(models, mm, bm, paths, o) do
    out = o[:out] || Mix.raise("--out DIR is required")
    key = Vapor.Certificate.keygen()

    {merged, label, extra} =
      if o[:try] do
        eval = File.read!(o[:eval] || Mix.raise("--try needs --eval held_out.txt"))
        tok = hd(models).tokenizer || Mix.raise("--eval needs a tokenizer in the first checkpoint")
        w = Vapor.Modal.Runner.worker()
        cands = parse_try(o[:try], bm)
        score = fn m -> Vapor.Quality.Model.bits_per_byte(Map.put(m, :tokenizer, tok), eval, 128, worker: w) end
        sel = ok!(Vapor.Merge.select(mm, cands, score, key: key, eval: Base.encode16(:crypto.hash(:sha256, eval), case: :lower)))

        for r <- sel.table,
            do: Mix.shell().info("  #{String.pad_trailing(r.label, 32)} #{if r.score, do: :erlang.float_to_binary(r.score, decimals: 4) <> " bits/byte", else: "refused: " <> r.refused}  (#{r.ms} ms)")

        {sel.merged, "selected #{sel.best}", %{"select.receipt" => Vapor.Certificate.encode(sel.receipt)}}
      else
        method = method!(o[:method] || "linear")
        grams = if method == :regmean, do: calibrate(models, o), else: nil

        opts =
          [method: method, key: key, base: bm, grams: grams] ++
            Enum.reject([weights: o[:weights] && Enum.map(String.split(o[:weights], ","), &parse_num/1), t: o[:t], density: o[:density],
                         lambda: o[:lambda], seed: o[:seed], alpha: o[:alpha]], fn {_, v} -> v == nil end)

        {t, merged} = :timer.tc(fn -> ok!(Vapor.Merge.merge(mm, opts)) end)
        {merged, "#{method} in #{div(t, 1000)} ms", %{}}
      end

    first = hd(models)
    dtype = String.upcase(o[:dtype] || "f32")

    case first.spec.config do
      %Vapor.Model.Config{} = c ->
        :ok = ok!(Vapor.Model.write(out, c, merged.weights, dtype: dtype))

      cfg ->
        File.mkdir_p!(out)
        File.write!(Path.join(out, "config.json"), Vapor.JSON.encode(Map.get(cfg, :raw) || %{}))
        :ok = Vapor.Ingest.Safetensors.write(Path.join(out, "model.safetensors"), Map.filter(merged.weights, fn {k, _} -> is_binary(k) end))
    end

    # the vocabulary and chat template come from the first input
    for f <- ~w(tokenizer.json tokenizer.gguf tokenizer_config.json chat_template.jinja special_tokens_map.json),
        src = Path.join(hd(paths), f), File.regular?(src), do: File.cp!(src, Path.join(out, f))

    File.write!(Path.join(out, "merge.receipt"), Vapor.Certificate.encode(merged.receipt))
    for {f, bin} <- extra, do: File.write!(Path.join(out, f), bin)
    File.write!(Path.join(out, "merge.pub"), Base.encode16(key.public, case: :lower))
    Mix.shell().info("fused #{length(models)} × #{first.spec.family}, #{label} → #{out} (output root #{merged.receipt.payload.output})")
  end

  defp stream(paths, o) do
    out = o[:out] || Mix.raise("--out DIR is required")
    if o[:try] || o[:diagnose], do: Mix.raise("--stream fuses one method; --try and --diagnose load the models")
    key = Vapor.Certificate.keygen()
    method = method!(o[:method] || "linear")

    opts =
      [method: method, key: key, base: o[:base], dtype: String.upcase(o[:dtype] || "f32")] ++
        Enum.reject([weights: o[:weights] && Enum.map(String.split(o[:weights], ","), &parse_num/1), t: o[:t], density: o[:density],
                     lambda: o[:lambda], seed: o[:seed], max_shard: o[:max_shard]], fn {_, v} -> v == nil end)

    {t, r} = :timer.tc(fn -> ok!(Vapor.Merge.stream(paths, out, opts)) end)

    for f <- ~w(config.json generation_config.json tokenizer.json tokenizer.gguf tokenizer_config.json chat_template.jinja special_tokens_map.json),
        src = Path.join(hd(paths), f), File.regular?(src), do: File.cp!(src, Path.join(out, f))

    File.write!(Path.join(out, "merge.receipt"), Vapor.Certificate.encode(r.receipt))
    File.write!(Path.join(out, "merge.pub"), Base.encode16(key.public, case: :lower))
    Mix.shell().info("streamed #{length(paths)} checkpoints, #{method}, #{r.tensors} tensors, #{div(r.bytes, 1_000_000)} MB in #{div(t, 1000)} ms → #{out} (output root #{r.receipt.payload.output})")
  end

  defp method!(name) do
    Enum.find(Vapor.Merge.methods(), &(Atom.to_string(&1) == name)) || Mix.raise("method: one of #{inspect(Vapor.Merge.methods())}")
  end

  # "ties:density=0.2,lambda=1;linear" → [{"ties density=0.2 lambda=1", [method: :ties, density: 0.2, …]}, …]
  defp parse_try(spec, base) do
    for item <- String.split(spec, ";", trim: true) do
      [m | kvs] = String.split(String.trim(item), [":", ","], trim: true)
      method = method!(m)
      params = for kv <- kvs, [k, v] = String.split(kv, "=", parts: 2), do: {String.to_existing_atom(k), parse_num(v)}
      params = if method in [:task_arithmetic, :ties, :dare_linear, :dare_ties], do: [base: base || Mix.raise("#{m} needs --base")] ++ params, else: params
      {String.trim(item), [method: method] ++ params}
    end
  end

  defp calibrate(models, o) do
    files = String.split(o[:calibrate] || Mix.raise("--method regmean needs --calibrate A.txt,B.txt (one per model)"), ",")
    if length(files) != length(models), do: Mix.raise("--calibrate: one text per model")
    w = Vapor.Modal.Runner.worker()

    Enum.zip_with(models, files, fn m, f ->
      tok = m.tokenizer || Mix.raise("--calibrate needs a tokenizer in every checkpoint")
      ids = Vapor.Tokenizer.encode(tok, File.read!(f), add_bos: false)
      ok!(Vapor.Merge.calibrate(%{spec: m.spec, weights: m.weights}, ids |> Enum.chunk_every(128) |> Enum.filter(&(length(&1) > 1)) |> Enum.take(32), worker: w))
    end)
  end

  defp parse_num(s) do
    case Float.parse(s) do
      {f, ""} -> f
      _ -> Mix.raise("expected a number, got #{inspect(s)}")
    end
  end

  defp ok!({:ok, v}), do: v
  defp ok!(:ok), do: :ok
  defp ok!({:error, %Vapor.Rejection{} = r}), do: Mix.raise("refused at #{inspect(r.node)}: #{r.bound} — #{r.repair}")
  defp ok!({:error, e}), do: Mix.raise(inspect(e))
end
