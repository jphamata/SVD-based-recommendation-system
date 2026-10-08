defmodule Vapor.LockHFTest do
  @moduledoc """
  The airlock's alias, blueprint and topology adapters against Hugging Face
  transformers itself (PyTorch, float32): checkpoints *written by*
  transformers (`test/python/hf_lock.py`) are admitted by `Vapor.Lock` with
  no tensor left unread and compared against transformers' own forward pass.

  This closes the gap the 0.4.0 round declared: Phi-3, Granite and ViT had
  been compared against NumPy references written from a reading of the HF
  code — which would share a misreading. Here the reference is the
  modelling code itself.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Tensor}
  alias Vapor.Compile.Lower
  alias Vapor.Ingest.Safetensors
  alias Vapor.Lock.Adapters.Encoder
  alias Vapor.Runtime.{Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :torch
  @moduletag timeout: 900_000

  @decoders ~w(phi3 phi3-partial granite)
  @encoders ~w(vit vit-pooler clip-vision)
  @texts ~w(clip-text clip-model)
  @tol 1.0e-5

  setup_all do
    root = Path.join(System.tmp_dir!(), "vapor-lockhf-#{System.unique_integer([:positive])}")
    script = File.read!(Path.expand("../python/hf_lock.py", __DIR__))

    for v <- @decoders ++ @encoders ++ @texts do
      File.mkdir_p!(Path.join(root, v))
      py!(script, [Path.join(root, v), v, "11"])
    end

    w = if Substrates.binary("vapor-worker", "native"), do: elem(Worker.start_link(exec: worker_exec(:host)), 1)
    on_exit(fn -> File.rm_rf!(root) end)
    {:ok, root: root, worker: w}
  end

  defp unread(m) do
    expected = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
    m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.reject(&MapSet.member?(expected, &1))
  end

  defp rel(got, want) do
    scale = want |> Enum.map(&abs/1) |> Enum.max()
    (Enum.zip_with(got, want, &abs(&1 - &2)) |> Enum.max()) / scale
  end

  for v <- @decoders do
    test "#{v}: admitted from transformers' own files, every tensor read, prefill and greedy decoding = transformers",
         %{root: root, worker: w} do
      dir = Path.join(root, unquote(v))
      {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
      {:ok, m} = Lock.open(dir)
      assert unread(m) == []

      {:ok, p} = Lock.build(m.spec, m.weights, max_seq: 32)
      prompt = Tensor.to_list(ref["prompt"])
      c = m.spec.config

      run = fn env ->
        if w do
          {:ok, comp} = Lower.lower(p)
          {:ok, got} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native)
          got.outputs
        else
          Oracle.eval_program(p, env)
        end
      end

      step = fn caches, toks, p0 ->
        n = length(toks)
        e = Map.merge(caches, %{tok: Tensor.from_list(:s32, [n], toks), pos: Tensor.from_list(:s32, [n], Enum.to_list(p0..(p0 + n - 1)))})
        out = run.(e)
        {out.logits, Map.new(caches, fn {k, _} -> {k, out[:"#{k}_next"]} end)}
      end

      {logits, caches} = step.(Vapor.Model.Decoder.empty_caches(c, 32), prompt, 0)
      err = rel(Tensor.to_floats(logits), Tensor.to_floats(ref["logits"]))
      IO.puts("\n  #{unquote(v)}: prefill max |Δ|/max|ref| = #{Float.round(err, 9)}")
      assert err <= @tol

      greedy = ref["greedy"] |> Tensor.to_list() |> Enum.drop(length(prompt))
      [_, vsz] = logits.shape

      {got, _} =
        Enum.map_reduce(Enum.with_index(greedy), {logits, caches}, fn {_, i}, {lg, cs} ->
          row = lg |> Tensor.to_floats() |> Enum.take(-vsz)
          tok = row |> Enum.with_index() |> Enum.max_by(&elem(&1, 0)) |> elem(1)
          next = if i < length(greedy) - 1, do: step.(cs, [tok], length(prompt) + i), else: {nil, cs}
          {tok, next}
        end)

      assert got == greedy
    end
  end

  # pixels CHW (already normalised by the caller of transformers) → patch rows
  defp rows(pixels, spec) do
    [ch, h, wd] = pixels.shape
    vals = Tensor.to_floats(pixels) |> List.to_tuple()
    hwc = for y <- 0..(h - 1), x <- 0..(wd - 1), c <- 0..(ch - 1), do: elem(vals, (c * h + y) * wd + x)
    img = Vapor.Modal.Image.new(wd, h, ch, hwc)
    Vapor.Modal.Image.patches(img, spec.config.image.patch)
  end

  # CLIP's text tower: token ids, causal attention, the pooled end-of-text
  # row and the projection — what text-to-image search needs on the text side
  for v <- @texts do
    test "#{v}: the text tower admitted from transformers' files, every tensor read, outputs = transformers",
         %{root: root, worker: w} do
      dir = Path.join(root, unquote(v))
      {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
      opts = if unquote(v) == "clip-model", do: [tower: :text], else: []
      if opts != [], do: assert({:error, %Vapor.Rejection{repair: "open it with tower: :text or tower: :vision"}} = Lock.open(dir))
      {:ok, m} = Lock.open(dir, opts)
      assert m.spec.family == "clip_text" and m.spec.interface == :encoder
      if opts == [], do: assert(unread(m) == [])
      {:ok, p} = Lock.build(m.spec, m.weights)
      ids = Tensor.to_list(ref["ids"])
      env = Encoder.text_input(m.spec, ids)
      out = if w, do: (fn -> {:ok, comp} = Lower.lower(p); {:ok, g} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native); g.outputs end).(),
                  else: Oracle.eval_program(p, env)

      for {key, want} <- ref, key != "ids" do
        n = length(Tensor.to_floats(want))
        err = rel(Enum.take(Tensor.to_floats(out[String.to_atom(key)]), n), Tensor.to_floats(want))
        IO.puts("\n  #{unquote(v)} #{key}: max |Δ|/max|ref| = #{Float.round(err, 9)}")
        assert err <= @tol, "#{key}: #{err}"
      end
    end
  end

  for v <- @encoders do
    test "#{v}: admitted from transformers' own files, every tensor read, outputs = transformers", %{root: root} do
      dir = Path.join(root, unquote(v))
      {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
      {:ok, m} = Lock.open(dir)
      assert unread(m) == []
      {:ok, p} = Lock.build(m.spec, m.weights)
      out = Oracle.eval_program(p, Encoder.input(m.spec, rows(ref["pixels"], m.spec)))

      for {key, want} <- ref, key != "pixels" do
        got = out[String.to_atom(key)]
        assert got, "#{key}: the program has no such output"
        n = length(Tensor.to_floats(want))
        err = rel(Enum.take(Tensor.to_floats(got), n), Tensor.to_floats(want))
        IO.puts("\n  #{unquote(v)} #{key}: max |Δ|/max|ref| = #{Float.round(err, 9)}")
        assert err <= @tol, "#{key}: #{err}"
      end
    end
  end
end
