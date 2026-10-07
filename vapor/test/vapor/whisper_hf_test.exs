defmodule Vapor.WhisperHFTest do
  @moduledoc """
  Whisper (`Vapor.Lock.Adapters.Whisper`) against transformers itself: a
  `WhisperForConditionalGeneration` written by it (`test/python/hf_whisper.py`)
  is admitted with no tensor left unread; the encoder's hidden states, the
  decoder's logits and the greedy continuation are compared with
  transformers' — on the oracle and on the native worker, whose bits must
  agree.

  Controls that must fail: the decoder fed the encoder output of *other*
  audio (the cross-attention really reads the utterance), and the encoder
  fed the frames reversed in time (the convolution and the positions
  really read the order).
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Tensor}
  alias Vapor.Compile.Lower
  alias Vapor.Ingest.Safetensors
  alias Vapor.Lock.Adapters.Whisper
  alias Vapor.Runtime.{Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :torch
  @moduletag timeout: 900_000

  setup_all do
    dir = Path.join(System.tmp_dir!(), "vapor-whisper-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    py!(File.read!(Path.expand("../python/hf_whisper.py", __DIR__)), [dir, "11"])
    on_exit(fn -> File.rm_rf!(dir) end)
    {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
    {:ok, m} = Lock.open(dir)
    w = if Substrates.binary("vapor-worker", "native"), do: elem(Worker.start_link(exec: worker_exec(:host)), 1)
    {:ok, ref: ref, m: m, worker: w}
  end

  defp rel(got, want) do
    s = want |> Enum.map(&abs/1) |> Enum.max()
    (Enum.zip_with(got, want, &abs(&1 - &2)) |> Enum.max()) / s
  end

  defp runner(nil), do: fn p, env -> Oracle.eval_program(p, env) end

  defp runner(w) do
    fn p, env ->
      {:ok, comp} = Lower.lower(p)
      {:ok, r} = Native.run(w, comp, env, isa: Substrates.host_isa(), mode: :native)
      r.outputs
    end
  end

  test "admitted from transformers' own files, every tensor read; the encoder is a declared part", %{m: m} do
    assert m.spec.adapter == Whisper
    expected = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
    assert m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.reject(&MapSet.member?(expected, &1)) == []
    assert {:ok, _} = Lock.build(m.spec, m.weights, part: :encoder)
    assert {:error, _} = Lock.build(m.spec, m.weights, part: :vocoder)
  end

  test "encoder hidden states, decoder logits and greedy decoding = transformers; native bits = oracle bits",
       %{ref: ref, m: m, worker: w} do
    prompt = Tensor.to_list(ref["prompt"])

    results =
      for wk <- Enum.uniq([nil, w]) do
        {:ok, ids, enc} = Whisper.transcribe(m.spec, m.weights, ref["features"], prompt: prompt, max_tokens: 10, run: runner(wk))
        assert rel(Tensor.to_floats(enc.hidden), Tensor.to_floats(ref["hidden"])) < 1.0e-5
        assert prompt ++ ids == Tensor.to_list(ref["greedy"])
        {ids, enc.hidden}
      end

    assert Enum.uniq(results) |> length() == 1
    logits = prompt_logits(m, ref, ref["features"])
    assert rel(logits, Tensor.to_floats(ref["logits"])) < 1.0e-5
  end

  test "controls: other audio changes the decoder's logits; frames reversed in time change the encoder", %{ref: ref, m: m} do
    want = Tensor.to_floats(ref["logits"])
    other = Tensor.random(:f32, ref["features"].shape, 99)
    assert rel(prompt_logits(m, ref, other), want) > 0.01

    [mels, frames] = ref["features"].shape
    v = Tensor.to_list(ref["features"])
    reversed = Tensor.new(:f32, [mels, frames], Vapor.F32.encode(for(r <- 0..(mels - 1), t <- (frames - 1)..0//-1, do: Enum.at(v, r * frames + t))))
    {:ok, ep} = Lock.build(m.spec, m.weights, part: :encoder)
    hid = Oracle.eval_program(ep, Whisper.encoder_input(m.spec, reversed)).hidden
    assert rel(Tensor.to_floats(hid), Tensor.to_floats(ref["hidden"])) > 0.01
  end

  defp prompt_logits(m, ref, features) do
    c = m.spec.config
    {:ok, ep} = Lock.build(m.spec, m.weights, part: :encoder)
    enc = Oracle.eval_program(ep, Whisper.encoder_input(m.spec, features))
    {:ok, dp} = Lock.build(m.spec, m.weights, max_seq: c.tgt)
    prompt = Tensor.to_list(ref["prompt"])
    k = length(prompt)
    ids = &Tensor.from_list(:s32, [k], &1)
    cross = for l <- 0..(c.dec_layers - 1), kv <- ["xk", "xv"], into: %{}, do: {:"#{kv}#{l}", enc[:"#{kv}#{l}"]}
    caches = for {i, _} <- dp.state, into: %{}, do: {i, Tensor.new(:f32, [c.tgt, c.d], :binary.copy(<<0::32>>, c.tgt * c.d))}
    env = cross |> Map.merge(caches) |> Map.merge(%{tok: ids.(prompt), pos: ids.(Enum.to_list(0..(k - 1))), xh: ids.(List.duplicate(c.src - 1, k))})
    Tensor.to_floats(Oracle.eval_program(dp, env).logits)
  end
end
