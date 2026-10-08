defmodule Vapor.SpeculativeTest do
  @moduledoc """
  Phase P8 — speculative decoding returns exactly the target's greedy
  tokens whatever the draft (batch invariance makes the k+1-row
  verification step bit-identical to k+1 single steps); a good draft
  saves target steps.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Sampler, Speculative, Tensor}
  alias Vapor.Model.{Config, Decoder}
  alias Vapor.Runtime.{Session, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag :native
  @moduletag timeout: 900_000

  defp session(c, ws) do
    {:ok, p} = Decoder.program(c, ws, max_seq: 64, max_tokens: 8)
    {:ok, comp} = Vapor.Compile.Lower.lower(p)
    {:ok, w} = Worker.start_link(exec: worker_exec(:host))
    {:ok, s} = Session.open(w, comp, isa: Substrates.host_isa())
    s
  end

  defp greedy(s, prompt, n, v) do
    ids = &Tensor.from_list(:s32, [length(&1)], &1)
    row = fn l -> binary_part(l.data, byte_size(l.data) - 4 * v, 4 * v) end
    {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids.(prompt), pos: ids.(Enum.to_list(0..(length(prompt) - 1)))}, [:logits])

    {toks, _} =
      Enum.map_reduce(0..(n - 1), {Sampler.argmax(row.(l)), length(prompt)}, fn _, {t, p} ->
        {:ok, %{logits: l}, _} = Session.step(s, %{tok: ids.([t]), pos: ids.([p])}, [:logits])
        {t, {Sampler.argmax(row.(l)), p + 1}}
      end)

    toks
  end

  # the target's weights with a small perturbation: a draft that mostly agrees
  defp perturbed(ws, eps) do
    Map.new(ws, fn {k, t} ->
      noise = Tensor.random(:f32, t.shape, :erlang.phash2(k), scale: eps) |> Tensor.to_floats()
      {k, Tensor.from_list(:f32, t.shape, Enum.zip_with(Tensor.to_floats(t), noise, &(&1 + &2)))}
    end)
  end

  test "output = target greedy, for an identical, a close and an unrelated draft; acceptance tracks the draft" do
    {:ok, c} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128}))
    ws = tiny_weights(c, 9)
    {:ok, c1} = Config.from_map(tiny_config("llama", %{"vocab_size" => 128, "num_hidden_layers" => 1}))
    prompt = [5, 17, 99, 3, 42]
    want = greedy(session(c, ws), prompt, 40, c.vocab)

    rates =
      for {name, dc, dws} <- [{"same", c, ws}, {"close", c, perturbed(ws, 0.01)}, {"unrelated", c1, tiny_weights(c1, 33)}] do
        {got, st} = Speculative.generate(session(c, ws), session(dc, dws), prompt, 40, 4, c.vocab)
        assert got == want, name
        {name, st.accepted / st.proposed, st.target_steps}
      end

    [{_, same, same_steps}, {_, close, _}, {_, unrelated, _} ] = rates
    IO.puts("\n  acceptance: same #{Float.round(same, 2)}, close #{Float.round(close, 2)}, unrelated #{Float.round(unrelated, 2)}; target steps with the same draft: #{same_steps} for 40 tokens")
    assert same == 1.0 and same_steps <= 11
    assert close >= unrelated
  end
end
