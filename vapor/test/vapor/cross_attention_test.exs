defmodule Vapor.CrossAttentionTest do
  @moduledoc """
  Cross-attention — one stream's queries over another stream's keys and
  values, the operator that fuses modalities in latent space without a text
  pivot (audio frames attending to video patches, a U-Net or DiT attending
  to a prompt) — is the existing attention operator: the other stream's
  projections are the key/value table and every query row's horizon is that
  table's last row. Checked against `torch.nn.MultiheadAttention`.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Ingest.Safetensors
  alias Vapor.Runtime.Oracle
  import Vapor.TestHelpers

  @moduletag :torch

  @script """
  import sys, torch
  from safetensors.torch import save_file
  torch.manual_seed(4)
  n, m, d, h = 5, 7, 32, 2
  mha = torch.nn.MultiheadAttention(d, h, batch_first=True)
  a, b = torch.randn(1, n, d), torch.randn(1, m, d)
  with torch.no_grad():
      y, _ = mha(a, b, b)
  W, B = mha.in_proj_weight, mha.in_proj_bias
  save_file({"a": a[0], "b": b[0], "wq": W[:d].contiguous(), "wk": W[d:2*d].contiguous(), "wv": W[2*d:].contiguous(),
             "bq": B[:d].contiguous(), "bk": B[d:2*d].contiguous(), "bv": B[2*d:].contiguous(),
             "wo": mha.out_proj.weight.detach().contiguous(), "bo": mha.out_proj.bias.detach().contiguous(),
             "y": y[0].contiguous()}, sys.argv[1])
  """

  test "queries of one stream over another's keys and values = torch.nn.MultiheadAttention" do
    path = Path.join(System.tmp_dir!(), "vapor-xattn-#{System.unique_integer([:positive])}.safetensors")
    py!(@script, [path])
    {:ok, r} = Safetensors.read(path)
    File.rm!(path)
    [n, d] = r["a"].shape
    [m, _] = r["b"].shape
    row = fn t -> Tensor.new(:f32, [1, d], t.data) end
    lin = fn x, w, b -> T.add(T.linear(x, T.const(r[w])), T.const(row.(r[b]))) end

    a = T.input(:a, :f32, [n, d])
    b = T.input(:b, :f32, [m, d])
    horizon = T.input(:horizon, :s32, [n])
    att = T.attention(lin.(a, "wq", "bq"), lin.(b, "wk", "bk"), lin.(b, "wv", "bv"), horizon, 2, 2)
    p = Program.new(y: lin.(att, "wo", "bo"))
    %{y: y} = Oracle.eval_program(p, %{a: r["a"], b: r["b"], horizon: Tensor.from_list(:s32, [n], List.duplicate(m - 1, n))})

    want = Tensor.to_floats(r["y"])
    scale = want |> Enum.map(&abs/1) |> Enum.max()
    assert (Enum.zip_with(Tensor.to_floats(y), want, &abs(&1 - &2)) |> Enum.max()) <= 1.0e-6 * scale
  end
end
