defmodule Vapor.SpatialTest do
  @moduledoc """
  Spatial operators without a spatial kernel (`Vapor.Spatial`): convolution
  as gather + selection + reshape + GEMV, GroupNorm by exact selector
  contractions, nearest upsampling by gather, pixel attention by the
  encoder's horizon — against PyTorch itself, bit-identical across
  substrates; and the continuous latent decoder of latent diffusion
  (diffusers' `AutoencoderKL`, `Vapor.Lock.Adapters.VAE`) against diffusers.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Lock, Program, Spatial, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Ingest.Safetensors
  alias Vapor.Lock.Adapters.VAE
  alias Vapor.Runtime.{Dispatch, Native, Oracle, Substrates, Worker}
  import Vapor.TestHelpers

  @moduletag timeout: 1_800_000
  @tol 1.0e-6

  defp rel(got, want) do
    scale = want |> Enum.map(&abs/1) |> Enum.max()
    (Enum.zip_with(got, want, &abs(&1 - &2)) |> Enum.max()) / scale
  end

  defp prog(y, lets), do: Program.new([y: y], lets: Enum.reverse(lets))

  test "reshape is an exact relabelling, on every path (oracle, lowering)" do
    x = T.input(:x, :f32, [6, 4])
    xv = Tensor.random(:f32, [6, 4], 1)
    p = Program.new(y: T.reshape(x, [3, 8]))
    assert Oracle.eval_program(p, %{x: xv}).y == %{xv | shape: [3, 8]}
    assert {:error, _} = T.infer(T.reshape(x, [5, 5]))
  end

  @tag :torch
  test "conv2d (padding, stride, dilation, 1×1, 2×5), conv3d, GroupNorm, upsampling, pixel attention = PyTorch" do
    path = Path.join(System.tmp_dir!(), "vapor-spatial-#{System.unique_integer([:positive])}.safetensors")
    py!(File.read!(Path.expand("../python/torch_spatial.py", __DIR__)), [path, "3"])
    {:ok, r} = Safetensors.read(path)
    File.rm!(path)
    eval = fn y, lets, env -> Oracle.eval_program(prog(y, lets), env).y end

    convs = %{"c3_p1" => [padding: 1], "c3_s2" => [stride: 2, padding: 1], "c3_d2" => [padding: 2, dilation: 2],
              "c1" => [], "c25" => [stride: {1, 2}, padding: {0, 2}]}

    for {name, o} <- convs do
      {rows, dims} = Spatial.from_nchw(r[name <> ".x"])
      {lets, y, d2} = Spatial.conv2d([], T.input(:x, :f32, rows.shape), dims, name, r[name <> ".w"], r[name <> ".b"], o)
      got = eval.(y, lets, %{x: rows}) |> Spatial.to_nchw(d2) |> Tensor.to_floats()
      assert rel(got, Tensor.to_floats(r[name <> ".y"])) <= @tol, name
    end

    # conv3d over (t, h, w)
    [c3, t3, h3, w3] = r["c3d.x"].shape
    x3 = r["c3d.x"] |> Tensor.to_floats() |> List.to_tuple()
    rows3 = Tensor.from_list(:f32, [t3 * h3 * w3, 16], for(z <- 0..(t3 - 1), y <- 0..(h3 - 1), x <- 0..(w3 - 1), ch <- 0..15,
                                                          do: if(ch < c3, do: elem(x3, ((ch * t3 + z) * h3 + y) * w3 + x), else: 0.0)))
    {lets, y, {to, ho, wo, co}} = Spatial.conv3d([], T.input(:x, :f32, rows3.shape), {t3, h3, w3, c3}, "c3d", r["c3d.w"], r["c3d.b"], stride: {1, 2, 1}, padding: 1)
    out = eval.(y, lets, %{x: rows3}) |> Tensor.to_floats() |> List.to_tuple()
    got = for ch <- 0..(co - 1), z <- 0..(to - 1), yy <- 0..(ho - 1), xx <- 0..(wo - 1), do: elem(out, ((z * ho + yy) * wo + xx) * 16 + ch)
    assert rel(got, Tensor.to_floats(r["c3d.y"])) <= @tol

    {rows, {h, w, c} = d} = Spatial.from_nchw(r["gn.x"])
    {lets, y} = Spatial.group_norm([], T.input(:x, :f32, rows.shape), h * w, c, 6, "gn", r["gn.w"], r["gn.b"], 1.0e-6)
    out = eval.(y, lets, %{x: rows})
    assert rel(Tensor.to_floats(Spatial.to_nchw(out, d)), Tensor.to_floats(r["gn.y"])) <= @tol
    # the padding channels stay exactly zero
    assert out.data |> Vapor.F32.decode() |> Enum.chunk_every(32) |> Enum.flat_map(&Enum.drop(&1, 24)) |> Enum.uniq() == [0]

    {rows, d} = Spatial.from_nchw(r["up.x"])
    {lets, y, d2} = Spatial.upsample_nearest([], T.input(:x, :f32, rows.shape), d, "up")
    assert Tensor.to_floats(Spatial.to_nchw(eval.(y, lets, %{x: rows}), d2)) == Tensor.to_floats(r["up.y"])

    {rows, {h, w, c} = d} = Spatial.from_nchw(r["attn.x"])
    p = &{r["attn.w#{&1}"], r["attn.b#{&1}"]}
    {lets, y} = Spatial.pixel_attention([], T.input(:x, :f32, rows.shape), h * w, c, "attn", p.(0), p.(1), p.(2), p.(3))
    assert rel(Tensor.to_floats(Spatial.to_nchw(eval.(y, lets, %{x: rows}), d)), Tensor.to_floats(r["attn.y"])) <= @tol
  end

  @tag :native
  test "a convolution block is bit-identical on host ISAs, the RVV interpreter and the fabric" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host), threads: 2)
    x = T.input(:x, :f32, [64, 16])
    w1 = Tensor.random(:f32, [12, 5, 3, 3], 2, scale: 0.3)
    {lets, y, d} = Spatial.conv2d([], x, {8, 8, 5}, "a", w1, Tensor.random(:f32, [12], 3, scale: 0.1), padding: 1)
    {lets, y} = Spatial.group_norm(lets, T.silu(y), 64, 12, 4, "n", Tensor.random(:f32, [12], 4), Tensor.random(:f32, [12], 5), 1.0e-6)
    {lets, y, _} = Spatial.conv2d(lets, y, d, "b", Tensor.random(:f32, [7, 12, 3, 3], 6, scale: 0.3), nil, stride: 2, padding: 1)
    {:ok, c} = Lower.lower(prog(y, lets))
    e = %{x: Spatial.from_nchw(Tensor.random(:f32, [5, 8, 8], 7)) |> elem(0)}
    {:ok, ref} = Native.run_oracle(c, e)
    for isa <- Substrates.host_isas(), do: assert(elem(Native.run(wk, c, e, isa: isa, mode: :native), 1).outputs == ref.outputs)
    {:ok, emu} = Native.run(wk, c, e, isa: :riscv64, mode: :emulate, poison: true, vlen: 128)
    assert emu.outputs == ref.outputs

    case Enum.find(Substrates.list(), &(&1.kind == :fabric)) do
      nil -> :ok
      fabric -> assert elem(Dispatch.run_on(fabric, c, e, []), 1).outputs == ref.outputs
    end
  end

  # the decoder of Stable Diffusion's VAE (4 latent channels) and the
  # Flux/SD3 shape (16 latent channels, three blocks, no post-quant conv)
  @tag :diffusers
  @tag :native
  test "AutoencoderKL decoder admitted from diffusers' files: every tensor read, pixels = diffusers" do
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host), threads: 2)

    for variant <- ["sd", "wide"] do
      dir = Path.join(System.tmp_dir!(), "vapor-vae-#{variant}-#{System.unique_integer([:positive])}")
      File.mkdir_p!(dir)
      py!(File.read!(Path.expand("../python/diffusers_vae.py", __DIR__)), [dir, "5", variant])
      {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
      {:ok, m} = Lock.open(dir)
      assert m.spec.family == "autoencoder_kl" and m.spec.interface == :map
      read = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
      decoder = m.weights |> Map.keys() |> Enum.filter(&(is_binary(&1) and String.starts_with?(&1, ["decoder.", "post_quant"])))
      assert Enum.reject(decoder, &MapSet.member?(read, &1)) == []

      [_, h, w] = ref["z"].shape
      {:ok, p} = Lock.build(m.spec, m.weights, latent: {h, w})
      {:ok, c} = Lower.lower(p)
      {:ok, got} = Native.run(wk, c, VAE.input(m.spec, ref["z"]), isa: Substrates.host_isa(), mode: :native)
      img = VAE.image(m.spec, got.outputs.out, {h, w})
      assert img.shape == ref["image"].shape
      err = rel(Tensor.to_floats(img), Tensor.to_floats(ref["image"]))
      IO.puts("\n  AutoencoderKL #{variant}: #{inspect(img.shape)} max |Δ|/max|ref| = #{Float.round(err, 9)}")
      assert err <= 5.0e-6
      File.rm_rf!(dir)
    end
  end

  # the denoiser of latent diffusion: DiT with adaLN-Zero conditioning
  @tag :diffusers
  @tag :native
  test "DiT admitted from diffusers' files: noise prediction = diffusers; position table bit-exact; features to an ulp" do
    dir = Path.join(System.tmp_dir!(), "vapor-dit-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    py!(File.read!(Path.expand("../python/diffusers_dit.py", __DIR__)), [dir, "2"])
    {:ok, ref} = Safetensors.read(Path.join(dir, "reference.safetensors"))
    {:ok, m} = Lock.open(dir)
    assert m.spec.family == "dit"
    read = m.spec |> Lock.expected() |> Enum.map(&elem(&1, 0)) |> MapSet.new()
    assert m.weights |> Map.keys() |> Enum.filter(&is_binary/1) |> Enum.reject(&MapSet.member?(read, &1)) == []

    {:ok, p} = Lock.build(m.spec, m.weights)
    [label] = Tensor.to_list(ref["label"])
    [t] = Tensor.to_list(ref["t"])
    {:ok, c} = Lower.lower(p)
    {:ok, wk} = Worker.start_link(exec: worker_exec(:host))
    {:ok, got} = Native.run(wk, c, Vapor.Lock.Adapters.DiT.input(m.spec, ref["z"], ref["temb"], label), isa: Substrates.host_isa(), mode: :native)
    out = Vapor.Lock.Adapters.DiT.image(m.spec, got.outputs.out)
    assert out.shape == ref["out"].shape
    assert rel(Tensor.to_floats(out), Tensor.to_floats(ref["out"])) <= 2.0e-6

    pos = Enum.find_value(p.lets, fn {n, {:const, tt}} -> if String.starts_with?(to_string(n), "$dit.pos"), do: tt; _ -> nil end)
    assert pos.data == ref["pos"].data
    assert rel(Tensor.to_floats(Vapor.Lock.Adapters.DiT.timestep_features(t)), Tensor.to_floats(ref["temb"])) <= 2.0e-6
    File.rm_rf!(dir)
  end
end
