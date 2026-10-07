defmodule Vapor.ZKTest do
  @moduledoc """
  Integer inference as R1CS (`Vapor.ZK`): the witness computed by the
  certified int8 kernels satisfies the circuit; every wire is determined
  (perturbing any one breaks a constraint); a wrong output has no witness;
  values that could reach p/2 are refused; and — with snarkjs — the files
  are accepted by the iden3 tooling and a Groth16 proof verifies, while a
  forged public output does not; the exported Solidity verifier compiles.
  """
  use ExUnit.Case, async: false
  alias Vapor.{Field, ZK}

  @moduletag timeout: 900_000

  defp rnd(n, m, lo, hi), do: for(_ <- 1..n, do: for(_ <- 1..m, do: lo + :rand.uniform(hi - lo + 1) - 1))

  defp model do
    :rand.seed(:exsss, {4, 2, 1})
    w1 = rnd(8, 16, -128, 127)
    b1 = hd(rnd(1, 8, -3000, 3000))
    w2 = rnd(3, 8, -50, 50)
    b2 = hd(rnd(1, 3, -100, 100))
    [{:linear, w1, b1}, :relu, {:linear, w2, b2}]
  end

  defp x, do: hd(rnd(1, 16, -128, 127))

  defp exact([{:linear, w1, b1}, :relu, {:linear, w2, b2}], x) do
    dot = fn r, v -> Enum.zip_with(r, v, &(&1 * &2)) |> Enum.sum() end
    h = Enum.zip_with(w1, b1, fn r, b -> max(b + dot.(r, x), 0) end)
    Enum.zip_with(w2, b2, fn r, b -> b + dot.(r, h) end)
  end

  @tag :native
  test "the certified kernel's witness satisfies the circuit; every wire is determined" do
    m = model()
    c = ZK.compile(m, k: 16)
    xs = x()
    {:ok, wit, out} = ZK.witness(c, xs)
    assert out == exact(m, xs)
    assert ZK.check(c, wit) == :ok
    # one constraint per output; B + 3 per ReLU neuron; 9 per private int8 input
    assert ZK.size(c) == 3 + 9 * 16 + Enum.sum(for {:relu, gs} <- c.plan, {_, nb, _} <- gs, do: nb + 3)
    IO.puts("\n  R1CS: #{ZK.size(c)} constraints, #{c.n_wires} wires for a 16→8→3 int8 network")

    # no wire is free: changing any single value (but the constant) breaks the circuit
    for i <- 1..(c.n_wires - 1) do
      bad = List.update_at(wit, i, &Field.add(c.field, &1, 1))
      assert {:error, {:constraint, _}} = ZK.check(c, bad), "wire #{i} is under-constrained"
    end
  end

  test "a wrong output has no witness; bounds against p/2 are enforced" do
    m = model()
    c = ZK.compile(m, k: 16)
    {:ok, wit, _} = ZK.witness(c, x(), certified: false)
    assert ZK.check(c, wit) == :ok
    forged = List.update_at(wit, 1, &Field.add(c.field, &1, 1))
    assert {:error, _} = ZK.check(c, forged)

    # an input outside int8 has no witness (without the range check, any
    # hidden activation — hence any output — would be reachable)
    {:ok, out_of_range, _} = ZK.witness(c, [300 | tl(x())], certified: false, unchecked: true)
    assert {:error, {:constraint, _}} = ZK.check(c, out_of_range)

    # in a small field, a ReLU whose range would wrap is refused
    assert_raise ArgumentError, ~r/bits/, fn ->
      ZK.compile([{:linear, [List.duplicate(127, 16)], [999_000_000]}, :relu, {:linear, [[1]], [0]}], k: 16, field: :babybear)
    end

    # in BabyBear (p ≈ 2³¹) a large enough contraction no longer fits below p/2
    big = [{:linear, [List.duplicate(127, 70_000)], [0]}]
    assert_raise ArgumentError, ~r/p\/2/, fn -> ZK.compile(big, k: 70_000, field: :babybear) end
    assert %ZK{} = ZK.compile(big, k: 70_000, field: :bn254)
  end

  test "binary formats" do
    c = ZK.compile(model(), k: 16)
    {:ok, wit, _} = ZK.witness(c, x(), certified: false)
    <<"r1cs", 1::little-32, 3::little-32, 1::little-32, hl::little-64, header::binary-size(hl), _::binary>> = ZK.r1cs(c)
    <<32::little-32, p::little-256, nw::little-32, nout::little-32, 0::little-32, 16::little-32, _::binary>> = header
    assert p == Field.get(:bn254).p and nw == c.n_wires and nout == 3
    <<"wtns", 2::little-32, 2::little-32, _::binary>> = ZK.wtns(c, wit)
    assert byte_size(ZK.wtns(c, wit)) == 12 + 12 + 40 + 12 + 32 * length(wit)
    assert ZK.digest(c) != ZK.digest(ZK.compile(List.replace_at(model(), 2, {:linear, [[1 | List.duplicate(0, 7)]], [0]}), k: 16))
  end

  @tag :snarkjs
  test "snarkjs: the files are accepted, a Groth16 proof verifies, a forged output does not; Solidity verifier compiles" do
    dir = Path.join(System.tmp_dir!(), "vapor-zk-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf!(dir) end)
    node_modules = System.fetch_env!("VAPOR_SNARKJS")
    snark = fn args -> System.cmd("node", [Path.join(node_modules, "snarkjs/build/cli.cjs") | args], cd: dir, stderr_to_stdout: true) end

    m = model()
    c = ZK.compile(m, k: 16)
    {:ok, wit, out} = ZK.witness(c, x())
    ZK.write_r1cs(c, Path.join(dir, "m.r1cs"))
    ZK.write_wtns(c, wit, Path.join(dir, "m.wtns"))

    {o, 0} = snark.(["r1cs", "info", "m.r1cs"])
    assert o =~ "# of Constraints: #{ZK.size(c)}"
    {o, 0} = snark.(["wtns", "check", "m.r1cs", "m.wtns"])
    assert o =~ "WITNESS IS CORRECT"

    # a toy ceremony (insecure, for the test): powers of tau, phase 2, key
    power = max(8, ceil(:math.log2(ZK.size(c) + c.n_wires)) + 1)
    for args <- [["powersoftau", "new", "bn128", "#{power}", "p0.ptau"],
                 ["powersoftau", "contribute", "p0.ptau", "p1.ptau", "--name=t", "-e=vapor"],
                 ["powersoftau", "prepare", "phase2", "p1.ptau", "pf.ptau"],
                 ["groth16", "setup", "m.r1cs", "pf.ptau", "m0.zkey"],
                 ["zkey", "contribute", "m0.zkey", "m.zkey", "--name=t", "-e=vapor"],
                 ["zkey", "export", "verificationkey", "m.zkey", "vk.json"],
                 ["groth16", "prove", "m.zkey", "m.wtns", "proof.json", "public.json"]] do
      {o, code} = snark.(args)
      assert code == 0, "snarkjs #{Enum.join(args, " ")}: #{o}"
    end

    # the public signals are the outputs, as integers in the field
    pub = File.read!(Path.join(dir, "public.json")) |> Vapor.JSON.decode!() |> Enum.map(&String.to_integer/1)
    assert Enum.map(pub, &Field.to_int(c.field, &1)) == out
    {o, 0} = snark.(["groth16", "verify", "vk.json", "public.json", "proof.json"])
    assert o =~ "OK!"

    # the same proof for another output: rejected
    forged = [Field.from_int(c.field, hd(out) + 1) | tl(pub)] |> Enum.map(&Integer.to_string/1)
    File.write!(Path.join(dir, "forged.json"), Vapor.JSON.encode(forged))
    {o, _} = snark.(["groth16", "verify", "vk.json", "forged.json", "proof.json"])
    refute o =~ "OK!"

    {_, 0} = snark.(["zkey", "export", "solidityverifier", "m.zkey", "Verifier.sol"])
    {o, code} = System.cmd("node", [Path.join(node_modules, "solc/solc.js"), "--bin", "--abi", "-o", "sol", "Verifier.sol"], cd: dir, stderr_to_stdout: true)
    assert code == 0, o
    [bin] = Path.wildcard(Path.join(dir, "sol/*Groth16Verifier.bin"))
    IO.puts("  Groth16: proof verified by snarkjs; Solidity verifier compiled (#{div(byte_size(String.trim(File.read!(bin))), 2)} bytes of EVM code)")

    # gas, measured: the verifier deployed in an in-memory EVM (ethereumjs)
    if File.dir?(Path.join(node_modules, "@ethereumjs/evm")) and File.dir?(Path.join(node_modules, "ethers")) do
      {cd, 0} = snark.(["zkey", "export", "soliditycalldata", "public.json", "proof.json"])
      File.write!(Path.join(dir, "calldata.txt"), cd)
      # ES modules resolve packages from the script's own directory
      script = Path.join(Path.dirname(node_modules), "vapor_groth16_gas.mjs")
      File.cp!(Path.expand("../js/groth16_gas.mjs", __DIR__), script)
      {o, 0} = System.cmd("node", [script, dir], stderr_to_stdout: true)
      r = o |> String.split("\n", trim: true) |> List.last() |> Vapor.JSON.decode!()
      assert r["valid"]["ok"] == true and r["forged"] == false
      IO.puts("  on-chain verification: #{r["valid"]["exec"]} gas of execution, #{r["valid"]["total"]} with the transaction (#{r["public_inputs"]} public inputs)")
    end
  end
end
