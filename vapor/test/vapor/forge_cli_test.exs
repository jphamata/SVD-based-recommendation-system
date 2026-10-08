defmodule Vapor.ForgeCliTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO
  alias Vapor.Main

  setup do
    dir = Path.join(System.tmp_dir!(), "vapor_forge_#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    System.put_env("VAPOR_TTY", "0")
    on_exit(fn -> File.rm_rf!(dir); System.delete_env("VAPOR_TTY") end)
    %{dir: dir}
  end

  defp file(dir, name, text), do: (p = Path.join(dir, name); File.write!(p, text); p)

  test "vapor logic: integer and causal claims with exit status; check decides a proposal", %{dir: d} do
    k = file(d, "k.lp", "maximize 8a + 11b + 6c + 4d\n5a + 7b + 4c + 3d <= 14\nbin a, b, c, d\n")
    out = capture_io(fn -> assert Main.run(["logic", k]) == 0 end)
    assert Vapor.JSON.decode!(out)["objective"]["exact"] == "21"
    bow = file(d, "bow.txt", "causal\nx -> y\nx <-> y\nidentify y | do(x)\n")
    assert capture_io(fn -> assert Main.run(["logic", bow]) == 1 end) =~ "not identifiable"
    prop = file(d, "p.json", ~s({"incumbent": {"a": "0", "b": "1", "c": "1", "d": "1"}, "objective": "22", "tree": {"leaf": "bound", "y": ["1"]}}))
    assert capture_io(fn -> assert Main.run(["logic", "check", k, prop]) == 1 end) =~ "accepted"
  end

  test "vapor qalib: map writes a proved netlist; check tells a different one apart", %{dir: d} do
    spec = file(d, "fa.net", "input a b cin\noutput s cout\ns = a ^ b ^ cin\ncout = maj(a, b, cin)\n")
    v = capture_io(fn -> capture_io(:stderr, fn -> assert Main.run(["qalib", "map", spec, "--style", "nand"]) == 0 end) end)
    assert v =~ "sky130_fd_sc_hd__nand2_1"
    mapped = file(d, "fa.v", v)
    assert capture_io(fn -> assert Main.run(["qalib", "check", spec, mapped]) == 0 end) =~ "equivalent"
    broken = file(d, "bad.v", String.replace(v, ".A(a), .B(b)", ".A(a), .B(a)", global: false))
    assert capture_io(fn -> assert Main.run(["qalib", "check", spec, broken]) == 1 end) =~ "different"
  end

  test "vapor recommend: a planted table is signal (0), an unstructured one is not (1)", %{dir: d} do
    users = 30
    items = 20
    header = "user," <> Enum.map_join(0..(items - 1), ",", &"i#{&1}")
    row = fn u, f -> "u#{u}," <> Enum.map_join(0..(items - 1), ",", fn i -> if :erlang.phash2({u, i}, 10) < 6, do: f.(u, i), else: "" end) end
    planted = fn u, i -> Float.to_string(3.0 + :math.cos(u * 0.7) * :math.cos(i * 0.9) * 2 + :math.sin(u * 1.3) * :math.sin(i * 0.4) * 2) end
    flat = fn u, i -> Float.to_string(3.0 + rem(u, 3) * 0.2 + :erlang.phash2({u, i, 7}, 100) / 100) end
    good = file(d, "good.csv", Enum.join([header | Enum.map(0..(users - 1), &row.(&1, planted))], "\n"))
    noisy = file(d, "flat.csv", Enum.join([header | Enum.map(0..(users - 1), &row.(&1, flat))], "\n"))
    assert Vapor.JSON.decode!(capture_io(fn -> assert Main.run(["recommend", good]) == 0 end))["verdict"] == "signal"
    refute Vapor.JSON.decode!(capture_io(fn -> assert Main.run(["recommend", noisy]) == 1 end))["verdict"] == "signal"
  end

  test "usage and the console's jail", %{dir: _d} do
    assert capture_io(fn -> assert Main.run(["palingenesis"]) == 0 end) =~ "plank by plank"
    assert capture_io(:stderr, fn -> assert Main.run(["palingenesis", "try", "x"]) == 2 end) =~ "--plank"
    Process.put(:vapor_jail, %{})
    assert capture_io(:stderr, fn -> assert Main.run(["palingenesis", "planks", "/etc"]) == 2 end) =~ "not read"
    Process.delete(:vapor_jail)
    assert "qalib" in Main.verbs() and "palingenesis" in Main.verbs()
  end
end
