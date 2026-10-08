defmodule Vapor.QalibTest do
  use ExUnit.Case, async: true
  alias Vapor.{Qalib, Rebis}

  @full_adder """
  input a b cin
  output s cout
  s = a ^ b ^ cin
  cout = maj(a, b, cin)
  """

  defp spec!(text), do: (fn {:ok, c} -> c end).(Rebis.parse(text))

  test "map a circuit to sky130 cells, write Verilog, read it back: the same function, proved" do
    spec = spec!(@full_adder)

    for style <- [:cells, :nand] do
      {:ok, v, stats} = Qalib.to_verilog(spec, style: style, module: "fa")
      assert v =~ "module fa (a, b, cin, s, cout);"
      assert v =~ "sky130_fd_sc_hd__"
      if style == :nand, do: assert(Map.keys(stats.by_type) -- ["nand2", "inv"] == [])
      assert {:equivalent, %{method: :truth_table}} = Qalib.certify(spec, v)
    end
  end

  test "a Yosys-style netlist with complex cells is checked against the specification" do
    netlist = """
    // full adder as an ABC mapping might write it
    module fa(a, b, cin, s, cout);
      input a; input b; input cin;
      output s; output cout;
      wire _0_, _1_, _2_;
      sky130_fd_sc_hd__xor2_1 _3_ (.A(a), .B(b), .X(_0_));
      sky130_fd_sc_hd__xor2_2 _4_ (.A(_0_), .B(cin), .X(s));
      sky130_fd_sc_hd__nand2_1 _5_ (.A(a), .B(b), .Y(_1_));
      sky130_fd_sc_hd__nand2_1 _6_ (.A(_0_), .B(cin), .Y(_2_));
      sky130_fd_sc_hd__nand2_4 _7_ (.A(_1_), .B(_2_), .Y(cout));
    endmodule
    """

    assert {:equivalent, _} = Qalib.certify(@full_adder, netlist)
    # the same cells, one pin swapped: a real bug, found with the input that shows it
    wrong = String.replace(netlist, ".B(cin), .Y(_2_)", ".B(a), .Y(_2_)")
    assert {:different, %{counterexample: cex}} = Qalib.certify(@full_adder, wrong)
    assert map_size(cex) == 3
  end

  test "complex cells match their sky130 definitions (each against its Boolean formula)" do
    cases = [
      {"a21oi", ".A1(a), .A2(b), .B1(c), .Y(y)", "y = ~((a & b) | c)"},
      {"o21ai", ".A1(a), .A2(b), .B1(c), .Y(y)", "y = ~((a | b) & c)"},
      {"maj3", ".A(a), .B(b), .C(c), .X(y)", "y = maj(a, b, c)"},
      {"mux2", ".A0(a), .A1(b), .S(c), .X(y)", "y = mux(c, b, a)"},
      {"nand2b", ".A_N(a), .B(b), .Y(y)", "y = ~(~a & b)"},
      {"xnor3", ".A(a), .B(b), .C(c), .X(y)", "y = ~(a ^ b ^ c)"}
    ]

    for {cell, pins, formula} <- cases do
      v = "module m(a, b, c, y); input a, b, c; output y; sky130_fd_sc_hd__#{cell}_1 u (#{pins}); endmodule"
      spec = "input a b c\noutput y\n#{formula}\n"
      assert {:equivalent, _} = Qalib.certify(spec, v), cell
    end
  end

  # a ripple-carry adder over n bits, in Rebis's netlist language and in Verilog (gate primitives)
  defp adder_spec(n) do
    ins = for i <- 0..(n - 1), x <- ["a", "b"], do: "#{x}#{i}"
    body = for i <- 0..(n - 1) do
      c = if i == 0, do: "cin", else: "c#{i}"
      "s#{i} = a#{i} ^ b#{i} ^ #{c}\nc#{i + 1} = maj(a#{i}, b#{i}, #{c})"
    end
    "input #{Enum.join(ins, " ")} cin\noutput #{Enum.map_join(0..(n - 1), " ", &"s#{&1}")} c#{n}\n" <> Enum.join(body, "\n") <> "\n"
  end

  defp adder_verilog(n, trojan) do
    ins = Enum.flat_map(0..(n - 1), &["a#{&1}", "b#{&1}"]) ++ ["cin"]
    outs = Enum.map(0..(n - 1), &"s#{&1}") ++ ["c#{n}"]
    gates = for i <- 0..(n - 1) do
      c = if i == 0, do: "cin", else: "c#{i}"
      "  xor (p#{i}, a#{i}, b#{i});\n  xor (t#{i}, p#{i}, #{c});\n  and (g#{i}, a#{i}, b#{i});\n  and (h#{i}, p#{i}, #{c});\n  or (c#{i + 1}, g#{i}, h#{i});\n"
    end

    # the trojan: bit 0 of the sum flips when a = 1010110101 and b = 0101001010 — 2²⁰ patterns hide it
    trig = if trojan do
      pat = for i <- 0..(n - 1), do: {"a#{i}", rem(div(0x2B5, Integer.pow(2, i)), 2)}
      pbt = for i <- 0..(n - 1), do: {"b#{i}", rem(div(0x14A, Integer.pow(2, i)), 2)}
      lits = Enum.map(pat ++ pbt, fn {x, 1} -> x; {x, 0} -> "~#{x}" end)
      "  assign trig = #{Enum.join(lits, " & ")};\n  assign s0 = t0 ^ trig;\n"
    else
      "  assign s0 = t0;\n"
    end

    rest = Enum.map(1..(n - 1), &"  assign s#{&1} = t#{&1};\n")
    "module add(#{Enum.join(ins ++ outs, ", ")});\n  input #{Enum.join(ins, ", ")};\n  output #{Enum.join(outs, ", ")};\n" <>
      Enum.join(gates) <> trig <> Enum.join(rest) <> "endmodule\n"
  end

  test "beyond the truth table: a 21-input adder proved equal by SAT with a checked DRUP proof; a hidden trigger is found" do
    spec = spec!(adder_spec(10))
    assert {:equivalent, %{method: :sat, checked_lemmas: k}} = Qalib.certify(spec, adder_verilog(10, false))
    assert k > 0
    assert {:different, %{counterexample: cex, method: :sat}} = Qalib.certify(spec, adder_verilog(10, true))
    a = Enum.reduce(0..9, 0, fn i, acc -> acc + cex["a#{i}"] * Integer.pow(2, i) end)
    b = Enum.reduce(0..9, 0, fn i, acc -> acc + cex["b#{i}"] * Integer.pow(2, i) end)
    assert {a, b} == {0x2B5, 0x14A}
  end

  test "BLIF: on-set and off-set covers, don't-cares, constants and gates" do
    blif = """
    .model fa
    .inputs a b cin
    .outputs s cout one
    .names a b p
    10 1
    01 1
    .names p cin s
    00 0
    11 0
    .names a b cin cout
    11- 1
    1-1 1
    -11 1
    .names one
    1
    .end
    """

    spec = "input a b cin\noutput s cout one\ns = a ^ b ^ cin\ncout = maj(a, b, cin)\none = 1\n"
    assert {:equivalent, _} = Qalib.certify(spec, blif)
    gate = ".model g\n.inputs a b\n.outputs y\n.gate sky130_fd_sc_hd__nor2_1 A=a B=b Y=y\n.end\n"
    assert {:equivalent, _} = Qalib.certify("input a b\noutput y\ny = ~(a | b)\n", gate)
  end

  test "buses, bit selects and concatenations" do
    v = """
    module sw(input [1:0] a, input s, output [1:0] y);
      assign y = {a[0], a[1]};
    endmodule
    """

    {:ok, c} = Qalib.read_verilog(v)
    assert c.inputs == ["a[1]", "a[0]", "s"]
    assert Enum.map(c.outputs, &elem(&1, 0)) == ["y[1]", "y[0]"]
    same = "module sw2(a, s, y); input [1:0] a; input s; output [1:0] y; buf (y[1], a[0]); buf b0 (y[0], a[1]); endmodule"
    assert {:equivalent, _} = Qalib.certify(v, same)
    crossed = String.replace(same, "buf (y[1], a[0])", "buf (y[1], a[1])")
    assert {:different, _} = Qalib.certify(v, crossed)
  end

  test "refusals: unknown cells, latches, loops, undriven and doubly driven nets, wide constants" do
    assert {:error, why} = Qalib.read_verilog("module m(a, y); input a; output y; my_secret_cell u (.A(a), .Y(y)); endmodule")
    assert why =~ "unknown cell"
    assert {:error, w2} = Qalib.read_blif(".model l\n.inputs d\n.outputs q\n.latch d q 0\n.end\n")
    assert w2 =~ "combinational"
    assert {:error, w3} = Qalib.read_verilog("module m(a, y); input a; output y; wire w; and (w, a, y); buf (y, w); endmodule")
    assert w3 =~ "loop"
    assert {:error, w4} = Qalib.read_verilog("module m(a, y); input a; output y; endmodule")
    assert w4 =~ "never driven"
    assert {:error, w5} = Qalib.read_verilog("module m(a, y); input a; output y; buf (y, a); not (y, a); endmodule")
    assert w5 =~ "driven twice"
    assert {:error, w6} = Qalib.read_verilog("module m(y); output y; assign y = 2'b01; endmodule")
    assert w6 =~ "bits wide"
  end

  test "escaped identifiers round-trip through the writer" do
    {:ok, c} = Qalib.read_verilog("module m(input [1:0] a, output y); assign y = a[1] & ~a[0]; endmodule")
    {:ok, v, _} = Qalib.to_verilog(c)
    assert v =~ "\\a[1] "
    assert {:equivalent, _} = Qalib.certify(c, v)
  end
end
