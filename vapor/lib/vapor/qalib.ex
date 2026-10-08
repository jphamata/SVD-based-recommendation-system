defmodule Vapor.Qalib do
  @moduledoc """
  **Qālib** (القالب, the mould: where the metal takes its shape) — the
  bridge from a Boolean term to a standard-cell netlist and back, with
  the equivalence of every step **proved** by `Vapor.Rebis`
  (docs/QALIB.md).

  The proposal it answers asked vapor to emit GDSII masks itself and
  replace "proprietary EDA". Both premises fail on inspection. The open
  SkyWater 130 nm flow is not proprietary: Yosys (synthesis), OpenROAD
  (placement and routing), Magic and KLayout (layout, DRC, LVS) are open
  source. And layout is the part of the flow where a home-grown tool
  would be least trustworthy: a GDS that has not passed the foundry's
  design-rule and layout-versus-schematic checks is not a chip. What the
  flow lacks, and what vapor has, is a **checker** that does not trust the
  flow's own tools. So Qālib does three things:

    * **reads** what the flow produces: structural Verilog (gate
      primitives, `assign` with `~ & | ^ ?:`, buses and bit selects, and
      instances of the `sky130_fd_sc_hd` cells in the table below with
      named pins) and BLIF (`.names` covers and `.gate` cells), the
      formats Yosys and ABC write;
    * **maps** a vapor circuit onto those cells (`style: :cells`, one cell
      per gate, or `style: :nand`, NAND2 and inverters only) and writes
      structural Verilog that Yosys and OpenROAD read;
    * **certifies** that two netlists compute the same function:
      `certify/3` is `Vapor.Rebis.equivalent/3`, by truth table up to 16
      inputs and by a SAT miter with a DRUP proof checked by separate code
      beyond. A difference comes with a shrunk counterexample.

  Used between every pair of steps (specification → mapped netlist →
  Yosys's output → the netlist Magic extracts from the layout), this is
  *translation validation* of the whole flow: a hardware trojan inserted
  by any tool, or a synthesis bug, shows up as a counterexample, however
  rare its trigger (the 32-bit trigger of `docs/REBIS.md` is the measured
  case).

  Not done here, stated: sequential logic (latches and flip-flops are
  refused: equivalence of state machines needs induction, `docs/TODO.md`),
  timing, power and area (the liberty files hold them; a mapped netlist's
  cell counts are reported, not its µm²), placement, routing and GDSII.
  """
  alias Vapor.Rebis
  alias Vapor.Rebis.Circuit

  @max_nets 500_000

  # ------------------------------------------------------- the cell table

  # sky130_fd_sc_hd cells by base name (drive strength `_N` stripped): output pin → function of the input pins
  @cells %{
    "inv" => [{"Y", {:not, {:pin, "A"}}}],
    "clkinv" => [{"Y", {:not, {:pin, "A"}}}],
    "buf" => [{"X", {:pin, "A"}}],
    "clkbuf" => [{"X", {:pin, "A"}}],
    "nand2" => [{"Y", {:not, {:and, [{:pin, "A"}, {:pin, "B"}]}}}],
    "nand3" => [{"Y", {:not, {:and, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}]}}}],
    "nand4" => [{"Y", {:not, {:and, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}, {:pin, "D"}]}}}],
    "nor2" => [{"Y", {:not, {:or, [{:pin, "A"}, {:pin, "B"}]}}}],
    "nor3" => [{"Y", {:not, {:or, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}]}}}],
    "nor4" => [{"Y", {:not, {:or, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}, {:pin, "D"}]}}}],
    "and2" => [{"X", {:and, [{:pin, "A"}, {:pin, "B"}]}}],
    "and3" => [{"X", {:and, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}]}}],
    "and4" => [{"X", {:and, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}, {:pin, "D"}]}}],
    "or2" => [{"X", {:or, [{:pin, "A"}, {:pin, "B"}]}}],
    "or3" => [{"X", {:or, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}]}}],
    "or4" => [{"X", {:or, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}, {:pin, "D"}]}}],
    "xor2" => [{"X", {:xor, [{:pin, "A"}, {:pin, "B"}]}}],
    "xor3" => [{"X", {:xor, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}]}}],
    "xnor2" => [{"Y", {:not, {:xor, [{:pin, "A"}, {:pin, "B"}]}}}],
    "xnor3" => [{"X", {:not, {:xor, [{:pin, "A"}, {:pin, "B"}, {:pin, "C"}]}}}],
    "nand2b" => [{"Y", {:not, {:and, [{:not, {:pin, "A_N"}}, {:pin, "B"}]}}}],
    "nor2b" => [{"Y", {:not, {:or, [{:pin, "A"}, {:not, {:pin, "B_N"}}]}}}],
    "and2b" => [{"X", {:and, [{:not, {:pin, "A_N"}}, {:pin, "B"}]}}],
    "or2b" => [{"X", {:or, [{:pin, "A"}, {:not, {:pin, "B_N"}}]}}],
    "mux2" => [{"X", {:mux, {:pin, "S"}, {:pin, "A1"}, {:pin, "A0"}}}],
    "mux2i" => [{"Y", {:not, {:mux, {:pin, "S"}, {:pin, "A1"}, {:pin, "A0"}}}}],
    "maj3" => [{"X", {:or, [{:and, [{:pin, "A"}, {:pin, "B"}]}, {:and, [{:pin, "A"}, {:pin, "C"}]}, {:and, [{:pin, "B"}, {:pin, "C"}]}]}}],
    "a21o" => [{"X", {:or, [{:and, [{:pin, "A1"}, {:pin, "A2"}]}, {:pin, "B1"}]}}],
    "a21oi" => [{"Y", {:not, {:or, [{:and, [{:pin, "A1"}, {:pin, "A2"}]}, {:pin, "B1"}]}}}],
    "o21a" => [{"X", {:and, [{:or, [{:pin, "A1"}, {:pin, "A2"}]}, {:pin, "B1"}]}}],
    "o21ai" => [{"Y", {:not, {:and, [{:or, [{:pin, "A1"}, {:pin, "A2"}]}, {:pin, "B1"}]}}}],
    "a22oi" => [{"Y", {:not, {:or, [{:and, [{:pin, "A1"}, {:pin, "A2"}]}, {:and, [{:pin, "B1"}, {:pin, "B2"}]}]}}}],
    "o22ai" => [{"Y", {:not, {:and, [{:or, [{:pin, "A1"}, {:pin, "A2"}]}, {:or, [{:pin, "B1"}, {:pin, "B2"}]}]}}}],
    "a211oi" => [{"Y", {:not, {:or, [{:and, [{:pin, "A1"}, {:pin, "A2"}]}, {:pin, "B1"}, {:pin, "C1"}]}}}],
    "o211ai" => [{"Y", {:not, {:and, [{:or, [{:pin, "A1"}, {:pin, "A2"}]}, {:pin, "B1"}, {:pin, "C1"}]}}}],
    "a31oi" => [{"Y", {:not, {:or, [{:and, [{:pin, "A1"}, {:pin, "A2"}, {:pin, "A3"}]}, {:pin, "B1"}]}}}],
    "o31ai" => [{"Y", {:not, {:and, [{:or, [{:pin, "A1"}, {:pin, "A2"}, {:pin, "A3"}]}, {:pin, "B1"}]}}}],
    "conb" => [{"HI", {:const, 1}}, {"LO", {:const, 0}}]
  }

  @lib "sky130_fd_sc_hd__"

  @doc "The cells Qālib reads, by base name (any drive strength): `%{name => [output pins]}`."
  def cells, do: Map.new(@cells, fn {k, outs} -> {k, Enum.map(outs, &elem(&1, 0))} end)

  # --------------------------------------------------------------- Verilog

  @doc """
  Read a structural Verilog module (the first one in the text) into a
  combinational `Vapor.Rebis.Circuit`. `{:ok, circuit}` or `{:error, why}`.
  """
  def read_verilog(text) when is_binary(text) do
    with {:ok, toks} <- vlex(strip_comments(text)),
         {:ok, m} <- vmodule(toks) do
      build(m)
    end
  catch
    {:qalib, why} -> {:error, why}
  end

  defp strip_comments(t), do: t |> String.replace(~r{/\*.*?\*/}s, " ") |> String.replace(~r{//[^\n]*}, "") |> String.replace(~r/\(\*.*?\*\)/s, " ")

  # tokens: {:id, name} | {:num, 0 | 1 | integer} | {:sym, s}
  defp vlex(text), do: vlex(text, [])
  defp vlex("", acc), do: {:ok, Enum.reverse(acc)}
  defp vlex(<<c, r::binary>>, acc) when c in [?\s, ?\t, ?\r, ?\n], do: vlex(r, acc)

  defp vlex("\\" <> r, acc) do
    [name | rest] = String.split(r, ~r/\s/, parts: 2)
    vlex(Enum.at(rest, 0, ""), [{:id, name} | acc])
  end

  defp vlex(text, acc) do
    cond do
      m = Regex.run(~r/^(\d+)'[bBhHdD]([0-9a-fA-F]+)/, text) -> [all, w, v] = m; vlex(rest(text, all), [constant(w, v) | acc])
      m = Regex.run(~r/^\d+/, text) -> [all] = m; vlex(rest(text, all), [{:int, String.to_integer(all)} | acc])
      m = Regex.run(~r/^[A-Za-z_][A-Za-z0-9_$]*/, text) -> [all] = m; vlex(rest(text, all), [{:id, all} | acc])
      m = Regex.run(~r/^[(),;.\[\]:=~&|^!?{}]/, text) -> [all] = m; vlex(rest(text, all), [{:sym, all} | acc])
      true -> throw({:qalib, "cannot read Verilog at #{inspect(String.slice(text, 0, 20))}"})
    end
  end

  defp rest(text, all), do: binary_part(text, byte_size(all), byte_size(text) - byte_size(all))

  # one-bit constants only: a wider literal is a bus, written bit by bit in a gate-level netlist
  defp constant("1", v) when v in ["0", "1"], do: {:num, String.to_integer(v)}
  defp constant(w, v), do: throw({:qalib, "the constant #{w}'…#{v} is #{w} bits wide: gate-level netlists use one-bit constants"})

  # the module: header, then statements until endmodule
  defp vmodule([{:id, "module"}, {:id, name} | toks]) do
    {ports, toks} = header(toks)
    st = %{name: name, inputs: [], outputs: [], wires: MapSet.new(), widths: %{}, defs: [], insts: 0}
    st = Enum.reduce(ports, st, fn {dir, range, n}, st -> declare(st, dir, range, n) end)
    {:ok, statements(toks, st)}
  end

  defp vmodule([_ | toks]), do: vmodule(toks)
  defp vmodule([]), do: {:error, "no module"}

  defp header([{:sym, "("} | toks]), do: ports(toks, nil, nil, [])
  defp header([{:sym, ";"} | toks]), do: {[], toks}
  defp header(_), do: throw({:qalib, "module header: expected ( or ;"})

  # ANSI ports carry a direction (and range) that sticks until changed; plain ports carry none
  defp ports([{:sym, ")"}, {:sym, ";"} | toks], _dir, _rng, acc), do: {Enum.reverse(acc), toks}
  defp ports([{:sym, ","} | toks], dir, rng, acc), do: ports(toks, dir, rng, acc)
  defp ports([{:id, d} | toks], _dir, _rng, acc) when d in ["input", "output"] do
    toks = skip_kind(toks)
    {rng, toks} = range(toks)
    ports(toks, d, rng, acc)
  end
  defp ports([{:id, n} | toks], dir, rng, acc), do: ports(toks, dir, rng, if(dir, do: [{dir, rng, n} | acc], else: acc))
  defp ports(_, _, _, _), do: throw({:qalib, "module header: unexpected token"})

  defp skip_kind([{:id, k} | toks]) when k in ["wire", "logic", "reg"], do: toks
  defp skip_kind(toks), do: toks

  defp range([{:sym, "["}, {:int, a}, {:sym, ":"}, {:int, b}, {:sym, "]"} | toks]), do: {{a, b}, toks}
  defp range(toks), do: {nil, toks}

  defp declare(st, dir, rng, n) do
    bits = bits(n, rng)
    st = %{st | widths: if(rng, do: Map.put(st.widths, n, bits), else: st.widths)}

    case dir do
      "input" -> %{st | inputs: st.inputs ++ bits}
      "output" -> %{st | outputs: st.outputs ++ bits}
      _ -> %{st | wires: MapSet.union(st.wires, MapSet.new(bits))}
    end
  end

  defp bits(n, nil), do: [n]
  defp bits(n, {a, b}), do: for(i <- a..b//if(a >= b, do: -1, else: 1), do: "#{n}[#{i}]")

  defp statements([{:id, "endmodule"} | _], st), do: st
  defp statements([], _st), do: throw({:qalib, "no endmodule"})

  defp statements([{:id, d} | toks], st) when d in ["input", "output", "wire"] do
    toks = skip_kind(toks)
    {rng, toks} = range(toks)
    {names, toks} = id_list(toks, [])
    statements(toks, Enum.reduce(names, st, &declare(&2, d, rng, &1)))
  end

  defp statements([{:id, "assign"} | toks], st) do
    {lhs, toks} = lvalue(toks, st)
    toks = expect(toks, "=")
    {rhs, toks} = expr(toks, st)
    toks = expect(toks, ";")
    rhs = List.wrap(rhs)
    if length(lhs) != length(rhs), do: throw({:qalib, "assign: #{length(lhs)} bits on the left, #{length(rhs)} on the right"})
    statements(toks, %{st | defs: st.defs ++ Enum.zip(lhs, rhs)})
  end

  defp statements([{:id, g} | toks], st) when g in ~w(and or xor nand nor xnor not buf) do
    {toks, _name} = case toks do [{:id, n} | t] -> {t, n}; t -> {t, nil} end
    toks = expect(toks, "(")
    {args, toks} = arg_list(toks, st, [])
    toks = expect(toks, ";")
    [out | ins] = args
    out = single(out)
    ins = Enum.map(ins, &single/1)

    e =
      case g do
        "and" -> {:and, ins}
        "or" -> {:or, ins}
        "xor" -> {:xor, ins}
        "nand" -> {:not, {:and, ins}}
        "nor" -> {:not, {:or, ins}}
        "xnor" -> {:not, {:xor, ins}}
        "not" -> {:not, hd(ins)}
        "buf" -> hd(ins)
      end

    statements(toks, %{st | defs: st.defs ++ [{net_name(out), e}]})
  end

  defp statements([{:id, cell}, {:id, _inst}, {:sym, "("} | toks], st) do
    base = cell |> String.replace_prefix(@lib, "") |> String.replace(~r/_\d+$/, "")
    outs = Map.get(@cells, base) || throw({:qalib, "unknown cell #{cell}: Qālib reads the sky130_fd_sc_hd cells #{@cells |> Map.keys() |> Enum.sort() |> Enum.join(", ")} (any drive strength) and refuses to guess at others"})
    {pins, toks} = named(toks, st, %{})
    toks = expect(toks, ";")

    defs =
      for {pin, f} <- outs, Map.has_key?(pins, pin) do
        {net_name(single(pins[pin])), subst_pins(f, pins, cell)}
      end

    statements(toks, %{st | defs: st.defs ++ defs})
  end

  defp statements([t | _], _st), do: throw({:qalib, "unexpected #{inspect(t)}"})

  defp id_list([{:id, n}, {:sym, ","} | toks], acc), do: id_list(toks, [n | acc])
  defp id_list([{:id, n}, {:sym, ";"} | toks], acc), do: {Enum.reverse([n | acc]), toks}
  defp id_list(_, _), do: throw({:qalib, "a declaration lists names separated by commas, ended by ;"})

  defp expect([{:sym, s} | toks], s), do: toks
  defp expect(_, s), do: throw({:qalib, "expected #{s}"})

  defp arg_list(toks, st, acc) do
    {e, toks} = expr(toks, st)

    case toks do
      [{:sym, ","} | t] -> arg_list(t, st, [e | acc])
      [{:sym, ")"} | t] -> {Enum.reverse([e | acc]), t}
      _ -> throw({:qalib, "expected , or ) in a gate's arguments"})
    end
  end

  defp named([{:sym, ")"} | toks], _st, acc), do: {acc, toks}
  defp named([{:sym, ","} | toks], st, acc), do: named(toks, st, acc)

  defp named([{:sym, "."}, {:id, pin}, {:sym, "("} | toks], st, acc) do
    case toks do
      [{:sym, ")"} | t] -> named(t, st, acc)
      _ ->
        {e, t} = expr(toks, st)
        named(expect(t, ")"), st, Map.put(acc, pin, e))
    end
  end

  defp named(_, _, _), do: throw({:qalib, "a cell's pins are named: .PIN(net)"})

  defp single([e]), do: e
  defp single(l) when is_list(l), do: throw({:qalib, "a #{length(l)}-bit bus where one bit is expected: select a bit"})
  defp single(e), do: e

  defp net_name({:net, n}), do: n
  defp net_name(e), do: throw({:qalib, "an output must be a net, not #{inspect(e)}"})

  defp subst_pins({:pin, p}, pins, cell), do: single(Map.get(pins, p) || throw({:qalib, "#{cell}: pin #{p} is not connected"}))
  defp subst_pins({:const, _} = c, _, _), do: c
  defp subst_pins({:not, e}, pins, cell), do: {:not, subst_pins(e, pins, cell)}
  defp subst_pins({op, es}, pins, cell) when op in [:and, :or, :xor], do: {op, Enum.map(es, &subst_pins(&1, pins, cell))}
  defp subst_pins({:mux, s, t, f}, pins, cell), do: {:mux, subst_pins(s, pins, cell), subst_pins(t, pins, cell), subst_pins(f, pins, cell)}

  # left-hand side: a net, a bit, a whole bus, or a concatenation of them
  defp lvalue([{:sym, "{"} | toks], st) do
    {parts, toks} = concat_items(toks, st, [], &lvalue/2)
    {Enum.flat_map(parts, & &1), toks}
  end

  defp lvalue(toks, st) do
    case primary(toks, st) do
      {l, t} when is_list(l) -> {Enum.map(l, &net_name/1), t}
      {e, t} -> {[net_name(e)], t}
    end
  end

  defp concat_items(toks, st, acc, f) do
    {e, toks} = f.(toks, st)

    case toks do
      [{:sym, ","} | t] -> concat_items(t, st, [e | acc], f)
      [{:sym, "}"} | t] -> {Enum.reverse([e | acc]), t}
      _ -> throw({:qalib, "expected , or } in a concatenation"})
    end
  end

  # expressions: ?: < | < ^ < & < unary ~ ! < primary (a bus stays a list of bits until it meets an operator)
  defp expr(toks, st) do
    {c, toks} = bor(toks, st)

    case toks do
      [{:sym, "?"} | t] ->
        {a, t} = expr(t, st)
        t = expect(t, ":")
        {b, t} = expr(t, st)
        {{:mux, single(c), single(a), single(b)}, t}

      _ ->
        {c, toks}
    end
  end

  defp bor(toks, st), do: binop(toks, st, "|", :or, &bxor/2)
  defp bxor(toks, st), do: binop(toks, st, "^", :xor, &band/2)
  defp band(toks, st), do: binop(toks, st, "&", :and, &unary/2)

  defp binop(toks, st, sym, op, next) do
    {a, toks} = next.(toks, st)
    collect(toks, st, sym, op, next, [a])
  end

  defp collect([{:sym, sym} | t], st, sym, op, next, acc) do
    {b, t} = next.(t, st)
    collect(t, st, sym, op, next, [b | acc])
  end

  defp collect(toks, _st, _sym, _op, _next, [one]), do: {one, toks}
  defp collect(toks, _st, _sym, op, _next, acc), do: {{op, acc |> Enum.reverse() |> Enum.map(&single/1)}, toks}

  defp unary([{:sym, s} | t], st) when s in ["~", "!"] do
    {e, t} = unary(t, st)
    {{:not, single(e)}, t}
  end

  defp unary(toks, st), do: primary(toks, st)

  defp primary([{:sym, "("} | t], st) do
    {e, t} = expr(t, st)
    {e, expect(t, ")")}
  end

  defp primary([{:sym, "{"} | t], st) do
    {parts, t} = concat_items(t, st, [], &expr/2)
    {Enum.flat_map(parts, &List.wrap/1), t}
  end

  defp primary([{:num, b} | t], _st) when b in [0, 1], do: {{:const, b}, t}
  defp primary([{:int, b} | t], _st) when b in [0, 1], do: {{:const, b}, t}

  defp primary([{:id, n}, {:sym, "["}, {:int, i}, {:sym, "]"} | t], _st), do: {{:net, "#{n}[#{i}]"}, t}

  defp primary([{:id, n} | t], st) do
    case Map.fetch(st.widths, n) do
      {:ok, bits} -> {Enum.map(bits, &{:net, &1}), t}
      :error -> {{:net, n}, t}
    end
  end

  defp primary(toks, _), do: throw({:qalib, "expected an expression at #{inspect(Enum.take(toks, 3))}"})

  # ---------------------------------------------------------------- BLIF

  @doc """
  Read a combinational BLIF model: `.inputs`, `.outputs`, `.names` (on-set
  or off-set covers, `-` for don't-care) and `.gate` with the cells above.
  `.latch` is refused.
  """
  def read_blif(text) when is_binary(text) do
    lines =
      text
      |> String.replace(~r/\\\r?\n/, " ")
      |> String.split("\n")
      |> Enum.map(&(&1 |> String.split("#") |> hd() |> String.trim()))
      |> Enum.reject(&(&1 == ""))

    st = %{name: "top", inputs: [], outputs: [], wires: MapSet.new(), widths: %{}, defs: [], cover: nil}

    st =
      Enum.reduce(lines ++ [".end"], st, fn line, st ->
        words = String.split(line)

        case words do
          [".model", n | _] -> close_cover(%{st | name: n})
          [".inputs" | ns] -> close_cover(%{st | inputs: st.inputs ++ ns})
          [".outputs" | ns] -> close_cover(%{st | outputs: st.outputs ++ ns})
          [".names" | ns] -> st |> close_cover() |> Map.put(:cover, {ns, []})
          [".gate", cell | conns] -> st |> close_cover() |> gate(cell, conns)
          [".latch" | _] -> throw({:qalib, ".latch: Qālib compares combinational logic only"})
          [".end" | _] -> close_cover(st)
          [kw | _] -> if String.starts_with?(kw, "."), do: throw({:qalib, "unsupported BLIF directive #{kw}"}), else: row(st, words)
        end
      end)

    build(st)
  catch
    {:qalib, why} -> {:error, why}
  end

  defp row(%{cover: {ns, rows}} = st, words), do: %{st | cover: {ns, rows ++ [words]}}
  defp row(_st, words), do: throw({:qalib, "a cover row outside .names: #{Enum.join(words, " ")}"})

  defp close_cover(%{cover: nil} = st), do: st

  defp close_cover(%{cover: {ns, rows}} = st) do
    {ins, [out]} = Enum.split(ns, -1)

    e =
      case rows do
        [] -> {:const, 0}
        [["1"]] when ins == [] -> {:const, 1}
        [["0"]] when ins == [] -> {:const, 0}
        _ ->
          on = Enum.map(rows, fn
            [pat, v] when byte_size(pat) == length(ins) and v in ["0", "1"] -> {pat, v}
            r -> throw({:qalib, ".names #{out}: bad cover row #{Enum.join(r, " ")}"})
          end)

          values = on |> Enum.map(&elem(&1, 1)) |> Enum.uniq()
          if length(values) != 1, do: throw({:qalib, ".names #{out}: a cover mixes on-set and off-set rows"})

          sop = {:or, Enum.map(on, fn {pat, _} -> {:and, cube(pat, ins)} end)}
          if values == ["1"], do: sop, else: {:not, sop}
      end

    %{st | cover: nil, defs: st.defs ++ [{out, e}]}
  end

  defp cube(pat, ins) do
    lits = for {c, n} <- Enum.zip(String.graphemes(pat), ins), c != "-", do: if(c == "1", do: {:net, n}, else: {:not, {:net, n}})
    if lits == [], do: [{:const, 1}], else: lits
  end

  defp gate(st, cell, conns) do
    base = cell |> String.replace_prefix(@lib, "") |> String.replace(~r/_\d+$/, "")
    outs = Map.get(@cells, base) || throw({:qalib, "unknown cell #{cell}"})
    pins = Map.new(conns, fn c -> case String.split(c, "=", parts: 2) do [p, n] -> {p, {:net, n}}; _ -> throw({:qalib, ".gate #{cell}: #{c} is not PIN=net"}) end end)
    defs = for {pin, f} <- outs, Map.has_key?(pins, pin), do: {net_name(pins[pin]), subst_pins(f, pins, cell)}
    %{st | defs: st.defs ++ defs}
  end

  # ------------------------------------------------------- to a circuit

  # nets defined in any order: topological by the nets each definition reads
  defp build(st) do
    defs = st.defs
    if length(defs) > @max_nets, do: throw({:qalib, "more than #{@max_nets} nets"})
    by_net = Enum.reduce(defs, %{}, fn {n, e}, m -> if Map.has_key?(m, n), do: throw({:qalib, "#{n} is driven twice"}), else: Map.put(m, n, e) end)
    if (dup = Enum.find(st.inputs, &Map.has_key?(by_net, &1))), do: throw({:qalib, "the input #{dup} is also driven"})
    if (u = Enum.find(st.outputs, &(not Map.has_key?(by_net, &1) and &1 not in st.inputs))), do: throw({:qalib, "the output #{u} is never driven"})
    if st.outputs == [], do: throw({:qalib, "no outputs"})

    b0 = %{nodes: [], count: 0, memo: %{}, net: %{}, inputs: st.inputs}
    b = Enum.reduce(st.inputs, b0, fn n, b -> {i, b} = add(b, {:in, n}); %{b | net: Map.put(b.net, n, i)} end)
    {b, _} = Enum.reduce(st.outputs, {b, MapSet.new()}, fn o, {b, path} -> resolve(b, o, by_net, path) end)
    outs = Enum.map(st.outputs, fn o -> {o, b.net[o]} end)
    {:ok, %Circuit{inputs: st.inputs, outputs: outs, nodes: b.nodes |> Enum.reverse() |> List.to_tuple(), size: b.count}}
  end

  defp resolve(b, n, by_net, path) do
    cond do
      Map.has_key?(b.net, n) -> {b, path}
      MapSet.member?(path, n) -> throw({:qalib, "a combinational loop through #{n}"})
      not Map.has_key?(by_net, n) -> throw({:qalib, "#{n} is read but never driven"})
      true ->
        path = MapSet.put(path, n)
        {i, b} = compile(b, by_net[n], by_net, path)
        {%{b | net: Map.put(b.net, n, i)}, MapSet.delete(path, n)}
    end
  end

  defp compile(b, {:net, n}, by_net, path) do
    {b, _} = resolve(b, n, by_net, path)
    {b.net[n], b}
  end

  defp compile(b, {:const, v}, _, _), do: add(b, {:const, v})

  defp compile(b, {:not, e}, by_net, path) do
    {i, b} = compile(b, e, by_net, path)
    add(b, {:not, i})
  end

  defp compile(b, {op, es}, by_net, path) when op in [:and, :or, :xor] do
    {is, b} = Enum.map_reduce(es, b, fn e, b -> compile(b, e, by_net, path) end)
    [first | more] = is
    Enum.reduce(more, {first, b}, fn i, {acc, b} -> add(b, {op, acc, i}) end)
  end

  defp compile(b, {:mux, s, t, f}, by_net, path) do
    {is, b} = compile(b, s, by_net, path)
    {it, b} = compile(b, t, by_net, path)
    {iff, b} = compile(b, f, by_net, path)
    add(b, {:mux, is, it, iff})
  end

  defp add(b, n) do
    case Map.fetch(b.memo, n) do
      {:ok, i} when elem(n, 0) != :in -> {i, b}
      _ -> {b.count, %{b | nodes: [n | b.nodes], count: b.count + 1, memo: Map.put(b.memo, n, b.count)}}
    end
  end

  # -------------------------------------------------------------- writing

  @doc """
  Write a circuit as structural Verilog over `sky130_fd_sc_hd` cells.
  `style: :cells` (default: INV, AND2, OR2, XOR2, MUX2, CONB, one per gate)
  or `:nand` (NAND2 and INV only). Returns `{:ok, text, stats}`;
  `stats` counts the cells by type.
  """
  def to_verilog(%Circuit{} = c, opts \\ []) do
    style = Keyword.get(opts, :style, :cells)
    name = Keyword.get(opts, :module, "vapor_top")
    clash = Enum.find(Enum.map(c.outputs, &elem(&1, 0)), &(&1 in c.inputs))

    if clash do
      {:error, "#{clash} is both an input and an output: a Verilog port has one direction"}
    else
      net = fn i -> case elem(c.nodes, i) do {:in, n} -> n; _ -> "_#{i}_" end end
      {insts, _} =
        Enum.flat_map_reduce(0..(c.size - 1)//1, 0, fn i, k ->
          cells = cells_for(elem(c.nodes, i), net, net.(i), style)
          {Enum.with_index(cells, k) |> Enum.map(fn {{cell, pins}, j} -> {cell, "g#{j}", pins} end), k + length(cells)}
        end)

      wires = insts |> Enum.flat_map(fn {_, _, pins} -> Enum.map(pins, &elem(&1, 1)) end) |> Enum.reject(&(&1 in c.inputs)) |> Enum.uniq() |> Enum.sort()
      outs = Enum.map(c.outputs, &elem(&1, 0))

      text =
        IO.iodata_to_binary([
          "// written by vapor's Qālib (style: #{style}); equivalence to the source: Vapor.Qalib.certify/3\n",
          "module #{id(name)} (", Enum.map_join(c.inputs ++ outs, ", ", &id/1), ");\n",
          Enum.map(c.inputs, &"  input #{id(&1)};\n"),
          Enum.map(outs, &"  output #{id(&1)};\n"),
          Enum.map(wires, &"  wire #{id(&1)};\n"),
          Enum.map(insts, fn {cell, inst, pins} -> "  #{@lib}#{cell}_1 #{inst} (" <> Enum.map_join(pins, ", ", fn {p, n} -> ".#{p}(#{id(n)})" end) <> ");\n" end),
          Enum.map(c.outputs, fn {o, i} -> "  assign #{id(o)} = #{id(net.(i))};\n" end),
          "endmodule\n"
        ])

      {:ok, text, %{cells: length(insts), by_type: insts |> Enum.frequencies_by(&elem(&1, 0))}}
    end
  end

  defp cells_for({:in, _}, _net, _out, _style), do: []
  defp cells_for({:const, 0}, _net, out, _), do: [{"conb", [{"LO", out}]}]
  defp cells_for({:const, 1}, _net, out, _), do: [{"conb", [{"HI", out}]}]
  defp cells_for({:not, a}, net, out, _), do: [{"inv", [{"A", net.(a)}, {"Y", out}]}]
  defp cells_for({:and, a, b}, net, out, :cells), do: [{"and2", [{"A", net.(a)}, {"B", net.(b)}, {"X", out}]}]
  defp cells_for({:or, a, b}, net, out, :cells), do: [{"or2", [{"A", net.(a)}, {"B", net.(b)}, {"X", out}]}]
  defp cells_for({:xor, a, b}, net, out, :cells), do: [{"xor2", [{"A", net.(a)}, {"B", net.(b)}, {"X", out}]}]
  defp cells_for({:mux, s, a, b}, net, out, :cells), do: [{"mux2", [{"A0", net.(b)}, {"A1", net.(a)}, {"S", net.(s)}, {"X", out}]}]

  # NAND2 + INV: intermediate nets are named after the output
  defp cells_for({:and, a, b}, net, out, :nand), do: [{"nand2", [{"A", net.(a)}, {"B", net.(b)}, {"Y", out <> "n"}]}, {"inv", [{"A", out <> "n"}, {"Y", out}]}]

  defp cells_for({:or, a, b}, net, out, :nand),
    do: [{"inv", [{"A", net.(a)}, {"Y", out <> "a"}]}, {"inv", [{"A", net.(b)}, {"Y", out <> "b"}]}, {"nand2", [{"A", out <> "a"}, {"B", out <> "b"}, {"Y", out}]}]

  defp cells_for({:xor, a, b}, net, out, :nand) do
    {x, y} = {net.(a), net.(b)}
    [{"nand2", [{"A", x}, {"B", y}, {"Y", out <> "m"}]}, {"nand2", [{"A", x}, {"B", out <> "m"}, {"Y", out <> "p"}]},
     {"nand2", [{"A", y}, {"B", out <> "m"}, {"Y", out <> "q"}]}, {"nand2", [{"A", out <> "p"}, {"B", out <> "q"}, {"Y", out}]}]
  end

  defp cells_for({:mux, s, a, b}, net, out, :nand) do
    [{"inv", [{"A", net.(s)}, {"Y", out <> "s"}]}, {"nand2", [{"A", net.(s)}, {"B", net.(a)}, {"Y", out <> "p"}]},
     {"nand2", [{"A", out <> "s"}, {"B", net.(b)}, {"Y", out <> "q"}]}, {"nand2", [{"A", out <> "p"}, {"B", out <> "q"}, {"Y", out}]}]
  end

  defp id(n), do: if(Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_]*$/, n), do: n, else: "\\" <> n <> " ")

  # -------------------------------------------------------------- certify

  @doc """
  Prove two circuits (or netlist texts, Verilog or BLIF, or Rebis netlists)
  compute the same function: `Vapor.Rebis.equivalent/3`, with its evidence.
  """
  def certify(a, b, opts \\ []) do
    with {:ok, ca} <- circuit(a), {:ok, cb} <- circuit(b) do
      Rebis.equivalent(ca, cb, opts)
    end
  end

  @doc "A circuit from a circuit, or from text in any format Qālib or Rebis reads (sniffed)."
  def circuit(%Circuit{} = c), do: {:ok, c}

  def circuit(text) when is_binary(text) do
    cond do
      text =~ ~r/^\s*module\b/m -> read_verilog(text)
      text =~ ~r/^\s*\.(model|inputs|names)\b/m -> read_blif(text)
      text =~ ~r/^\s*aag\s/ -> Rebis.from_aiger(text)
      true -> Rebis.parse(text)
    end
  end
end
