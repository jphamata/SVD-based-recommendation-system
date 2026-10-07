defmodule Vapor.Engineering.Circuit do
  @moduledoc """
  Circuit simulation from a SPICE-style netlist (docs/ENGENHARIA.md §1):
  modified nodal analysis, the DC operating point by Newton with
  junction-voltage limiting, DC sweeps, small-signal AC sweeps
  (complex MNA linearised at the operating point) and transient analysis
  by the trapezoidal rule (backward Euler on request).

  Elements: `R`, `C` (`IC=`), `L` (`IC=`), independent `V` and `I`
  (`DC`, `AC mag [phase]`, `SIN(off amp freq [delay damp phase])`,
  `PULSE(v1 v2 delay rise fall width period)`, `PWL(t1 v1 t2 v2 …)`),
  diodes `D` (`IS= N=`), **MOSFETs** `M d g s [b] NMOS|PMOS` (level 1,
  Shichman–Hodges: `KP= VTO= LAMBDA= W= L=`) and **bipolar transistors**
  `Q c b e NPN|PNP` (Ebers–Moll transport model: `IS= BF= BR=`) — both
  pinned against ngspice in `engenharia_test.exs` —, controlled sources `E` (VCVS), `G` (VCCS), `F`
  (CCCS) and `H` (CCVS) controlled by a voltage source's current, and
  ideal op-amps `O` (out, +, −: the virtual short). Values take the
  SPICE suffixes `f p n u µ m k meg g t` (`4.7k`, `1meg`, `100n`).
  Analyses: `.op`, `.dc SRC start stop step`, `.ac dec|lin points fstart
  fstop`, `.tran tstep tstop [tstart]`, `.method be`.

  The operating point comes with its **certificate**: Kirchhoff's
  current law re-evaluated at every node from the element equations
  (not from the matrix that was solved), and the power balance — what the
  sources deliver equals what the elements absorb.
  """
  alias Vapor.Dense

  @vt 0.025851991024560

  # =============================================================== parsing

  @doc "Parse a netlist: `{:ok, circuit}` or `{:error, why}` naming the line."
  def parse(text) do
    text
    |> String.split("\n")
    |> Enum.with_index(1)
    |> Enum.reduce_while({:ok, %{elements: [], analyses: [], nodes: MapSet.new(), method: :trap, title: nil}}, fn {raw, n}, {:ok, acc} ->
      line = raw |> String.split(~r/[;$]/, parts: 2) |> hd() |> String.trim()
      cond do
        line == "" or String.starts_with?(line, "*") -> {:cont, {:ok, acc}}
        String.starts_with?(line, ".") ->
          case directive(line, acc) do
            {:ok, acc} -> {:cont, {:ok, acc}}
            {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
          end
        true ->
          case element(line) do
            {:ok, els} ->
              els = List.wrap(els)
              {:cont, {:ok, %{acc | elements: acc.elements ++ els, nodes: Enum.reduce(Enum.flat_map(els, &nodes_of/1), acc.nodes, &MapSet.put(&2, &1))}}}
            {:error, w} -> {:halt, {:error, "line #{n}: #{w}"}}
          end
      end
    end)
    |> case do
      {:ok, c} -> finish(c)
      e -> e
    end
  end

  defp finish(c) do
    names = c.elements |> Enum.map(& &1.name) |> Enum.frequencies() |> Enum.filter(fn {_, k} -> k > 1 end)
    nodes = c.nodes |> MapSet.delete("0") |> Enum.sort()
    cond do
      c.elements == [] -> {:error, "no elements"}
      names != [] -> {:error, "element names repeat: #{names |> Enum.map(&elem(&1, 0)) |> Enum.join(", ")}"}
      not MapSet.member?(c.nodes, "0") -> {:error, "no ground: one node must be 0 (or gnd)"}
      true ->
        refs = for %{ctrl: v} <- c.elements, v != nil, do: v
        vs = for %{kind: k, name: n} <- c.elements, k in [:v, :e, :h, :l, :o], do: n
        case Enum.reject(refs, &(&1 in vs)) do
          [] -> {:ok, Map.merge(c, %{nodes: nodes, analyses: if(c.analyses == [], do: [{:op}], else: c.analyses)})}
          m -> {:error, "controlling source(s) not found: #{Enum.join(m, ", ")}"}
        end
    end
  end

  defp nodes_of(%{n: ns}), do: ns

  defp nd(s), do: (s = String.downcase(s); if s in ["gnd", "ground"], do: "0", else: s)

  @doc "A SPICE number: `4.7k` → 4700.0, `1meg` → 1.0e6, `100n` → 1.0e-7 (trailing unit letters ignored)."
  def value(s) do
    s = String.downcase(String.trim(s))
    case Regex.run(~r/^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(meg|mil|[fpnuµmkgt])?[a-zω]*$/u, s) do
      [_, num | rest] ->
        {v, _} = Float.parse(if String.starts_with?(num, "."), do: "0" <> num, else: (if String.starts_with?(num, "-."), do: "-0" <> String.slice(num, 1..-1), else: num))
        mult = %{"f" => 1.0e-15, "p" => 1.0e-12, "n" => 1.0e-9, "u" => 1.0e-6, "µ" => 1.0e-6, "m" => 1.0e-3, "k" => 1.0e3, "meg" => 1.0e6, "g" => 1.0e9, "t" => 1.0e12, "mil" => 25.4e-6}
        {:ok, v * Map.get(mult, List.first(rest) || "", 1.0)}
      _ -> {:error, "not a number: #{inspect(s)}"}
    end
  end

  defp v!(s), do: (case value(s) do {:ok, v} -> v; {:error, w} -> throw({:netlist, w}) end)

  defp element(line) do
    toks = Regex.scan(~r/\w+\([^)]*\)|[^\s(),]+(?:\([^)]*\))?/u, line) |> Enum.map(&hd/1)
    [name | rest] = toks
    kind = name |> String.first() |> String.downcase()

    case {kind, rest} do
      {"r", [a, b, v | _]} -> {:ok, %{kind: :r, name: name, n: [nd(a), nd(b)], value: v!(v), ctrl: nil}}
      {"c", [a, b, v | opts]} -> {:ok, %{kind: :c, name: name, n: [nd(a), nd(b)], value: v!(v), ic: kv(opts)["ic"], ctrl: nil}}
      {"l", [a, b, v | opts]} -> {:ok, %{kind: :l, name: name, n: [nd(a), nd(b)], value: v!(v), ic: kv(opts)["ic"], ctrl: nil}}
      {k, [a, b | spec]} when k in ["v", "i"] -> {:ok, Map.merge(%{kind: String.to_atom(k), name: name, n: [nd(a), nd(b)], ctrl: nil}, source(spec))}
      {"d", [a, b | opts]} -> (o = kv(opts); {:ok, %{kind: :d, name: name, n: [nd(a), nd(b)], is: o["is"] || 1.0e-14, nf: o["n"] || 1.0, ctrl: nil}})
      {"m", [d, g, s | rest]} ->
        {typ, opts} = device_type(rest, ["nmos", "pmos"])
        o = kv(opts)
        pol = if typ == "pmos", do: -1, else: 1
        beta = (o["kp"] || 2.0e-5) * (o["w"] || 1.0e-4) / (o["l"] || 1.0e-4)
        # the current d → s depends on v(d), v(s) and v(g): the third node is the gate
        {:ok, %{kind: :x3, dev: :mos, name: name, n: [nd(d), nd(s), nd(g)], pol: pol, beta: beta, vto: o["vto"] || (pol * 1.0), lambda: o["lambda"] || 0.0, ctrl: nil}}
      {"q", [c, b, e | rest]} ->
        {typ, opts} = device_type(rest, ["npn", "pnp"])
        o = kv(opts)
        pol = if typ == "pnp", do: -1, else: 1
        par = %{pol: pol, is: o["is"] || 1.0e-16, bf: o["bf"] || 100.0, br: o["br"] || 1.0}
        # two branches: collector → emitter (Ic) and base → emitter (Ib); the emitter current is their sum (KCL)
        {:ok, [Map.merge(%{kind: :x3, dev: :bjt_c, name: name <> ".c", n: [nd(c), nd(e), nd(b)], ctrl: nil}, par),
               Map.merge(%{kind: :x3, dev: :bjt_b, name: name <> ".b", n: [nd(b), nd(e), nd(c)], ctrl: nil}, par)]}
      {"e", [a, b, c, d, g]} -> {:ok, %{kind: :e, name: name, n: [nd(a), nd(b), nd(c), nd(d)], gain: v!(g), ctrl: nil}}
      {"g", [a, b, c, d, g]} -> {:ok, %{kind: :g, name: name, n: [nd(a), nd(b), nd(c), nd(d)], gain: v!(g), ctrl: nil}}
      {"f", [a, b, src, g]} -> {:ok, %{kind: :f, name: name, n: [nd(a), nd(b)], ctrl: src, gain: v!(g)}}
      {"h", [a, b, src, g]} -> {:ok, %{kind: :h, name: name, n: [nd(a), nd(b)], ctrl: src, gain: v!(g)}}
      {"o", [o, p, m | _]} -> {:ok, %{kind: :o, name: name, n: [nd(o), nd(p), nd(m)], ctrl: nil}}
      _ -> {:error, "element #{name}: not understood (#{line})"}
    end
  catch
    {:netlist, w} -> {:error, w}
  end

  # a device line: nodes, an optional bulk node, the type word, then KEY=VALUE options
  defp device_type(rest, types) do
    low = Enum.map(rest, &String.downcase/1)
    case Enum.find_index(low, &(&1 in types)) do
      nil -> {hd(types), Enum.filter(rest, &String.contains?(&1, "="))}
      i -> {Enum.at(low, i), Enum.drop(rest, i + 1)}
    end
  end

  @doc false
  # the current p → q of a three-terminal element at node voltages (vp, vq, vr)
  def x3_current(%{dev: :mos} = e, vp, vq, vr) do
    {vd, vs, vg} = {e.pol * vp, e.pol * vq, e.pol * vr}
    # a symmetric device: the lower of drain and source acts as the source
    {vd, vs, sgn} = if vd >= vs, do: {vd, vs, 1}, else: {vs, vd, -1}
    vov = vg - vs - e.pol * e.vto
    vds = vd - vs
    id =
      cond do
        vov <= 0 -> 0.0
        vds < vov -> e.beta * (vov * vds - vds * vds / 2) * (1 + e.lambda * vds)
        true -> e.beta / 2 * vov * vov * (1 + e.lambda * vds)
      end
    e.pol * sgn * id
  end

  def x3_current(%{dev: dev} = e, vp, vq, vr) when dev in [:bjt_c, :bjt_b] do
    {vc, ve, vb} = if dev == :bjt_c, do: {vp, vq, vr}, else: {vr, vq, vp}
    vbe = e.pol * (vb - ve); vbc = e.pol * (vb - vc)
    xe = :math.exp(min(vbe / @vt, 80.0)); xc = :math.exp(min(vbc / @vt, 80.0))
    case dev do
      :bjt_c -> e.pol * (e.is * (xe - xc) - e.is / e.br * (xc - 1))
      :bjt_b -> e.pol * (e.is / e.bf * (xe - 1) + e.is / e.br * (xc - 1))
    end
  end

  # its three conductances ∂i/∂v by central differences (h = 10⁻⁷ V: error ~ (h/V_T)² ≈ 10⁻¹¹)
  defp x3_partials(e, vp, vq, vr) do
    h = 1.0e-7
    f = &x3_current(e, &1, &2, &3)
    {f.(vp, vq, vr), (f.(vp + h, vq, vr) - f.(vp - h, vq, vr)) / (2 * h), (f.(vp, vq + h, vr) - f.(vp, vq - h, vr)) / (2 * h), (f.(vp, vq, vr + h) - f.(vp, vq, vr - h)) / (2 * h)}
  end

  # junction limiting for the bipolar branches (each junction stepped as a diode would be)
  defp x3_limit(%{dev: :mos}, vp, vq, vr), do: {vp, vq, vr}
  defp x3_limit(e, vp, vq, vr) do
    lim = fn v -> limit(v, %{nf: 1.0, is: e.is}) end
    # base–emitter of each branch: keep the emitter, limit the junction
    case e.dev do
      :bjt_c -> {vp, vq, vq + e.pol * lim.(e.pol * (vr - vq))}
      :bjt_b -> {vq + e.pol * lim.(e.pol * (vp - vq)), vq, vr}
    end
  end

  defp kv(opts) do
    for o <- opts, [k, v] <- [String.split(o, "=", parts: 2)], into: %{}, do: {String.downcase(k), v!(v)}
  end

  defp source(spec) do
    base = %{dc: 0.0, ac: 0.0, ac_phase: 0.0, wave: nil}
    up = Enum.map(spec, &String.downcase/1)
    parse_src(up, base)
  end

  defp parse_src([], acc), do: acc
  defp parse_src(["dc", v | r], acc), do: parse_src(r, %{acc | dc: v!(v)})
  defp parse_src(["ac", m, p | r], acc) do
    case value(p) do
      {:ok, ph} -> parse_src(r, %{acc | ac: v!(m), ac_phase: ph})
      _ -> parse_src([p | r], %{acc | ac: v!(m)})
    end
  end
  defp parse_src(["ac", m], acc), do: %{acc | ac: v!(m)}
  defp parse_src([w | r], acc) do
    case Regex.run(~r/^(sin|pulse|pwl)\((.*)\)$/, w) do
      [_, kind, args] ->
        vals = args |> String.split(~r/[\s,]+/, trim: true) |> Enum.map(&v!/1)
        dc = if kind == "pulse" or kind == "pwl", do: (if kind == "pwl", do: Enum.at(vals, 1, 0.0), else: hd(vals)), else: hd(vals)
        parse_src(r, %{acc | wave: {String.to_atom(kind), vals}, dc: (if acc.dc == 0.0, do: dc, else: acc.dc)})
      nil ->
        case value(w) do
          {:ok, v} -> parse_src(r, %{acc | dc: v})
          _ -> throw({:netlist, "source: not understood #{inspect(w)}"})
        end
    end
  end

  defp directive(line, acc) do
    toks = line |> String.downcase() |> String.split()
    case toks do
      [".op" | _] -> {:ok, %{acc | analyses: acc.analyses ++ [{:op}]}}
      [".dc", src, a, b, s | _] -> {:ok, %{acc | analyses: acc.analyses ++ [{:dc, src, v!(a), v!(b), v!(s)}]}}
      [".ac", mode, pts, f1, f2 | _] when mode in ["dec", "lin", "oct"] -> {:ok, %{acc | analyses: acc.analyses ++ [{:ac, mode, trunc(v!(pts)), v!(f1), v!(f2)}]}}
      [".tran", st, stop | rest] -> {:ok, %{acc | analyses: acc.analyses ++ [{:tran, v!(st), v!(stop), (case rest do [t0 | _] -> v!(t0); _ -> 0.0 end)}]}}
      [".method", "be" | _] -> {:ok, %{acc | method: :be}}
      [".method", _ | _] -> {:ok, %{acc | method: :trap}}
      [".title" | _] -> {:ok, %{acc | title: line |> String.replace(~r/^\.title\s*/i, "")}}
      [".end" | _] -> {:ok, acc}
      [d | _] -> {:error, "directive #{d} not supported (.op .dc .ac .tran .method .title .end)"}
    end
  catch
    {:netlist, w} -> {:error, w}
  end

  # ============================================================== analysis

  @doc "Run every analysis of a netlist: `{:ok, %{op, dc, ac, tran, …}}` or `{:error, why}`."
  def run(text) do
    with {:ok, c} <- parse(text) do
      Enum.reduce_while(c.analyses, {:ok, %{nodes: c.nodes, elements: length(c.elements), title: c.title}}, fn a, {:ok, acc} ->
        case analysis(c, a) do
          {:ok, k, r} -> {:cont, {:ok, Map.put(acc, k, r)}}
          {:error, w} -> {:halt, {:error, w}}
        end
      end)
    end
  end

  defp analysis(c, {:op}), do: with({:ok, r} <- op(c), do: {:ok, :op, r})

  defp analysis(c, {:dc, src, a, b, s}) do
    if Enum.any?(c.elements, &(String.downcase(&1.name) == src and &1.kind in [:v, :i])) do
      n = max(1, round((b - a) / s))
      {pts, _} =
        Enum.map_reduce(0..n, nil, fn k, guess ->
          v = a + (b - a) * k / n
          c2 = %{c | elements: Enum.map(c.elements, fn e -> if String.downcase(e.name) == src, do: %{e | dc: v, wave: nil}, else: e end)}
          case op(c2, guess) do
            {:ok, r} -> {%{sweep: v, nodes: r.nodes, currents: r.currents}, r.x}
            {:error, _} -> {%{sweep: v, nodes: nil}, guess}
          end
        end)
      {:ok, :dc, %{source: src, points: pts}}
    else
      {:error, ".dc: no independent source named #{src}"}
    end
  end

  defp analysis(c, {:ac, mode, pts, f1, f2}) do
    with {:ok, opr} <- op(c), do: {:ok, :ac, ac(c, opr, mode, pts, f1, f2)}
  end

  defp analysis(c, {:tran, st, stop, t0}), do: with({:ok, r} <- tran(c, st, stop, t0), do: {:ok, :tran, r})

  # ---------------------------------------------------------- the system

  # unknowns: the nodes, then one current per voltage-defined branch (V, E, H, L, O)
  defp layout(c) do
    ni = c.nodes |> Enum.with_index() |> Map.new()
    branches = c.elements |> Enum.filter(&(&1.kind in [:v, :e, :h, :l, :o])) |> Enum.map(& &1.name)
    bi = branches |> Enum.with_index(length(c.nodes)) |> Map.new()
    {ni, bi, length(c.nodes) + length(branches)}
  end

  defp idx(_ni, "0"), do: nil
  defp idx(ni, n), do: Map.fetch!(ni, n)

  # build A x = z (real) for a DC or transient Newton iterate; `mode` is :dc or {:tran, h, prev, method}
  defp stamp(c, {ni, bi, n}, x, mode, t) do
    xt = List.to_tuple(x)
    v = fn node -> (case idx(ni, node) do nil -> 0.0; i -> elem(xt, i) end) end
    acc = {%{}, %{}}

    {a, z} =
      Enum.reduce(c.elements, acc, fn e, {a, z} ->
        [p, q | _] = e.n
        {ip, iq} = {idx(ni, p), idx(ni, q)}
        case e.kind do
          :r -> cond_stamp(a, z, ip, iq, 1 / e.value, 0.0)
          :g -> (({cp, cq} = {idx(ni, Enum.at(e.n, 2)), idx(ni, Enum.at(e.n, 3))}); {a |> add(ip, cp, e.gain) |> add(ip, cq, -e.gain) |> add(iq, cp, -e.gain) |> add(iq, cq, e.gain), z})
          :i -> {a, z |> addz(ip, -src_value(e, t)) |> addz(iq, src_value(e, t))}
          :x3 ->
            r3 = Enum.at(e.n, 2); ir = idx(ni, r3)
            {vp, vq, vr} = x3_limit(e, v.(p), v.(q), v.(r3))
            {i0, gp, gq, gr} = x3_partials(e, vp, vq, vr)
            ieq = i0 - (gp * vp + gq * vq + gr * vr)
            a = a |> add(ip, ip, gp) |> add(ip, iq, gq) |> add(ip, ir, gr) |> add(iq, ip, -gp) |> add(iq, iq, -gq) |> add(iq, ir, -gr)
            # GMIN across the channel/junction (10⁻¹² S, as SPICE): a device in cut-off leaves no node floating
            cond_stamp(a, z |> addz(ip, -ieq) |> addz(iq, ieq), ip, iq, 1.0e-12, 0.0)
          :d ->
            vd = limit(v.(p) - v.(q), e)
            nvt = e.nf * @vt
            ex = :math.exp(min(vd / nvt, 80.0))
            id = e.is * (ex - 1)
            gd = e.is * ex / nvt + 1.0e-12
            cond_stamp(a, z, ip, iq, gd, id - gd * vd)
          :c ->
            case mode do
              :dc -> cond_stamp(a, z, ip, iq, 1.0e-12, 0.0)
              {:tran, h, prev, meth} ->
                {vp, ip_prev} = prev[e.name]
                if meth == :be, do: cond_stamp(a, z, ip, iq, e.value / h, -e.value / h * vp),
                                else: cond_stamp(a, z, ip, iq, 2 * e.value / h, -(2 * e.value / h * vp + ip_prev))
            end
          k when k in [:v, :e, :h, :l, :o] ->
            b = bi[e.name]
            a = a |> add(ip, b, 1.0) |> add(iq, b, -1.0)
            case k do
              :v -> {a |> add(b, ip, 1.0) |> add(b, iq, -1.0), Map.update(z, b, src_value(e, t), &(&1 + src_value(e, t)))}
              :e ->
                {cp, cq} = {idx(ni, Enum.at(e.n, 2)), idx(ni, Enum.at(e.n, 3))}
                {a |> add(b, ip, 1.0) |> add(b, iq, -1.0) |> add(b, cp, -e.gain) |> add(b, cq, e.gain), z}
              :h -> {a |> add(b, ip, 1.0) |> add(b, iq, -1.0) |> add(b, bi[e.ctrl], -e.gain), z}
              :o ->
                # ideal op-amp: the output branch carries what it must so that v(+) = v(−)
                [_, pp, mm] = e.n
                {a |> Map.delete({ip, b}) |> add(ip, b, 1.0) |> Map.delete({iq, b}) |> add(b, idx(ni, pp), 1.0) |> add(b, idx(ni, mm), -1.0), z}
              :l ->
                case mode do
                  :dc -> {a |> add(b, ip, 1.0) |> add(b, iq, -1.0), z}
                  {:tran, h, prev, meth} ->
                    {vprev, iprev} = prev[e.name]
                    r = if meth == :be, do: e.value / h, else: 2 * e.value / h
                    zz = if meth == :be, do: -r * iprev, else: -(r * iprev + vprev)
                    {a |> add(b, ip, 1.0) |> add(b, iq, -1.0) |> add(b, b, -r), Map.update(z, b, zz, &(&1 + zz))}
                end
            end
          :f ->
            cb = bi[e.ctrl]
            {a |> add(ip, cb, e.gain) |> add(iq, cb, -e.gain), z}
        end
      end)

    {dense(a, n), for(i <- 0..(n - 1), do: Map.get(z, i, 0.0))}
  end

  # an op-amp's output node gets the branch current; its own KCL row stays (the current flows out of the op-amp)
  defp add(m, nil, _, _), do: m
  defp add(m, _, nil, _), do: m
  defp add(m, i, j, v), do: Map.update(m, {i, j}, v, &(&1 + v))
  defp addz(z, nil, _), do: z
  defp addz(z, i, v), do: Map.update(z, i, v, &(&1 + v))

  # a conductance g between p and q with a current source ieq from p to q (companion models)
  defp cond_stamp(a, z, ip, iq, g, ieq) do
    a = a |> add(ip, ip, g) |> add(iq, iq, g) |> add(ip, iq, -g) |> add(iq, ip, -g)
    {a, z |> addz(ip, -ieq) |> addz(iq, ieq)}
  end

  defp dense(m, n), do: for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: Map.get(m, {i, j}, 0.0)))

  # SPICE's pnjlim, simplified: no step beyond the critical voltage in one go
  defp limit(vd, e) do
    nvt = e.nf * @vt
    vcrit = nvt * :math.log(nvt / (:math.sqrt(2) * e.is))
    if vd > vcrit, do: min(vd, vcrit + 2 * nvt * :math.log(1 + (vd - vcrit) / (2 * nvt)) + 0.0), else: vd
  end

  @doc false
  def src_value(%{wave: nil, dc: dc}, _t), do: dc
  def src_value(%{dc: dc}, nil), do: dc

  def src_value(%{wave: {:sin, vals}}, t) do
    [off, amp, f | r] = vals ++ [0.0, 0.0, 0.0]
    [delay, damp, ph] = Enum.take(r ++ [0.0, 0.0, 0.0], 3)
    if t < delay, do: off + amp * :math.sin(ph * :math.pi() / 180), else: off + amp * :math.exp(-damp * (t - delay)) * :math.sin(2 * :math.pi() * f * (t - delay) + ph * :math.pi() / 180)
  end

  def src_value(%{wave: {:pulse, vals}}, t) do
    [v1, v2, td, tr, tf, pw, per] = Enum.take(vals ++ [0.0, 0.0, 0.0, 1.0e-9, 1.0e-9, 1.0e30, 1.0e30], 7)
    if t < td do
      v1
    else
      tt = if per > 0, do: :math.fmod(t - td, per), else: t - td
      cond do
        tt < tr -> v1 + (v2 - v1) * tt / max(tr, 1.0e-30)
        tt < tr + pw -> v2
        tt < tr + pw + tf -> v2 + (v1 - v2) * (tt - tr - pw) / max(tf, 1.0e-30)
        true -> v1
      end
    end
  end

  def src_value(%{wave: {:pwl, vals}}, t) do
    pts = Enum.chunk_every(vals, 2, 2, :discard) |> Enum.map(&List.to_tuple/1)
    case pts do
      [] -> 0.0
      _ ->
        {t0, v0} = hd(pts)
        if t <= t0, do: v0, else:
          (Enum.chunk_every(pts, 2, 1, :discard) |> Enum.find_value(fn [{ta, va}, {tb, vb}] -> if t >= ta and t <= tb, do: va + (vb - va) * (t - ta) / max(tb - ta, 1.0e-30) end)) || elem(List.last(pts), 1)
    end
  end

  # -------------------------------------------------------------- DC / OP

  @doc "The DC operating point: node voltages, branch currents, element powers, and the KCL/power certificate."
  def op(c, guess \\ nil) do
    lay = {_, _, n} = layout(c)
    x0 = guess || List.duplicate(0.0, n)
    case newton(c, lay, x0, :dc, nil, 0) do
      {:ok, x, its} -> {:ok, report(c, lay, x, its, nil)}
      {:error, _} = e ->
        # source stepping: ramp the sources from 10 % to 100 %
        Enum.reduce_while(1..10, {:ok, x0}, fn k, {:ok, x} ->
          cs = %{c | elements: Enum.map(c.elements, fn el -> if el.kind in [:v, :i], do: %{el | dc: el.dc * k / 10, wave: nil}, else: el end)}
          case newton(cs, lay, x, :dc, nil, 0) do
            {:ok, x2, _} -> {:cont, {:ok, x2}}
            _ -> {:halt, e}
          end
        end)
        |> case do
          {:ok, x} -> (case newton(c, lay, x, :dc, nil, 0) do {:ok, x2, its} -> {:ok, report(c, lay, x2, its, "source stepping")}; err -> err end)
          err -> err
        end
    end
  end

  defp newton(c, lay, x, mode, t, its) do
    {a, z} = stamp(c, lay, x, mode, t)
    case Dense.solve(a, z) do
      {:ok, xn} ->
        dx = Dense.norm_inf(Dense.sub(xn, x))
        cond do
          dx < 1.0e-9 + 1.0e-6 * Dense.norm_inf(xn) -> {:ok, xn, its + 1}
          its > 200 -> {:error, "Newton did not converge (|Δx| = #{dx})"}
          true -> newton(c, lay, xn, mode, t, its + 1)
        end

      {:error, {:singular, k}} ->
        {ni, bi, _} = lay
        who = Enum.find_value(ni, fn {nm, i} -> if i == k, do: "node #{nm}" end) || Enum.find_value(bi, fn {nm, i} -> if i == k, do: "the current of #{nm}" end)
        {:error, "the circuit matrix is singular at #{who} (a floating node, a loop of voltage sources, or a cut-set of current sources)"}
    end
  end

  defp report(c, {ni, bi, _}, x, its, note) do
    xt = List.to_tuple(x)
    v = fn node -> (case idx(ni, node) do nil -> 0.0; i -> elem(xt, i) end) end
    # every element's current from its own equation, p → q through the element
    cur =
      Map.new(c.elements, fn e ->
        [p, q | _] = e.n
        i = case e.kind do
          :r -> (v.(p) - v.(q)) / e.value
          :c -> 0.0
          :d -> e.is * (:math.exp(min((v.(p) - v.(q)) / (e.nf * @vt), 80.0)) - 1)
          :x3 -> x3_current(e, v.(p), v.(q), v.(Enum.at(e.n, 2)))
          :i -> src_value(e, nil)
          :g -> e.gain * (v.(Enum.at(e.n, 2)) - v.(Enum.at(e.n, 3)))
          :f -> e.gain * elem(xt, bi[e.ctrl])
          _ -> elem(xt, bi[e.name])
        end
        {e.name, i}
      end)

    kcl = for nd <- c.nodes do
      Enum.reduce(c.elements, 0.0, fn e, s ->
        [p, q | _] = e.n
        i = cur[e.name]
        s + (if p == nd, do: i, else: 0.0) - (if q == nd and e.kind != :o, do: i, else: 0.0)
      end)
    end

    power = Map.new(c.elements, fn e -> [p, q | _] = e.n; {e.name, (v.(p) - v.(q)) * cur[e.name]} end)
    scale = power |> Map.values() |> Enum.map(&abs/1) |> Enum.max(fn -> 1.0 end) |> max(1.0e-15)
    # an op-amp's output current returns through its supply rails, which are not in the netlist: KCL there is not checked
    oa_out = for %{kind: :o, n: [o | _]} <- c.elements, do: o
    kcl_checked = for {nd, r} <- Enum.zip(c.nodes, kcl), nd not in oa_out, do: abs(r)

    %{nodes: Map.new(c.nodes, &{&1, v.(&1)}), currents: cur, power: power, iterations: its, x: x, note: note, warnings: dc_path_warnings(c),
      certificate: %{kcl_max: Enum.max(kcl_checked, fn -> 0.0 end), power_balance: if(oa_out == [], do: abs(power |> Map.values() |> Enum.sum()) / scale), op_amp_outputs: oa_out}}
  end

  # nodes reached from ground only through capacitors or current sources float at DC (held by gmin)
  defp dc_path_warnings(c) do
    edges = for e <- c.elements, e.kind not in [:c, :i, :g, :f], [p, q | _] = e.n, do: {p, q}
    edges = edges ++ for(%{kind: :o, n: [o | _]} <- c.elements, do: {o, "0"})
    reach = grow(MapSet.new(["0"]), edges)
    for nd <- c.nodes, not MapSet.member?(reach, nd), do: "node #{nd} has no DC path to ground: its operating-point voltage is set only by the 1 pS minimum conductance"
  end

  defp grow(set, edges) do
    next = Enum.reduce(edges, set, fn {a, b}, s -> if MapSet.member?(s, a) or MapSet.member?(s, b), do: s |> MapSet.put(a) |> MapSet.put(b), else: s end)
    if MapSet.size(next) == MapSet.size(set), do: set, else: grow(next, edges)
  end

  # ------------------------------------------------------------------- AC

  defp ac(c, opr, mode, pts, f1, f2) do
    lay = {ni, bi, n} = layout(c)
    freqs =
      case mode do
        "lin" -> for k <- 0..max(pts - 1, 1), do: f1 + (f2 - f1) * k / max(pts - 1, 1)
        m ->
          per = if m == "oct", do: :math.log(2), else: :math.log(10)
          total = max(1, round(:math.log(f2 / f1) / per * pts))
          for k <- 0..total, do: f1 * :math.exp(:math.log(f2 / f1) * k / total)
      end

    vop = fn node -> Map.get(opr.nodes, node, 0.0) end

    pts =
      for f <- freqs do
        w = 2 * :math.pi() * f
        {a, z} = ac_stamp(c, lay, w, vop)
        case Dense.csolve(a, z) do
          {:ok, x} ->
            %{f: f, nodes: Map.new(ni, fn {nm, i} -> {nm, polar(Enum.at(x, i))} end), branches: Map.new(bi, fn {nm, i} -> {nm, polar(Enum.at(x, i))} end)}
          _ -> %{f: f, nodes: nil}
        end
      end

    _ = n
    %{points: pts}
  end

  defp polar({re, im}), do: %{mag: :math.sqrt(re * re + im * im), db: 20 * :math.log10(max(:math.sqrt(re * re + im * im), 1.0e-300)), phase: :math.atan2(im, re) * 180 / :math.pi(), re: re, im: im}

  defp ac_stamp(c, {ni, bi, n}, w, vop) do
    cz = {0.0, 0.0}
    {a, z} =
      Enum.reduce(c.elements, {%{}, %{}}, fn e, {a, z} ->
        [p, q | _] = e.n
        {ip, iq} = {idx(ni, p), idx(ni, q)}
        y = fn g -> a |> cadd(ip, ip, g) |> cadd(iq, iq, g) |> cadd(ip, iq, neg(g)) |> cadd(iq, ip, neg(g)) end
        case e.kind do
          :r -> {y.({1 / e.value, 0.0}), z}
          :c -> {y.({1.0e-12, w * e.value}), z}
          :d -> (vd = vop.(p) - vop.(q); {y.({e.is * :math.exp(min(vd / (e.nf * @vt), 80.0)) / (e.nf * @vt) + 1.0e-12, 0.0}), z})
          :x3 ->
            r3 = Enum.at(e.n, 2); ir = idx(ni, r3)
            {_, gp, gq, gr} = x3_partials(e, vop.(p), vop.(q), vop.(r3))
            {a |> cadd(ip, ip, {gp, 0.0}) |> cadd(ip, iq, {gq, 0.0}) |> cadd(ip, ir, {gr, 0.0}) |> cadd(iq, ip, {-gp, 0.0}) |> cadd(iq, iq, {-gq, 0.0}) |> cadd(iq, ir, {-gr, 0.0}), z}
          :g -> (({cp, cq} = {idx(ni, Enum.at(e.n, 2)), idx(ni, Enum.at(e.n, 3))}); {a |> cadd(ip, cp, {e.gain, 0.0}) |> cadd(ip, cq, {-e.gain, 0.0}) |> cadd(iq, cp, {-e.gain, 0.0}) |> cadd(iq, cq, {e.gain, 0.0}), z})
          :i -> (s = phasor(e); {a, z |> caddz(ip, neg(s)) |> caddz(iq, s)})
          :f -> (cb = bi[e.ctrl]; {a |> cadd(ip, cb, {e.gain, 0.0}) |> cadd(iq, cb, {-e.gain, 0.0}), z})
          k ->
            b = bi[e.name]
            a = a |> cadd(ip, b, {1.0, 0.0}) |> cadd(iq, b, {-1.0, 0.0})
            case k do
              :v -> {a |> cadd(b, ip, {1.0, 0.0}) |> cadd(b, iq, {-1.0, 0.0}), Map.put(z, b, phasor(e))}
              :e -> (({cp, cq} = {idx(ni, Enum.at(e.n, 2)), idx(ni, Enum.at(e.n, 3))}); {a |> cadd(b, ip, {1.0, 0.0}) |> cadd(b, iq, {-1.0, 0.0}) |> cadd(b, cp, {-e.gain, 0.0}) |> cadd(b, cq, {e.gain, 0.0}), z})
              :h -> {a |> cadd(b, ip, {1.0, 0.0}) |> cadd(b, iq, {-1.0, 0.0}) |> cadd(b, bi[e.ctrl], {-e.gain, 0.0}), z}
              :l -> {a |> cadd(b, ip, {1.0, 0.0}) |> cadd(b, iq, {-1.0, 0.0}) |> cadd(b, b, {0.0, -w * e.value}), z}
              :o -> (([_, pp, mm] = e.n); {a |> Map.delete({iq, b}) |> cadd(b, idx(ni, pp), {1.0, 0.0}) |> cadd(b, idx(ni, mm), {-1.0, 0.0}), z})
            end
        end
      end)

    {for(i <- 0..(n - 1), do: for(j <- 0..(n - 1), do: Map.get(a, {i, j}, cz))), for(i <- 0..(n - 1), do: Map.get(z, i, cz))}
  end

  defp phasor(e), do: {e.ac * :math.cos(e.ac_phase * :math.pi() / 180), e.ac * :math.sin(e.ac_phase * :math.pi() / 180)}
  defp neg({a, b}), do: {-a, -b}
  defp cadd(m, nil, _, _), do: m
  defp cadd(m, _, nil, _), do: m
  defp cadd(m, i, j, v), do: Map.update(m, {i, j}, v, &Dense.cadd(&1, v))
  defp caddz(z, nil, _), do: z
  defp caddz(z, i, v), do: Map.update(z, i, v, &Dense.cadd(&1, v))

  # ---------------------------------------------------------- transient

  @doc false
  def tran(c, st, stop, tstart \\ 0.0) do
    lay = {ni, bi, _n} = layout(c)
    steps = round(stop / st)
    if steps > 200_000, do: throw({:netlist, ".tran: at most 200 000 steps (#{steps} asked)"})
    h = stop / steps
    # initial state: the operating point at t = 0, with capacitor/inductor ICs where given
    c0 = %{c | elements: Enum.map(c.elements, fn e -> if e.kind in [:v, :i], do: e, else: e end)}
    {:ok, o} = op_at(c0, lay, 0.0)
    xt = List.to_tuple(o)
    v = fn x, node -> (case idx(ni, node) do nil -> 0.0; i -> elem(x, i) end) end
    prev0 = Map.new(for e <- c.elements, e.kind in [:c, :l] do
      [p, q | _] = e.n
      case e.kind do
        :c -> {e.name, {e[:ic] || v.(xt, p) - v.(xt, q), 0.0}}
        :l -> {e.name, {0.0, e[:ic] || elem(xt, bi[e.name])}}
      end
    end)

    every = max(1, div(steps, 2000))
    {_, _, out} =
      Enum.reduce(1..steps, {o, prev0, [{0.0, o}]}, fn k, {x, prev, out} ->
        t = k * h
        meth = if k == 1, do: :be, else: c.method
        case newton(c, lay, x, {:tran, h, prev, meth}, t, 0) do
          {:ok, xn, _} ->
            xnt = List.to_tuple(xn)
            prev = Map.new(prev, fn {name, {vp, ip}} ->
              e = Enum.find(c.elements, &(&1.name == name))
              [p, q | _] = e.n
              vn = v.(xnt, p) - v.(xnt, q)
              case e.kind do
                :c -> {name, {vn, if(meth == :be, do: e.value / h * (vn - vp), else: 2 * e.value / h * (vn - vp) - ip)}}
                :l -> {name, {vn, elem(xnt, bi[name])}}
              end
            end)
            out = if rem(k, every) == 0 or k == steps, do: [{t, xn} | out], else: out
            {xn, prev, out}
          {:error, w} -> throw({:netlist, "transient failed at t = #{t}: #{w}"})
        end
      end)

    out = out |> Enum.reverse() |> Enum.filter(fn {t, _} -> t >= tstart end)
    {:ok, %{t: Enum.map(out, &elem(&1, 0)), nodes: Map.new(ni, fn {nm, i} -> {nm, Enum.map(out, fn {_, x} -> Enum.at(x, i) end)} end),
            branches: Map.new(bi, fn {nm, i} -> {nm, Enum.map(out, fn {_, x} -> Enum.at(x, i) end)} end), steps: steps, h: h,
            method: if(c.method == :be, do: "backward Euler", else: "trapezoidal (first step backward Euler)")}}
  catch
    {:netlist, w} -> {:error, w}
  end

  defp op_at(c, lay, t) do
    {_, _, n} = lay
    # the sources at t; capacitors open, inductors short
    cs = %{c | elements: Enum.map(c.elements, fn e -> if e.kind in [:v, :i], do: %{e | dc: src_value(e, t), wave: nil}, else: e end)}
    case newton(cs, lay, List.duplicate(0.0, n), :dc, t, 0) do
      {:ok, x, _} -> {:ok, x}
      _ -> with {:ok, r} <- op(cs), do: {:ok, r.x}
    end
  end
end
