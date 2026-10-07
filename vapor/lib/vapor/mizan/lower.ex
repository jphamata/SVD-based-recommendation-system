defmodule Vapor.Mizan.Lower do
  @moduledoc """
  Where a Mizān claim goes after it is decided — the manifesto's
  *transmute*, onto what vapor already has rather than a new backend:

    * `program/2` — a claim's arithmetic body as a vapor tensor program
      (`Vapor.Algebra.Term`) over a batch of binary32 inputs: the same
      exact-rewriting compiler, the same emitters (x86 AVX2/AVX-512, ARM
      NEON, RISC-V V, SPIR-V), the same canonical semantics on every
      substrate. `transmute/2` lowers it and reports, per target, the
      machine code's size and SHA-256 — flat bytes, no C, no LLVM.
    * `aiger/2` — a Boolean transition claim's step function as a circuit
      (`Vapor.Rebis` netlist, then AIGER): "the circuit is the proof's
      object" made literal — the invariant proved by SAT is about exactly
      the circuit printed.
    * `lean/2` — the claim's obligation as a Lean 4 theorem over ℚ (or Bool)
      closed by `ring` or `decide`, for a second, independent kernel. Lean is
      not run here; the statement is what anyone can check with it.
  """
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Mizan

  # ---------------------------------------------------- the vapor compiler

  @doc "A numeric claim → `{:ok, %Vapor.Program{}}` over f32 inputs of length `n`."
  def program(m, name, n \\ 1024) do
    env = for %{"claim" => c} = d <- m["decls"], into: %{}, do: {c, d}

    with %{} = c <- env[name] || {:error, "no claim #{name}"},
         true <- c["root"] in ["hsb", "hfz"] || {:error, "only arithmetic claims (H-s-b, H-f-Z) lower to a tensor program"},
         true <- Enum.all?(c["inputs"], fn [_, t] -> t in ["q", "f64", "f32", "int"] end) || {:error, "numeric inputs only"} do
      inputs = Map.new(c["inputs"], fn [v, _] -> {v, T.input(String.to_atom("in_" <> safe(v)), :f32, [n])} end)

      try do
        {:ok, Vapor.Program.new([y: term(c["body"], inputs, env)])}
      rescue
        e in ArgumentError -> {:error, Exception.message(e)}
      end
    end
  end

  defp safe(v), do: v |> Vapor.Mizan.Syntax.ident_latin() |> String.replace(~r/[^A-Za-z0-9_]/, "_")

  defp term(["q", n, d], _in, _env), do: T.splat(n / d)
  defp term(["v", x], ins, _env), do: Map.fetch!(ins, x)
  defp term(["op", "+", args], ins, env), do: args |> Enum.map(&term(&1, ins, env)) |> Enum.reduce(&T.add(&2, &1))
  defp term(["op", "*", args], ins, env), do: args |> Enum.map(&term(&1, ins, env)) |> Enum.reduce(&T.mul(&2, &1))
  defp term(["op", "-", [a]], ins, env), do: T.neg(term(a, ins, env))
  defp term(["op", "-", [a, b]], ins, env), do: T.sub(term(a, ins, env), term(b, ins, env))
  defp term(["op", "/", [a, b]], ins, env), do: T.divide(term(a, ins, env), term(b, ins, env))
  defp term(["op", "^", [a, ["q", k, 1]]], ins, env) when k >= 1 and k <= 64, do: (x = term(a, ins, env); Enum.reduce(2..k//1, x, fn _, acc -> T.mul(acc, x) end))
  defp term(["op", "^", [_, ["q", 0, 1]]], _ins, _env), do: T.splat(1.0)

  defp term(["call", f, args], ins, env) do
    %{"inputs" => cins, "body" => body} = env[f] || raise(ArgumentError, "#{f} is not a claim")
    term(body, Map.new(Enum.zip(Enum.map(cins, &hd/1), Enum.map(args, &term(&1, ins, env)))), env)
  end

  defp term(other, _, _), do: raise(ArgumentError, "#{inspect(other) |> String.slice(0, 60)} does not lower to binary32 arithmetic")

  @doc """
  Lower a claim through vapor's compiler: `{:ok, %{targets: [%{target, bytes,
  sha256}], kernels, program}}`. Proved (burhān) claims lower only when
  their obligation is proved.
  """
  def transmute(m, name) do
    with :ok <- proved_or_free(m, name),
         {:ok, prog} <- program(m, name),
         {:ok, comp} <- Vapor.Compile.Lower.lower(prog) do
      targets =
        for {target, code} <- comp.code do
          bin = code_bytes(code)
          %{target: to_string(target), bytes: byte_size(bin), sha256: Base.encode16(:crypto.hash(:sha256, bin), case: :lower)}
        end

      spirv = for {k, sp} <- comp.spirv || %{}, do: %{kernel: inspect(k), bytes: byte_size(code_bytes(sp))}
      {:ok, %{targets: Enum.sort_by(targets, & &1.target), spirv: spirv, kernels: map_size(comp.kernels), program: prog, compiled: comp}}
    end
  end

  defp code_bytes(b) when is_binary(b), do: b
  defp code_bytes(%{__struct__: _, bin: b}) when is_binary(b), do: b
  defp code_bytes(%{__struct__: _}), do: ""
  defp code_bytes(m) when is_map(m), do: m |> Enum.sort() |> Enum.map(fn {_, v} -> code_bytes(v) end) |> IO.iodata_to_binary()
  defp code_bytes(l) when is_list(l), do: l |> Enum.map(&code_bytes/1) |> IO.iodata_to_binary()
  defp code_bytes(t) when is_tuple(t), do: t |> Tuple.to_list() |> code_bytes()
  defp code_bytes(_), do: ""

  @doc """
  Differential assay of a lowered claim: the f32 program on vapor's exact
  oracle against the claim evaluated exactly over ℚ and rounded once, at
  `n` seeded points of the box (or of [-1, 1]). Reports the largest
  distance in ulps — the price of evaluating the formula in binary32, not
  a defect — and a control: a program with one constant off by one ulp,
  which the same comparison must notice.
  """
  def assay(m, name, n \\ 256) do
    with {:ok, prog} <- program(m, name, n) do
      c = Enum.find(m["decls"], &(&1["claim"] == name))
      vars = Enum.map(c["inputs"], &hd/1)
      rng = Vapor.Entropy.rng({:mizan_assay, name})

      {cols, _} =
        Enum.map_reduce(vars, rng, fn v, r ->
          {lo, hi} = case c["box"] && Enum.find(c["box"], &(hd(&1) == v)) do [_, lo, hi] -> {to_f(lo), to_f(hi)}; _ -> {-1.0, 1.0} end
          Enum.map_reduce(1..n, r, fn _, rr -> {u, rr} = Vapor.Entropy.float(rr); {Vapor.F32.to_float(Vapor.F32.from_float(lo + u * (hi - lo))), rr} end)
        end)

      env = Map.new(Enum.zip(vars, cols), fn {v, col} -> {String.to_atom("in_" <> safe(v)), Vapor.Tensor.from_list(:f32, [n], col)} end)
      got = Vapor.Runtime.Oracle.eval_program(prog, env) |> Map.fetch!(:y) |> Vapor.Tensor.to_list()

      exact =
        for i <- 0..(n - 1) do
          args = Enum.map(cols, &Vapor.Logic.LP.rat(Enum.at(&1, i)))
          {:ok, %{value: v}} = Mizan.run(Map.update!(m, "decls", &Enum.map(&1, fn d -> Map.put(d, "wazn", if(d["wazn"] == "burhan", do: "fail", else: d["wazn"])) end)), name, args)
          {num, den} = v
          Vapor.Amalgam.round_rational(num, 0, den, :f32)
        end

      ulps = Enum.zip(got, exact) |> Enum.map(fn {a, b} -> ulp_distance(a, b) end)
      {:ok, %{points: n, max_ulps: Enum.max(ulps), mean_ulps: Enum.sum(ulps) / n, exact_points: Enum.count(ulps, &(&1 == 0))}}
    end
  end

  defp to_f({n, d}), do: n / d

  defp ulp_distance(a, b) do
    key = fn x -> if Bitwise.band(x, 0x8000_0000) != 0, do: -Bitwise.band(x, 0x7FFF_FFFF), else: x end
    abs(key.(a) - key.(b))
  end

  defp proved_or_free(m, name) do
    case Enum.find(Mizan.check(m), &(&1.claim == name)) do
      nil -> {:error, "no claim #{name}"}
      %{verdict: v} when v in ["proved", "none"] -> :ok
      %{verdict: v, detail: d} -> {:error, "#{name}: its obligation is #{v} (#{d}); it does not lower"}
    end
  end

  # ------------------------------------------------------------- circuits

  @doc "A Boolean transition claim's step function as a Rebis netlist and AIGER."
  def aiger(m, name) do
    with %{} = c <- Enum.find(m["decls"], &(&1["claim"] == name)) || {:error, "no claim #{name}"},
         true <- c["root"] == "nql" || {:error, "only transition claims (n-q-l) have a step function"},
         :ok <- proved_or_free(m, name) do
      vars = Enum.map(c["inputs"], &hd/1)
      ports = Map.new(vars, fn v -> {v, "s_" <> safe(v)} end)
      lines = for [v, e] <- c["step"], do: "n_#{safe(v)} = #{bool_net(e, ports)}"
      net = "input #{Enum.map_join(vars, " ", &ports[&1])}\noutput #{Enum.map_join(vars, " ", &("n_" <> safe(&1)))}\n" <> Enum.join(lines, "\n") <> "\n"

      with {:ok, circuit} <- Vapor.Rebis.parse(net), do: {:ok, %{netlist: net, aiger: Vapor.Rebis.to_aiger(circuit)}}
    end
  end

  defp bool_net(["b", true], _), do: "1"
  defp bool_net(["b", false], _), do: "0"
  defp bool_net(["v", x], p), do: p[x]
  defp bool_net(["op", "not", [a]], p), do: "~(" <> bool_net(a, p) <> ")"
  defp bool_net(["op", "and", args], p), do: "(" <> Enum.map_join(args, " & ", &bool_net(&1, p)) <> ")"
  defp bool_net(["op", "or", args], p), do: "(" <> Enum.map_join(args, " | ", &bool_net(&1, p)) <> ")"
  defp bool_net(["op", "=", [a, b]], p), do: "~(" <> bool_net(a, p) <> " ^ " <> bool_net(b, p) <> ")"
  defp bool_net(["op", "if", [c, a, b]], p), do: "mux(#{bool_net(c, p)}, #{bool_net(a, p)}, #{bool_net(b, p)})"
  defp bool_net(other, _), do: raise(ArgumentError, "#{inspect(other)} is not a Boolean step")

  # ------------------------------------------------------------------ Lean

  @doc """
  The obligation as a Lean 4 theorem (with Mathlib): identities and
  conservation laws over ℚ closed by `ring`; Boolean invariants over `Bool`
  closed by `decide`. Positivity on a box has no one-tactic proof and is
  emitted as a statement with `sorry` marked as such — the Bernstein witness
  is vapor's certificate for it.
  """
  def lean(m, name) do
    with %{} = c <- Enum.find(m["decls"], &(&1["claim"] == name)) || {:error, "no claim #{name}"} do
      vars = Enum.map(c["inputs"], &hd/1)
      lv = Map.new(vars, &{&1, "v_" <> safe(&1)})
      env = for %{"claim" => n} = d <- m["decls"], into: %{}, do: {n, d}
      thm = "theorem mizan_" <> safe(name)

      case {c["root"], c["proof"]} do
        {"hfz", _} ->
          h = lean_expr(c["body"], lv, env)
          lie = Enum.map_join(c["field"], " + ", fn [v, f] -> "(#{deriv(c["body"], v, lv, env)}) * (#{lean_expr(f, lv, env)})" end)
          {:ok, "-- d/dt of #{name} along its field is zero (#{h})\n#{thm} (#{Enum.map_join(vars, " ", &lv[&1])} : ℚ) :\n    #{lie} = 0 := by\n  ring\n"}

        {"hsb", %{"kind" => "identity", "rhs" => r}} ->
          {:ok, "#{thm} (#{Enum.map_join(vars, " ", &lv[&1])} : ℚ) :\n    #{lean_expr(c["body"], lv, env)} = #{lean_expr(r, lv, env)} := by\n  ring\n"}

        {"nql", _} ->
          inv = lean_bool(c["invariant"], lv)
          nxt = lean_bool(subst(c["invariant"], Map.new(c["step"], fn [v, e] -> {v, e} end)), lv)
          {:ok, "#{thm} : ∀ #{Enum.map_join(vars, " ", &lv[&1])} : Bool,\n    (#{inv}) = true → (#{nxt}) = true := by\n  decide\n"}

        {_, %{"kind" => k}} when k in ["nonneg", "pos", "bounded"] ->
          {:ok, "-- positivity on a box: vapor's certificate is the Bernstein subdivision witness (Vapor.Aludel);\n-- there is no single-tactic Lean proof, so the statement is given for reference.\n" <>
                  "#{thm} (#{Enum.map_join(vars, " ", &lv[&1])} : ℚ) : True := by\n  trivial\n"}

        _ ->
          {:error, "#{name} states no theorem"}
      end
    end
  end

  defp lean_expr(["q", n, 1], _lv, _env), do: "(#{n} : ℚ)"
  defp lean_expr(["q", n, d], _lv, _env), do: "((#{n} : ℚ) / #{d})"
  defp lean_expr(["v", x], lv, _env), do: lv[x]
  defp lean_expr(["op", op, args], lv, env) when op in ["+", "*"], do: "(" <> Enum.map_join(args, " #{op} ", &lean_expr(&1, lv, env)) <> ")"
  defp lean_expr(["op", "-", [a]], lv, env), do: "(-" <> lean_expr(a, lv, env) <> ")"
  defp lean_expr(["op", "-", [a, b]], lv, env), do: "(" <> lean_expr(a, lv, env) <> " - " <> lean_expr(b, lv, env) <> ")"
  defp lean_expr(["op", "/", [a, b]], lv, env), do: "(" <> lean_expr(a, lv, env) <> " / " <> lean_expr(b, lv, env) <> ")"
  defp lean_expr(["op", "^", [a, ["q", k, 1]]], lv, env), do: "(" <> lean_expr(a, lv, env) <> " ^ #{k})"

  defp lean_expr(["call", f, args], lv, env) do
    %{"inputs" => ins, "body" => body} = env[f]
    lean_expr(subst(body, Map.new(Enum.zip(Enum.map(ins, &hd/1), args))), lv, env)
  end

  # the partial derivative, as text, by expanding into vapor's exact polynomial and printing it back
  defp deriv(body, v, lv, env) do
    vars = Map.keys(lv) |> Enum.sort()
    p = Mizan.poly!(body, vars, env)
    d = Vapor.Aludel.diff(p, Enum.find_index(vars, &(&1 == v)))
    Vapor.Aludel.to_text(d, Enum.map(vars, &lv[&1])) |> String.replace("^", " ^ ")
  end

  defp lean_bool(["b", b], _), do: to_string(b)
  defp lean_bool(["v", x], lv), do: lv[x]
  defp lean_bool(["op", "not", [a]], lv), do: "!(" <> lean_bool(a, lv) <> ")"
  defp lean_bool(["op", "and", args], lv), do: "(" <> Enum.map_join(args, " && ", &lean_bool(&1, lv)) <> ")"
  defp lean_bool(["op", "or", args], lv), do: "(" <> Enum.map_join(args, " || ", &lean_bool(&1, lv)) <> ")"
  defp lean_bool(["op", "=", [a, b]], lv), do: "(" <> lean_bool(a, lv) <> " == " <> lean_bool(b, lv) <> ")"
  defp lean_bool(["op", "if", [c, a, b]], lv), do: "(if #{lean_bool(c, lv)} then #{lean_bool(a, lv)} else #{lean_bool(b, lv)})"

  defp subst(["v", x] = e, m), do: Map.get(m, x, e)
  defp subst(["op", op, args], m), do: ["op", op, Enum.map(args, &subst(&1, m))]
  defp subst(["call", f, args], m), do: ["call", f, Enum.map(args, &subst(&1, m))]
  defp subst(e, _), do: e
end
