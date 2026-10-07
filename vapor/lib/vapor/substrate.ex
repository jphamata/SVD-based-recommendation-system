defmodule Vapor.Substrate do
  @moduledoc """
  **The substrate airlock** — no accelerator is trusted by declaration.

  `Vapor.Lock` is where a model family is known, so that the core knows
  none; this is the same idea for hardware. A device enters the dispatcher
  only after running, on arrival, a battery of known-answer programs and
  having its results compared bit for bit with the exact oracle. The
  battery is designed so that each way a device can depart from the
  canonical semantics leaves a distinct fingerprint:

  | probe | what it isolates | how |
  |---|---|---|
  | `contraction` | `a·b + c` fused into one rounding | operands whose product is a rounding tie |
  | `subnormal_out` / `subnormal_in` | flush-to-zero / denormals-are-zero | a product that underflows; a subnormal operand |
  | `signed_zero`, `nan_select` | the sign of zero, NaN through comparisons and selections | `−0` arithmetic; `relu`, `max`, `sel` on NaN |
  | `reduction` | the order of a sum | a row whose sum depends on the association |
  | `input_precision` | operands rounded before a contraction (bf16, TF32…) | rows `1 + 2⁻ᵏ` through a GEMV |
  | `division`, `functions` | the canonical microprograms | hard cases, the elementary functions |
  | kernel families | every operator the dispatcher may send | linear f32/bf16, 4-bit, int8, attention, sampling |

  The verdict:

  * **`:canonical`** — every result equals the oracle's bits: the device
    joins the chain for every program.
  * **`:envelope`** — some results differ, but only in ways the rigorous
    envelope covers (fused multiply-add, another reduction order, flush to
    zero) and every differing output lies inside its bound: the device may
    run programs compiled under the `:fast` policy, never canonical ones.
  * **`:refused`** — anything else (operands rounded below binary32,
    integer kernels that do not wrap, an output outside its bound, a
    crash), with the reasons.

  The admission is a record — device, fingerprint, per-probe outcome,
  semantics version — encoded canonically and signable like a certificate
  (`attest/2`), so a cluster can carry it and a dossier can cite it.
  `Vapor.Runtime.Dispatch` consults `allowed?/2`.

  Why measure instead of reading the driver's flags: the flags describe
  intent (`MTLMathModeSafe`, `NoContraction`, `shaderDenormPreserveFloat32`);
  a bug, an older driver or a compiler that ignores a decoration is found
  only by running. And it is what makes a device that cannot be tested on
  the machine where vapor was written — a Mac, a Tenstorrent card — safe to
  ship for: the first run on it decides, and refuses itself if it differs.
  """
  alias Vapor.{Canonical, F32, Program, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Compile.Lower
  alias Vapor.Runtime.Native
  alias Vapor.Verify.Envelope

  @analytic ~w(contraction subnormal_out subnormal_in signed_zero reduction input_precision linear_f32 linear_bf16 gemm_i8)a

  defmodule Admission do
    @moduledoc "The outcome of `Vapor.Substrate.admit/2`."
    defstruct [:substrate, :device, :verdict, :fingerprint, :probes, :reasons, :semantics]
  end

  # ------------------------------------------------------------- probes --

  @doc """
  The battery: `[{name, program, env, run_opts}]`. Every program is small
  (the whole battery runs in seconds on a CPU) and deterministic.
  """
  def probes do
    tie = F32.from_float(1.0 + :math.pow(2, -12))
    neg_tie = F32.from_float(-(1.0 + :math.pow(2, -11)))
    row = fn bits_list -> Tensor.new(:f32, [length(bits_list)], for(b <- bits_list, into: <<>>, do: <<b::32-little>>)) end
    f = fn xs -> Tensor.from_list(:f32, [length(xs)], xs) end
    n16 = fn name -> T.input(name, :f32, [16]) end
    nan = 0x7FC0_0000

    # a sum whose value depends on the association: big terms that cancel,
    # small ones that survive only in some orders
    red = for i <- 0..255, do: if(rem(i, 3) == 0, do: :math.pow(2, 24) * (1 - 2 * rem(i, 2)), else: 1.0 + i / 512)

    # rows 1 + 2^-k (k = 1…23) times e_0: row k survives with k significand bits
    w_prec = for k <- 1..32, j <- 0..15, do: if(j == 0 and k <= 23, do: 1.0 + :math.pow(2, -k), else: 0.0)

    [
      {:contraction, Program.new(y: T.fma(n16.(:a), n16.(:b), n16.(:c))),
       %{a: row.(List.duplicate(tie, 16)), b: row.(List.duplicate(tie, 16)), c: row.(List.duplicate(neg_tie, 16))}, []},
      {:subnormal_out, Program.new(y: T.mul(n16.(:a), n16.(:b))),
       %{a: f.(List.duplicate(:math.pow(2, -100), 16)), b: f.(for(i <- 0..15, do: :math.pow(2, -30 - i)))}, []},
      {:subnormal_in, Program.new(y: T.add(n16.(:a), n16.(:b))),
       %{a: row.(for(i <- 1..16, do: i * 37)), b: f.(List.duplicate(0.0, 16))}, []},
      {:signed_zero, Program.new(y: T.add(n16.(:a), n16.(:b)), z: T.mul(n16.(:a), T.neg(n16.(:b))), n: T.neg(n16.(:a))),
       %{a: f.(List.duplicate(0.0, 8) ++ List.duplicate(-0.0, 8)), b: f.(List.duplicate(-0.0, 16))}, []},
      {:nan_select, Program.new(r: T.relu(n16.(:a)), m: T.max(n16.(:a), n16.(:b)), s: T.sel(n16.(:a), n16.(:b), n16.(:a), n16.(:b))),
       %{a: row.(List.duplicate(nan, 8) ++ for(i <- 1..8, do: F32.from_float(i * 1.0))), b: f.(for(i <- 0..15, do: 4.0 - i))}, []},
      {:reduction, Program.new(y: T.reduce(:sum, T.input(:x, :f32, [1, 256]))), %{x: Tensor.from_list(:f32, [1, 256], red)}, []},
      {:input_precision, Program.new(y: T.linear(T.input(:x, :f32, [1, 16]), T.const(Tensor.from_list(:f32, [32, 16], w_prec)))),
       %{x: Tensor.from_list(:f32, [1, 16], [1.0 | List.duplicate(0.0, 15)])}, []},
      {:division, Program.new(y: T.divide(n16.(:a), n16.(:b))),
       %{a: f.([1.0, 1.0, 2.0, 1.0e-38, 3.0e38, -7.0, 0.1, 1.0, 5.0, 1.0, -1.0, 1.0e10, 6.0, 1.0, 1.5, 2.5]),
         b: f.([3.0, 7.0, 3.0, 3.0, 0.5, 9.0, 0.3, 1.0e-38, 1.0e-40, 1.0e38, 49.0, 7.0, 1.0e-7, 11.0, 1.7, 0.1])}, []},
      {:functions, Program.new(e: T.exp(T.input(:x, :f32, [64])), s: T.silu(T.input(:x, :f32, [64])), r: T.rsqrt(T.mul(T.input(:x, :f32, [64]), T.input(:x, :f32, [64])))),
       %{x: Tensor.random(:f32, [64], 71, scale: 6.0)}, []},
      {:linear_f32, Program.new(y: T.linear(T.input(:x, :f32, [5, 80]), T.const(Tensor.random(:f32, [33, 80], 72)))),
       %{x: Tensor.random(:f32, [5, 80], 73)}, []},
      {:linear_bf16, Program.new(y: T.linear(T.input(:x, :f32, [3, 64]), T.const(Tensor.to_bf16(Tensor.random(:f32, [21, 64], 74, scale: 2.0))))),
       %{x: Tensor.random(:f32, [3, 64], 75)}, []},
      {:qgemv_sb4, Program.new(y: T.qgemv(T.const(Vapor.Quant.Sb4.quantize(Tensor.random(:f32, [32, 256], 76))), T.input(:x, :f32, [256]))),
       %{x: Tensor.random(:f32, [256], 77)}, []},
      {:gemm_i8, Program.new(c: T.gemm_i8(T.input(:a, :s8, [7, 96]), T.const(Tensor.random(:s8, [19, 96], 78)))),
       %{a: Tensor.random(:s8, [7, 96], 79)}, []},
      {:attention, attention_probe(), attention_env(), []},
      {:sample, Program.new(t: T.sample(T.input(:z, :f32, [3, 96]), T.input(:p, :f32, [3, 2]))),
       %{z: Tensor.random(:f32, [3, 96], 80, scale: 4.0),
         p: Tensor.from_list(:f32, [3, 2], [0.0, 0.3, 1.25, 0.61, 0.5, 0.07])}, []}
    ]
  end

  # gather → RMSNorm → q/k → RoPE → KV write → GQA attention
  defp attention_probe do
    {d, h, hkv, s, v} = {64, 4, 2, 16, 37}
    dh = div(d, h)
    half = div(dh, 2)
    ang = for p <- 0..(s - 1), i <- 0..(half - 1), do: p / :math.pow(10_000.0, 2 * i / dh)
    cos = T.const(Tensor.from_list(:f32, [s, half], Enum.map(ang, &:math.cos/1)))
    sin = T.const(Tensor.from_list(:f32, [s, half], Enum.map(ang, &:math.sin/1)))
    w = fn n, k, seed -> T.const(Tensor.random(:f32, [n, k], seed, scale: 0.25)) end
    tok = T.input(:tok, :s32, [4])
    pos = T.input(:pos, :s32, [4])
    x = T.gather_row(T.const(Tensor.random(:f32, [v, d], 81)), tok)
    xn = T.mul(x, T.rsqrt(T.add(T.mul(T.reduce(:sum, T.mul(x, x)), T.splat(1 / d)), T.splat(1.0e-5))))
    q = T.rope(T.linear(xn, w.(d, d, 82)), cos, sin, pos, h)
    k = T.rope(T.linear(xn, w.(hkv * dh, d, 83)), cos, sin, pos, hkv)
    vv = T.linear(xn, w.(hkv * dh, d, 84))
    kn = T.kv_write(T.input(:k, :f32, [s, hkv * dh]), pos, k)
    vn = T.kv_write(T.input(:v, :f32, [s, hkv * dh]), pos, vv)
    Program.new(y: T.attention(q, kn, vn, pos, h, hkv))
  end

  defp attention_env do
    zeros = Tensor.from_list(:f32, [16, 32], List.duplicate(0.0, 512))
    %{tok: Tensor.from_list(:s32, [4], [3, 11, 7, 36]), pos: Tensor.from_list(:s32, [4], [0, 1, 2, 3]), k: zeros, v: zeros}
  end

  # ------------------------------------------------------------ admission --

  @doc """
  Admit a substrate. `target` is a descriptor from
  `Vapor.Runtime.Substrates.list/0` (or any map with `:id`), and `opts`
  may give `runner: fun(compiled, env, run_opts) -> {:ok, result} | error`
  (default: `Vapor.Runtime.Dispatch.run_on/4` on the descriptor) and
  `device:` (a name; default: the fabric's reported name, else the id).
  Returns `%Admission{}`.
  """
  def admit(target, opts \\ []) do
    runner = Keyword.get(opts, :runner) || fn c, env, o -> Vapor.Runtime.Dispatch.run_on(target, c, env, o) end
    only = Keyword.get(opts, :only)
    # results gathered elsewhere (a kit run on a remote device): name → {:ok, outputs} | {:error, why}
    given = Keyword.get(opts, :results)

    results =
      for {name, prog, env, ro} <- probes(), only == nil or name in only, given == nil or Map.has_key?(given, name) do
        r = if given, do: (fn _c, _e, _o -> with({:ok, outs} <- given[name], do: {:ok, %{outputs: outs}}) end), else: runner
        {name, probe(name, prog, env, ro, r)}
      end

    fp = fingerprint(Map.new(results))
    {verdict, reasons} = verdict(results, fp)

    %Admission{substrate: target[:id], device: Keyword.get_lazy(opts, :device, fn -> device_name(target) end),
               verdict: verdict, fingerprint: fp, reasons: reasons, semantics: Vapor.Canon.version(),
               probes: Map.new(results, fn {n, r} -> {n, Map.drop(r, [:got, :want, :got_steps])} end)}
  end

  defp device_name(%{kind: :fabric, server: s}) when is_pid(s) do
    Vapor.Runtime.Fabric.info(s).name
  catch
    :exit, _ -> "fabric"
  end

  defp device_name(%{kind: :native, isa: isa, mode: mode}), do: "vapor-worker #{isa} (#{mode})"
  defp device_name(%{kind: :oracle}), do: "exact oracle (BEAM)"
  defp device_name(t), do: to_string(t[:id] || "substrate")

  defp probe(_name, prog, env, ro, runner) do
    with {:ok, c} <- Lower.lower(prog),
         {:ok, want} <- Native.run_oracle(c, env, ro) do
      names = prog.outputs |> Enum.map(&elem(&1, 0)) |> Enum.sort()

      # a device that crashes, answers out of protocol or leaves an output
      # out is refused, never a crash of the airlock itself
      case safe(fn -> well_formed(runner.(c, env, ro), names, want) end) do
        {:ok, got} ->
          diff = for n <- names, got.outputs[n].data != want.outputs[n].data, do: n

          envelope =
            if diff == [] do
              :equal
            else
              case safe(fn -> {:ok, Envelope.bounds(c, env, ro)} end) do
                {:ok, [bounds | _]} ->
                  per = for n <- diff, do: if(Envelope.analytic?(bounds[n] |> as_list()), do: Envelope.check(bounds[n], got.outputs[n]), else: :no_bound)
                  cond do
                    Enum.all?(per, &(&1 == :ok)) -> :within
                    Enum.any?(per, &(&1 == :no_bound)) -> :no_bound
                    true -> :outside
                  end

                _ ->
                  :no_bound
              end
            end

          %{equal: diff == [], differs: diff, envelope: envelope,
            ulps: Map.new(diff, fn n -> {n, max_ulps(got.outputs[n], want.outputs[n])} end),
            got: Map.new(names, &{&1, got.outputs[&1]}), want: Map.new(names, &{&1, want.outputs[&1]})}

        {:error, reason} ->
          %{equal: false, differs: :failed, envelope: :failed, error: inspect(reason, limit: 6)}
      end
    else
      err -> %{equal: false, differs: :failed, envelope: :failed, error: inspect(err, limit: 6)}
    end
  end

  defp well_formed({:ok, %{outputs: outs} = got}, names, want) do
    case Enum.find(names, fn n -> not match?(%Tensor{}, outs[n]) or outs[n].dtype != want.outputs[n].dtype or byte_size(outs[n].data) != byte_size(want.outputs[n].data) end) do
      nil -> {:ok, got}
      n -> {:error, "output #{n} missing or of the wrong type or size"}
    end
  end

  defp well_formed({:error, _} = e, _names, _want), do: e
  defp well_formed(other, _names, _want), do: {:error, "out-of-protocol answer: #{inspect(other, limit: 4)}"}

  defp as_list(%{elems: e}), do: e
  defp as_list(l) when is_list(l), do: l
  defp as_list(_), do: [:na]

  defp safe(f) do
    f.()
  rescue
    e -> {:error, Exception.message(e)}
  catch
    :exit, r -> {:error, {:exit, r}}
  end

  # largest distance in units in the last place over a tensor (f32), or
  # the count of differing elements for integer tensors
  defp max_ulps(%Tensor{dtype: :f32, data: a}, %Tensor{dtype: :f32, data: b}) do
    for({<<x::32-little>>, <<y::32-little>>} <- Enum.zip(chunks(a), chunks(b)), reduce: 0, do: (m -> max(m, abs(ord(x) - ord(y)))))
  end

  defp max_ulps(%Tensor{data: a}, %Tensor{data: b}),
    do: Enum.count(Enum.zip(chunks(a), chunks(b)), fn {x, y} -> x != y end)

  defp chunks(bin), do: for(<<w::binary-size(4) <- bin>>, do: w)
  # a total order on bit patterns where adjacent floats differ by one
  defp ord(b) when b >= 0x8000_0000, do: -(b - 0x8000_0000)
  defp ord(b), do: b

  # --------------------------------------------------------- fingerprint --

  defp fingerprint(r) do
    get = fn name -> Map.get(r, name, %{equal: :unmeasured, got: %{}, want: %{}}) end
    # a probe not run (an `only:` admission, a kit without it) says nothing
    says = fn name, yes, no -> case get.(name).equal do :unmeasured -> :unmeasured; true -> yes; false -> no end end

    contraction =
      case get.(:contraction) do
        %{equal: :unmeasured} -> :unmeasured
        %{equal: true} -> false
        %{got: %{y: %Tensor{data: <<b::32-little, _::binary>>}}} ->
          # the fused value: (1+2^-12)^2 - (1+2^-11) = 2^-24 exactly
          b == F32.from_float(:math.pow(2, -24))
        _ -> :unknown
      end

    zero? = fn %Tensor{data: d} -> for(<<w::32-little <- d>>, do: w) |> Enum.all?(&(&1 in [0, 0x8000_0000])) end

    ftz = case get.(:subnormal_out) do
      %{equal: :unmeasured} -> :unmeasured
      %{equal: true} -> false
      %{got: %{y: t}} -> zero?.(t)
      _ -> :unknown
    end

    daz = case get.(:subnormal_in) do
      %{equal: :unmeasured} -> :unmeasured
      %{equal: true} -> false
      %{got: %{y: t}} -> zero?.(t)
      _ -> :unknown
    end

    p_in =
      case get.(:input_precision) do
        %{got: %{y: %Tensor{} = t}} ->
          vals = Tensor.to_floats(t)
          # significand bits = 1 + the largest k whose row kept 1 + 2^-k
          kept = vals |> Enum.take(23) |> Enum.with_index(1) |> Enum.take_while(fn {v, k} -> v == 1.0 + :math.pow(2, -k) end) |> length()
          kept + 1

        _ ->
          :unknown
      end

    %{contraction: contraction, flush_to_zero: ftz, denormals_are_zero: daz,
      reduction_order: says.(:reduction, :canonical, :other),
      significand_bits: if(Map.has_key?(r, :input_precision), do: p_in, else: :unmeasured),
      signed_zero: says.(:signed_zero, :ieee, :other),
      nan_select: says.(:nan_select, :ieee, :other),
      division: says.(:division, :correctly_rounded, :other),
      functions: says.(:functions, :canonical, :other)}
  end

  # ------------------------------------------------------------- verdict --

  defp verdict(results, fp) do
    failed = for {n, %{differs: :failed} = r} <- results, do: "#{n}: #{r[:error]}"
    differ = for {n, %{equal: false, differs: d}} <- results, d != :failed, do: n

    cond do
      failed != [] ->
        {:refused, failed}

      differ == [] ->
        {:canonical, []}

      is_integer(fp.significand_bits) and fp.significand_bits < 24 ->
        {:refused, ["operands are rounded to #{fp.significand_bits} significand bits before a contraction (binary32 has 24)"]}

      true ->
        env = fn n -> results |> List.keyfind(n, 0) |> elem(1) |> Map.get(:envelope) end
        # outside a bound, wherever one exists; analytic probes must be inside theirs
        bad = for n <- differ, env.(n) == :outside or (n in @analytic and env.(n) != :within), do: n
        integer = for n <- differ, n in [:gemm_i8, :sample], do: n
        # a difference no bound covers is admitted only when the probes
        # measured a cause for it (a contraction, flushed subnormals, another
        # summation order); unexplained, it is a device computing something else
        cause? = fp.contraction == true or fp.flush_to_zero == true or fp.denormals_are_zero == true or fp.reduction_order == :other
        unbound = for n <- differ, env.(n) == :no_bound, do: n

        cond do
          integer != [] -> {:refused, ["integer kernels must be exact (#{Enum.join(integer, ", ")})"]}
          bad != [] -> {:refused, Enum.map(bad, &"#{&1}: outside the rigorous envelope")}
          unbound != [] and not cause? -> {:refused, Enum.map(unbound, &"#{&1}: differs, no bound covers it and no measured cause explains it")}
          true -> {:envelope, Enum.map(differ, &"#{&1}: differs from the oracle #{if env.(&1) == :within, do: "within the envelope", else: "(explained by the measured fingerprint)"}")}
        end
    end
  end

  # ----------------------------------------------------------- the record --

  @doc "The admission as a canonical term (what `attest/2` signs)."
  def record(%Admission{} = a) do
    {:vapor_admission, 1,
     %{"device" => a.device, "substrate" => to_string(a.substrate), "verdict" => to_string(a.verdict),
       "semantics" => a.semantics, "reasons" => a.reasons,
       "fingerprint" => Map.new(a.fingerprint, fn {k, v} -> {to_string(k), if(is_atom(v), do: to_string(v), else: v)} end),
       "probes" => Map.new(a.probes, fn {k, p} -> {to_string(k), %{"equal" => p.equal, "envelope" => to_string(p.envelope)}} end)}}
  end

  @doc "Canonical bytes of the record and an Ed25519 signature over them: `{bytes, signature}`."
  def attest(%Admission{} = a, %{private: priv}) do
    bytes = Canonical.encode(record(a))
    {bytes, :crypto.sign(:eddsa, :none, bytes, [priv, :ed25519])}
  end

  @doc "Check an attestation against a public key."
  def verify({bytes, sig}, pub), do: :crypto.verify(:eddsa, :none, bytes, sig, [pub, :ed25519])

  # ---------------------------------------------------------- the registry --

  defp arrive(%{kind: :fabric, server: s} = target) when is_pid(s) do
    if Application.get_env(:vapor, :admit_on_arrival, true) and node(s) == node() and Process.alive?(s),
      do: target |> admit() |> register(s)
  end

  defp arrive(_), do: nil

  @doc "Remember an admission for its substrate (by id and, for daemons, by process)."
  def register(%Admission{} = a, key \\ nil) do
    :persistent_term.put({__MODULE__, key || a.substrate}, a)
    a
  end

  @doc "The admission recorded for a substrate descriptor, or `nil`."
  def admission(target) do
    key = if is_pid(target[:server]), do: target.server, else: target[:id]
    :persistent_term.get({__MODULE__, key}, nil) || :persistent_term.get({__MODULE__, target[:id]}, nil)
  end

  @doc """
  May `target` run a program of `policy`? An accelerator (a Vulkan or
  Metal daemon) met for the first time is admitted on arrival — the
  battery runs once per daemon (`config :vapor, admit_on_arrival: false`
  turns this off). CPU tiers have no record: they are what the suite
  validates. A record decides: canonical programs need `:canonical`,
  `:fast` programs accept `:envelope`, a refused device runs nothing.
  """
  def allowed?(target, policy) do
    case admission(target) || arrive(target) do
      nil -> true
      %Admission{verdict: :canonical} -> true
      %Admission{verdict: :envelope} -> policy == :fast
      %Admission{verdict: :refused} -> false
    end
  end
end
