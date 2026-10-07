defmodule Vapor.KIR do
  @moduledoc """
  Portable kernel IR — the single description every backend selects from.

  A kernel is a linear list of instructions over *virtual registers*
  `{:vr, id, kind}`; kinds are substrate-neutral and are sized by each backend
  (`g` is the register-group factor):

  | kind      | meaning                          | RVV (LMUL)  | AVX2 (ymm) | NEON (q) |
  |-----------|----------------------------------|-------------|------------|----------|
  | `:gpr`    | 64-bit integer / pointer         | x-reg       | r64        | x-reg    |
  | `:fpr`    | scalar binary32                  | f-reg       | xmm lane 0 | s-reg    |
  | `:strip`  | f32 strip of a strip-mined loop  | m`g`        | `g`        | `g`      |
  | `:f16l`   | exactly 16 binary32 lanes        | m4          | 2          | 4        |
  | `:i32acc` | s32 accumulator of an i8 strip   | m`4g`       | `g`        | 2        |

  The group factor generalises RVV's LMUL to every ISA: an AVX2 strip with
  `g = 4` is four ymm registers processed in lockstep. The cut sweep picks
  the largest `g` whose allocation succeeds (never spilling).

  Control flow is explicit (`{:label, l}`, branches) except for one
  structured form, `{:strip, n, ptrs, body}` — "process `n` f32 elements,
  advancing every pointer in `ptrs` by 4 bytes per element" — which each
  backend expands into its own strip-mining discipline (RVV `vsetvli`
  loops; AVX2/NEON vector loop plus a scalar tail on lane 0 of the same
  registers). `{:strip_i8, n, ptrs, body, tail}` is the byte-stride variant
  with an explicit scalar tail.

  ## Instruction set

      {:arg, x, i}                 x ← args[i] (u64 argument block)
      {:li, x, imm}  {:mov, x, y}  {:addi, x, y, imm}  {:add|:sub|:mul, x, y, z}
      {:label, l}  {:jmp, l}  {:bnez|:beqz, x, l}  {:blt_imm, x, imm, l}  (signed x < imm)
      {:ld_u8|:ld_s8, x, base, off}  {:st_i32, x, base, off}
      {:ldf, f, base, off}  {:stf, f, base, off}  {:lif, f, bits}  {:cvt_u8f, f, x}
      {:fadd|:fsub|:fmul, f, a, b}  {:fmacc, acc, a, b}   (policy-aware acc += a·b)
      {:vld|:vst, v, base, off}  {:vsplat, v, f}  {:vfadd|:vfsub|:vfmul, v, a, b}
      {:vfma, v, a, b, c}  {:vfneg|:vrelu, v, a}          (strip kind)
      {:vzero16, v}  {:vld_nib, lo, hi, base, off}  {:vmul_sf, v, a, f}
      {:vfmacc_mem, acc, w, base, off}  {:vld16, v, base, off}  {:vfadd16, v, a, b}
      {:vld_bf16, v, base, off}          (16 bfloat16 → 16 binary32, exact; f16l kind)
      {:vreduce16, f, v}                                  (canonical tree, f16l kind)
      {:vzero_i32, acc}  {:vi8mac, acc, pa, pb}  {:vred_i32, x, acc}
      :ret
  """

  @type vreg :: {:vr, non_neg_integer, atom}

  def gpr(i), do: {:vr, i, :gpr}
  def fpr(i), do: {:vr, i, :fpr}
  def vec(i, kind), do: {:vr, i, kind}

  @doc "Largest register id used (backends number their temporaries above it)."
  def max_id(code) do
    code
    |> Enum.flat_map(&regs/1)
    |> Enum.map(fn {:vr, i, _} -> i end)
    |> Enum.max(fn -> 0 end)
  end

  @doc "Every virtual register mentioned by a portable instruction (recursing into strips)."
  def regs({:strip, n, ptrs, body}), do: [n | ptrs] ++ Enum.flat_map(body, &regs/1)

  def regs({:strip_i8, n, ptrs, body, tail}),
    do: [n | ptrs] ++ Enum.flat_map(body ++ tail, &regs/1)

  def regs(inst) when is_tuple(inst), do: inst |> Tuple.to_list() |> Enum.filter(&match?({:vr, _, _}, &1))
  def regs(_), do: []
end

defmodule Vapor.KIR.Kernel do
  @moduledoc "A portable kernel: name, argument roles, IR code, metadata."
  @enforce_keys [:name, :args, :code]
  defstruct [:name, :args, :code, meta: %{}]
end
