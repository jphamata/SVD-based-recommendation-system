defmodule Vapor.Lock.Adapters.Codec do
  @moduledoc """
  Tier 3, a **topology**: the vector-quantised codec — the bridge between a
  continuous modality (image patches, audio frames, encoder rows) and the
  discrete tokens a decoder can read and write. With it, *any-to-any* is a
  vocabulary question: every modality becomes ids in one shared space.

  Both directions are programs of the algebra:

      encode:  codes = argmax_j (2·x·c_j − ‖c_j‖²)  =  argmin_j ‖x − c_j‖²
               — a contraction, a constant row, and the `sample` operator in
                 its greedy mode (ties to the lower index)
      decode:  rows  = codebook[codes]          — `gather_row`, exact

  so a code is the same on every substrate, and decoding is bit-exact.
  The codebook size is padded to a multiple of 16 with entries whose score
  is −FLT_MAX, which are never chosen.

  `config.json`: `{"model_type": "vapor_vq", "codebook_size": K,
  "row_width": k, "modality": "image" | "audio" | "rows", …}` (extra keys —
  patch geometry, sample rate — travel in `raw` for the modality codecs);
  weights: `codebook : f32[K, k]`. `Vapor.Modal.VQ.fit/3` makes one.

  Build options: `direction: :encode | :decode` (default `:encode`), `rows`
  (static row count, default 64).
  """
  @behaviour Vapor.Lock.Adapter
  alias Vapor.{F32, Program, Rejection, Tensor}
  alias Vapor.Algebra.Term, as: T
  alias Vapor.Lock.Spec

  @neg_max -3.4028234663852886e38

  defmodule Config do
    @moduledoc "An admitted VQ codec configuration."
    defstruct [:size, :row_width, :modality, :raw]
  end

  @impl true
  def id, do: "codec"

  @impl true
  def claim(%{config: %{"model_type" => "vapor_vq"}}), do: {:claim, 100}
  def claim(_), do: :no

  @impl true
  def owns?(%Config{}), do: true
  def owns?(_), do: false

  @impl true
  def admit(%{config: c}, ws, _opts) do
    k = c["codebook_size"]
    w = c["row_width"]
    m = c["modality"] || "rows"

    cond do
      not (is_integer(k) and k > 1) -> no("codebook_size", "an integer > 1")
      not (is_integer(w) and w > 0 and rem(w, 16) == 0) -> no("row_width", "a positive multiple of 16")
      m not in ~w(rows image audio) -> no("modality", "rows, image or audio")
      not match?(%Tensor{shape: [^k, ^w]}, ws["codebook"]) -> {:error, Rejection.new({:weight, "codebook"}, "f32[#{k}, #{w}]", "check the checkpoint")}
      true ->
        cfg = %Config{size: k, row_width: w, modality: String.to_atom(m), raw: Map.drop(c, ~w(torch_dtype dtype))}
        {:ok, spec(cfg), ws}
    end
  end

  @impl true
  def spec(%Config{} = c) do
    %Spec{adapter: __MODULE__, family: "vapor_vq", lineage: ["vapor_vq", "codec"], interface: :codec, config: c,
          vocab: c.size, width: c.row_width, in_width: c.row_width, features: [:encode, :decode],
          modality: %{in: [c.modality, :codes], out: [:codes, c.modality]},
          digest: Vapor.Canonical.hex_digest({:codec, c.raw})}
  end

  @impl true
  def expected(%Spec{config: c}), do: [{"codebook", [c.size, c.row_width], :matrix}]

  @impl true
  def build(%Spec{config: c}, ws, opts) do
    t = Keyword.get(opts, :rows, 64)
    book = Tensor.widen(ws["codebook"])
    kp = div(c.size + 15, 16) * 16
    cb = T.ref(:codebook, T.const(book))

    case Keyword.get(opts, :direction, :encode) do
      :encode ->
        rows = book |> Tensor.to_floats() |> Enum.chunk_every(c.row_width)
        # −‖c_j‖² (binary64, rounded once), −FLT_MAX on the padding
        bias = Enum.map(rows, fn r -> -Enum.reduce(r, 0.0, &(&1 * &1 + &2)) end) ++ List.duplicate(@neg_max, kp - c.size)
        twice = Tensor.new(:f32, [kp, c.row_width], (for x <- Tensor.to_floats(book), into: <<>>, do: <<F32.from_float(2 * x)::32-little>>) <> :binary.copy(<<0::32>>, (kp - c.size) * c.row_width))
        x = T.input(:rows, :f32, [t, c.row_width])
        scores = T.add(T.linear(x, T.ref(:twice, T.const(twice))), T.ref(:neg_norm, T.const(Tensor.from_list(:f32, [1, kp], bias))))
        greedy = T.ref(:greedy, T.const(Tensor.from_list(:f32, [t, 2], List.duplicate(0.0, 2 * t))))

        {:ok, Program.new([codes: T.sample(scores, greedy)],
                          lets: [twice: T.const(twice), neg_norm: T.const(Tensor.from_list(:f32, [1, kp], bias)),
                                 greedy: T.const(Tensor.from_list(:f32, [t, 2], List.duplicate(0.0, 2 * t)))])}

      :decode ->
        codes = T.input(:codes, :s32, [t])
        {:ok, Program.new([rows: T.gather_row(cb, codes)], lets: [codebook: T.const(book)])}

      other ->
        {:error, Rejection.new({:codec, :direction}, ":encode or :decode (got #{inspect(other)})", "pick a direction")}
    end
  end

  defp no(field, bound), do: {:error, Rejection.new({:config, field}, bound, "check the config")}
end
