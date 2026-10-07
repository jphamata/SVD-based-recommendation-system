defmodule Vapor.Studio.Video do
  @moduledoc "A video in the studio: frames (`Vapor.Modal.Image`, all one size) at `fps` frames per second, with optional audio."
  @enforce_keys [:fps, :frames]
  defstruct [:fps, :frames, audio: nil]

  def size(%__MODULE__{frames: [f | _]}), do: {f.w, f.h}
  def size(%__MODULE__{frames: []}), do: {0, 0}
end

defmodule Vapor.Studio.Value do
  @moduledoc """
  The types that flow along the studio's wires, and their digests.

  | type | Elixir value |
  |---|---|
  | `:image` | `Vapor.Modal.Image` (values in `[0, 1]`, channels last) |
  | `:mask` | `Vapor.Modal.Image` with one channel (`1` = selected) |
  | `:audio` | `Vapor.Modal.Audio` |
  | `:video` | `Vapor.Studio.Video` |
  | `:mesh` | `Vapor.Geom.Mesh` |
  | `:text` | a binary (UTF-8) |
  | `:number` | an integer or a float |
  | `:tensor`, `:latent` | `Vapor.Tensor` |
  | `:json` | any JSON-shaped term |

  A digest is the SHA-256 of a canonical byte form that includes the type
  and the shape, so two values with the same bytes and different meanings
  never collide (a 4×1 image is not a 1×4 one). Floats are hashed by their
  binary64 bits (images: the pixel tuple in Erlang's external term format,
  whose floats are exactly those bits, tagged): the digest of a result is a
  statement about its exact bits.
  """
  alias Vapor.Modal.{Audio, Image}
  alias Vapor.Studio.Video
  alias Vapor.Tensor

  @types [:image, :mask, :audio, :video, :mesh, :text, :number, :tensor, :latent, :json]

  @doc "The value types."
  def types, do: @types

  @doc "Whether `value` has type `type`."
  def is?(:image, %Image{}), do: true
  def is?(:mask, %Image{c: 1}), do: true
  def is?(:audio, %Audio{}), do: true
  def is?(:video, %Video{}), do: true
  def is?(:mesh, %{__struct__: Vapor.Geom.Mesh}), do: true
  def is?(:text, v) when is_binary(v), do: String.valid?(v)
  def is?(:number, v) when is_number(v), do: true
  def is?(t, %Tensor{}) when t in [:tensor, :latent], do: true
  def is?(:json, _), do: true
  def is?(_, _), do: false

  @doc "Whether an output of type `from` may feed an input of type `to`."
  def compatible?(t, t), do: true
  def compatible?(:mask, :image), do: true
  def compatible?(_, :json), do: true
  def compatible?(:latent, :tensor), do: true
  def compatible?(_, _), do: false

  @doc "SHA-256 (raw) of a value."
  def digest(v), do: :crypto.hash(:sha256, bytes(v))

  @doc "Hex digest."
  def hex(v), do: Base.encode16(digest(v), case: :lower)

  # the pixel tuple in Erlang's external term format: each value tagged and
  # 8 bytes of IEEE binary64, big-endian — built off-heap in one call, so a
  # long video is hashed without filling the process heap with garbage
  defp bytes(%Image{w: w, h: h, c: c, px: px}),
    do: ["image", <<w::32, h::32, c::8>>, :erlang.term_to_binary(px, minor_version: 1)] |> IO.iodata_to_binary()

  defp bytes(%Audio{rate: r, samples: s}),
    do: ["audio", <<r::32, length(s)::32>>, for(v <- s, into: <<>>, do: <<v * 1.0::float-64>>)] |> IO.iodata_to_binary()

  defp bytes(%Video{fps: fps, frames: frames, audio: a}),
    do: ["video", <<fps * 1.0::float-64, length(frames)::32>>, Enum.map(frames, &digest/1), if(a, do: digest(a), else: <<>>)] |> IO.iodata_to_binary()

  defp bytes(%Tensor{} = t), do: ["tensor", Atom.to_string(t.dtype), Enum.map(t.shape, &<<&1::32>>), t.data] |> IO.iodata_to_binary()
  defp bytes(%{__struct__: Vapor.Geom.Mesh} = m), do: "mesh" <> Vapor.Canonical.encode(Map.from_struct(m))
  defp bytes(v) when is_binary(v), do: "text" <> v
  defp bytes(v) when is_integer(v), do: "int" <> Integer.to_string(v)
  defp bytes(v) when is_float(v), do: <<"float", v::float-64>>
  defp bytes(v), do: "json" <> Vapor.Canonical.encode(v)

  @doc "A short, JSON-friendly description of a value (for receipts and the console)."
  def describe(%Image{w: w, h: h, c: c}), do: %{type: if(c == 1, do: "mask", else: "image"), width: w, height: h, channels: c}
  def describe(%Audio{rate: r, samples: s}), do: %{type: "audio", rate: r, samples: length(s), seconds: Float.round(length(s) / r, 3)}
  def describe(%Video{fps: fps, frames: f} = v), do: (fn {w, h} -> %{type: "video", fps: fps, frames: length(f), width: w, height: h} end).(Video.size(v))
  def describe(%Tensor{shape: s, dtype: d}), do: %{type: "tensor", shape: s, dtype: d}
  def describe(%{__struct__: Vapor.Geom.Mesh} = m), do: %{type: "mesh", vertices: div(length(m.vertices), 3), faces: div(length(m.faces), 3)}
  def describe(v) when is_binary(v), do: %{type: "text", bytes: byte_size(v), preview: String.slice(v, 0, 200)}
  def describe(v) when is_number(v), do: %{type: "number", value: v}
  def describe(_), do: %{type: "json"}
end
