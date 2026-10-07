defmodule Vapor.Studio.Export do
  @moduledoc """
  Bytes for a studio value, in a format other tools open: images as PNG (or
  PPM), video as animated GIF (or Y4M for ffmpeg), audio as 16-bit WAV,
  meshes as GLB/OBJ/PLY, text as UTF-8, anything else as JSON.
  """
  alias Vapor.Modal.{Audio, Image}
  alias Vapor.Studio.{Value, Video}

  @doc "`{mime, extension, bytes}` for `value` in `format` (`:auto` picks the default)."
  def encode(value, format \\ :auto)
  def encode(%Image{} = img, f) when f in [:auto, "png"], do: {"image/png", "png", Image.png(img)}
  def encode(%Image{} = img, "ppm"), do: {"image/x-portable-anymap", if(img.c == 3, do: "ppm", else: "pgm"), Image.encode(img)}
  def encode(%Video{} = v, f) when f in [:auto, "gif"], do: {"image/gif", "gif", Vapor.Media.GIF.encode(v.frames, fps: v.fps)}
  def encode(%Video{} = v, "y4m"), do: {"video/x-yuv4mpeg", "y4m", Vapor.Media.Video.y4m(v)}
  def encode(%Audio{} = a, f) when f in [:auto, "wav"], do: {"audio/wav", "wav", Audio.encode(a)}
  def encode(%{__struct__: Vapor.Geom.Mesh} = m, f) when f in [:auto, "glb"], do: {"model/gltf-binary", "glb", Vapor.Geom.glb(m)}
  def encode(%{__struct__: Vapor.Geom.Mesh} = m, "obj"), do: {"text/plain", "obj", Vapor.Geom.obj(m)}
  def encode(%{__struct__: Vapor.Geom.Mesh} = m, "ply"), do: {"application/octet-stream", "ply", Vapor.Geom.ply(m)}
  def encode(t, _) when is_binary(t), do: {"text/plain; charset=utf-8", "txt", t}
  def encode(v, _), do: {"application/json", "json", Vapor.JSON.encode(Vapor.Quality.Report.plain(Value.describe(v) |> Map.put(:value, plainable(v))))}

  defp plainable(%Vapor.Tensor{} = t), do: Vapor.Tensor.to_floats(t) |> Enum.take(4096)
  defp plainable(v) when is_number(v), do: v
  defp plainable(v), do: Vapor.Quality.Report.plain(v)

  @doc "A data URI for previews (images, GIFs, WAVs; anything else as JSON)."
  def data_uri(value, format \\ :auto) do
    {mime, _, bytes} = encode(value, format)
    "data:" <> mime <> ";base64," <> Base.encode64(bytes)
  end
end
