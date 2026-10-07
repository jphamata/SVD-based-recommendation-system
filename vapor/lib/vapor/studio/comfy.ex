defmodule Vapor.Studio.Comfy do
  @moduledoc """
  ComfyUI workflows (the **API format**, "Save (API)": `{"id": {"class_type",
  "inputs"}}`) translated into studio graphs.

  Only nodes whose meaning is pinned down here are translated, and each
  translation is stated (`notes`):

  | ComfyUI | studio |
  |---|---|
  | `LoadImage` (image) | `image.load` (path inside the studio's directory) |
  | `SaveImage`, `PreviewImage` | `studio.output` |
  | `EmptyImage` (width, height, batch 1, colour 0xRRGGBB) | `image.solid` |
  | `ImageScale` (method, width, height, crop) | `image.resize` (nearest-exact → nearest; torch semantics for bilinear/bicubic/area, Pillow for Lanczos) |
  | `ImageScaleBy` | `image.scale_by` |
  | `ImageInvert` | `image.invert` |
  | `ImageCrop` | `image.crop` |
  | `ImageBlur` (radius r, sigma σ) | `image.blur` with σ·r pixels — ComfyUI lays its kernel on [−1, 1], so its σ is in units of the radius; the Gaussian is separable, so the 2-D kernel is the product of two 1-D ones |
  | `ImageCompositeMasked` (x, y, resize_source false) | `image.composite` |
  | `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `VAEEncode` | the diffusion nodes, when present (`Vapor.Studio.Nodes.Diffusion`) |

  Everything else is refused, every unsupported node named in one
  rejection — a workflow is never run half-translated.
  """
  alias Vapor.Rejection

  @methods %{"nearest-exact" => "nearest", "bilinear" => "bilinear", "bicubic" => "bicubic", "area" => "area", "lanczos" => "lanczos"}

  @doc "Translate a ComfyUI API-format prompt (map or JSON text): `{:ok, graph_json, notes}` or a rejection."
  def import(text) when is_binary(text), do: with({:ok, m} <- Vapor.JSON.decode(text), do: __MODULE__.import(m))

  def import(%{"prompt" => p}) when is_map(p), do: __MODULE__.import(p)

  def import(prompt) when is_map(prompt) do
    # each node's outputs by index, so a link [src, i] names the right port
    ports = Map.new(prompt, fn {id, n} -> {to_string(id), outputs_of(n["class_type"])} end)
    Process.put({__MODULE__, :ports}, ports)
    results = Enum.map(Enum.sort_by(prompt, fn {id, _} -> id end), fn {id, n} -> {id, translate(id, n)} end)
    Process.delete({__MODULE__, :ports})

    case for {id, {:error, why}} <- results, do: "#{id} (#{why})" do
      [] ->
        nodes = for {id, {:ok, node, _}} <- results, into: %{}, do: {id, node}
        notes = for {id, {:ok, _, note}} <- results, note != nil, do: "#{id}: #{note}"
        {:ok, %{"nodes" => nodes}, notes}

      bad ->
        {:error, Rejection.new({:comfy, :unsupported}, "nodes with a translation (#{Enum.join(Map.keys(table()), ", ")})", "not translated: " <> Enum.join(bad, "; "))}
    end
  end

  def import(_), do: {:error, Rejection.new(:comfy, "a ComfyUI API-format prompt (an object of {class_type, inputs})", "export with \"Save (API)\"")}

  defp table do
    %{"LoadImage" => true, "SaveImage" => true, "PreviewImage" => true, "EmptyImage" => true, "ImageScale" => true, "ImageScaleBy" => true,
      "ImageInvert" => true, "ImageCrop" => true, "ImageBlur" => true, "ImageCompositeMasked" => true}
    |> Map.merge(if Code.ensure_loaded?(Vapor.Studio.Nodes.Diffusion), do: Map.new(Vapor.Studio.Nodes.Diffusion.comfy_types(), &{&1, true}), else: %{})
  end

  defp outputs_of("LoadImage"), do: ["image", "alpha_mask"]
  defp outputs_of(t) when t in ["EmptyImage", "ImageScale", "ImageScaleBy", "ImageInvert", "ImageCrop", "ImageBlur", "ImageCompositeMasked"], do: ["image"]

  defp outputs_of(t) do
    if Code.ensure_loaded?(Vapor.Studio.Nodes.Diffusion), do: Vapor.Studio.Nodes.Diffusion.comfy_outputs(t), else: []
  end

  # a link [src, index] → [src, the port name of that output]
  defp out([src, i], _expected) when is_integer(i) do
    names = Map.get(Process.get({__MODULE__, :ports}, %{}), to_string(src), [])
    [to_string(src), Enum.at(names, i, "output#{i}")]
  end

  defp out(other, _), do: other

  defp translate(_id, %{"class_type" => ct, "inputs" => ins}) do
    case {ct, ins} do
      {"LoadImage", %{"image" => file}} ->
        {:ok, %{"type" => "image.load", "params" => %{"path" => file}}, "LoadImage reads #{file} from the studio's directory (its MASK output is not translated)"}

      {t, %{"images" => src}} when t in ["SaveImage", "PreviewImage"] ->
        {:ok, %{"type" => "studio.output", "params" => %{"name" => Map.get(ins, "filename_prefix", t)}, "inputs" => %{"value" => out(src, "image")}}, nil}

      {"EmptyImage", %{"width" => w, "height" => h} = i} ->
        if Map.get(i, "batch_size", 1) != 1 do
          {:error, "batch_size > 1"}
        else
          c = Map.get(i, "color", 0)
          {:ok, %{"type" => "image.solid", "params" => %{"width" => w, "height" => h, "red" => (Bitwise.bsr(c, 16) &&& 255) / 255,
                                                          "green" => (Bitwise.bsr(c, 8) &&& 255) / 255, "blue" => (c &&& 255) / 255}}, nil}
        end

      {"ImageScale", %{"image" => src, "upscale_method" => m, "width" => w, "height" => h} = i} ->
        with {:ok, m} <- method(m) do
          {:ok, %{"type" => "image.resize", "params" => %{"width" => w, "height" => h, "method" => m, "crop" => Map.get(i, "crop", "disabled")},
                  "inputs" => %{"image" => out(src, "image")}}, nil}
        end

      {"ImageScaleBy", %{"image" => src, "upscale_method" => m, "scale_by" => s}} ->
        with {:ok, m} <- method(m), do: {:ok, %{"type" => "image.scale_by", "params" => %{"factor" => s, "method" => m}, "inputs" => %{"image" => out(src, "image")}}, nil}

      {"ImageInvert", %{"image" => src}} -> {:ok, %{"type" => "image.invert", "inputs" => %{"image" => out(src, "image")}}, nil}

      {"ImageCrop", %{"image" => src, "width" => w, "height" => h, "x" => x, "y" => y}} ->
        {:ok, %{"type" => "image.crop", "params" => %{"width" => w, "height" => h, "x" => x, "y" => y}, "inputs" => %{"image" => out(src, "image")}}, nil}

      {"ImageBlur", %{"image" => src, "blur_radius" => r, "sigma" => s}} ->
        {:ok, %{"type" => "image.blur", "params" => %{"radius" => r, "sigma" => s * r}, "inputs" => %{"image" => out(src, "image")}},
         "ImageBlur sigma #{s} (in radius units) = #{s * r} pixels"}

      {"ImageCompositeMasked", %{"destination" => d, "source" => s, "x" => x, "y" => y} = i} ->
        if Map.get(i, "resize_source", false) do
          {:error, "resize_source true"}
        else
          base = %{"destination" => out(d, "image"), "source" => out(s, "image")}
          ins = if m = i["mask"], do: Map.put(base, "mask", out(m, "mask")), else: base
          {:ok, %{"type" => "image.composite", "params" => %{"x" => x, "y" => y}, "inputs" => ins}, nil}
        end

      {t, ins} ->
        if Code.ensure_loaded?(Vapor.Studio.Nodes.Diffusion) and t in Vapor.Studio.Nodes.Diffusion.comfy_types(),
          do: Vapor.Studio.Nodes.Diffusion.from_comfy(t, ins),
          else: {:error, "#{t} has no translation"}
    end
  end

  defp translate(_id, _), do: {:error, "not a {class_type, inputs} object"}

  defp method(m), do: (case @methods do %{^m => v} -> {:ok, v}; _ -> {:error, "upscale method #{m}"} end)

  defp a &&& b, do: Bitwise.band(a, b)
end
