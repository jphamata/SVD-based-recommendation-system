defmodule Vapor.Console.Lab11 do
  @moduledoc """
  The console's laboratories for the 0.11 round, each a function from a
  small request to a JSON-ready map (docs/CONSOLE.md): sketch → drawing /
  plan, and saving results as verifiable archives (`Vapor.Archive`).
  """
  alias Vapor.{Archive, Sketch}

  @samples %{"shapes" => "sketch/shapes.png", "plan" => "sketch/plan.png"}

  # a sample picture by name (the quality fixtures), or the uploaded one
  defp picture("sample:" <> name, _data) do
    case Map.fetch(@samples, name) do
      {:ok, rel} ->
        bytes = File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "quality", rel]))
        {:ok, %{image: img}} = Vapor.Docs.Pictures.read(:png, bytes)
        {:ok, img}

      :error ->
        {:error, "sample: one of #{Enum.join(Map.keys(@samples), ", ")}"}
    end
  end

  defp picture(name, data) do
    with {:ok, bytes} <- Base.decode64(data, ignore: :whitespace) |> ok("data: base64"),
         {:ok, %{image: img}} <- Vapor.Docs.Pictures.read(Vapor.Docs.sniff(name, bytes), bytes) |> ok("a PNG, JPEG or PPM picture") do
      {:ok, img}
    else
      {:error, why} -> {:error, why}
      # read, but not decoded (too many pixels, or a format only described): said, not crashed on
      {:ok, _meta} -> {:error, "a PNG, JPEG or PPM picture of at most 4 megapixels"}
    end
  end

  defp ok({:ok, v}, _), do: {:ok, v}
  defp ok(:error, why), do: {:error, why}
  defp ok({:error, _}, why), do: {:error, why}

  # ----------------------------------------------------------------- sketch

  def sketch(name, data, mode, opts) do
    with {:ok, img} <- picture(name, data) do
      # the sketch itself, as the vectoriser saw it (for the side-by-side)
      seen = "data:image/png;base64," <> Base.encode64(Vapor.Modal.Image.png(Vapor.Raster.fit(img, 640)))
      with {:ok, r} <- sketch_of(img, mode, opts), do: {:ok, Map.put(r, :image, seen)}
    end
  end

  defp sketch_of(img, "plan", opts), do: do_plan(img, opts)

  defp sketch_of(img, _mode, opts) do
    v = Sketch.vectorize(img, snap: opts["snap"] != false)
    {:ok, Map.merge(v, %{mode: "vector", svg: Sketch.svg(v), dxf: Sketch.dxf(v, opts["scale"] || 1.0)})}
  end

  defp do_plan(img, opts) do
    p = Sketch.plan(img, plan_opts(opts))
    mesh = %{vertices: p.mesh.vertices, faces: p.mesh.faces, colors: p.mesh.colors}

    {:ok, %{mode: "plan", w: p.w, h: p.h, walls: p.walls, doors: Enum.map(p.doors, &Map.put(&1, :metres, &1.width * p.scale)),
            rooms: Enum.map(p.rooms, fn r -> %{polygon: Enum.map(r.polygon, &Tuple.to_list/1), area: r.area} end), scale: p.scale, glb: p.glb, mesh: mesh}}
  end


  defp plan_opts(o) do
    [longest: o["longest"] || 8.0, height: o["height"] || 2.7] ++ if(o["scale"], do: [scale: o["scale"]], else: [])
  end

  def archive(kind, recipe, result) when is_binary(kind) and byte_size(kind) <= 64 do
    # a replayable kind is packed with the result its producer computes now, so replay compares like with like
    result =
      if kind in Archive.replayable() and is_map(recipe) do
        case Archive.produce(kind, recipe) do
          {:ok, r} -> r
          _ -> result
        end
      else
        result
      end

    a = Archive.pack(kind, recipe || %{}, result || %{})
    # signed with the operator's key when the server has one (VAPOR_ARCHIVE_KEY: a `mix vapor.audit keygen` file)
    {zip, signed} =
      case System.get_env("VAPOR_ARCHIVE_KEY") do
        nil -> {a.zip, nil}
        path -> (k = Archive.load_key(path); {:ok, z} = Archive.sign(a.zip, k); {z, Vapor.Certificate.key_id(k.public)})
      end
    {:ok, %{id: a.id, name: "vapor-#{kind |> String.replace(~r/[^a-z0-9._-]/, "-")}-#{String.slice(a.id, 0, 12)}.zip", data: Base.encode64(zip), replayable: kind in Archive.replayable(), signed_by: signed}}
  end

  def archive(_, _, _), do: {:error, "kind: a short string"}

  def archive_check(data) do
    with {:ok, zip} <- Base.decode64(data, ignore: :whitespace) |> ok("data: base64") do
      case Archive.verify(zip) do
        {:ok, b} ->
          replay = if b.manifest["kind"] in Archive.replayable(), do: inspect(Archive.replay(zip)), else: "not replayable (#{b.manifest["kind"]})"
          {:ok, %{id: b.id, kind: b.manifest["kind"], vapor: b.manifest["vapor"], files: Map.keys(b.manifest["files"]), recipe: b.manifest["recipe"], intact: true, replay: replay,
                  signature: b.signature}}

        {:error, why} ->
          {:ok, %{intact: false, why: inspect(why)}}
      end
    end
  end
end
