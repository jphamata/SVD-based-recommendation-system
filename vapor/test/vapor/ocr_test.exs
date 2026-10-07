defmodule Vapor.OCRTest do
  @moduledoc """
  The shipped OCR reader (`priv/ocr`) measured, not assumed: admitted by
  the airlock, reading lines rendered in fonts it never saw and a
  photographed page within stated error rates, refusing to invent text on
  a blank page — and, with PyTorch present, its per-frame logits equal to
  an independent PyTorch implementation of the same checkpoint
  (`test/python/ocr_reference.py`).
  """
  use ExUnit.Case, async: false
  alias Vapor.Docs.Pictures
  alias Vapor.Tensor
  alias Vapor.Vision.{OCR, Segment}

  @moduletag timeout: 600_000
  @held_out Path.expand("../../priv/quality/ocr", __DIR__)
  @page Path.expand("../fixtures/ocr", __DIR__)

  defp picture(path), do: elem(Pictures.read(Vapor.Docs.sniff(path, File.read!(path)), File.read!(path)), 1).image

  test "the shipped reader is admitted: a per-row encoder, the charset, programs bucketed up to its rows" do
    assert {:ok, m} = OCR.default()
    assert m.spec.interface == :encoder and :row_logits in m.spec.features
    assert tuple_size(m.labels) == m.spec.config.raw["num_labels"]
    assert Enum.map(m.programs, &elem(&1, 0)) == [64, 128, 256, 512]

    # a checkpoint of another contract is refused with what is missing
    assert {:error, %Vapor.Rejection{}} = OCR.load(Path.expand("../../priv/digits/classifier", __DIR__))
  end

  @tag :native
  test "held-out lines (fonts never seen in training): CER under 10 %; a blank page reads as nothing" do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(@held_out, "labels.json")))
    items = meta["lines"] |> Enum.sort() |> Enum.take(20)

    {errs, total} =
      Enum.reduce(items, {0, 0}, fn {file, %{"text" => text}}, {e, n} ->
        {:ok, r} = OCR.read(picture(Path.join(@held_out, file)))
        ref = String.trim(Regex.replace(~r/ +/, text, " "))
        {e + OCR.levenshtein(String.graphemes(r.text), String.graphemes(ref)), n + String.length(ref)}
      end)

    assert errs / total < 0.10

    blank = %{w: 200, h: 60, gray: :binary.copy(<<255>>, 200 * 60)}
    assert {:ok, %{text: "", lines: []}} = OCR.read(blank)
  end

  @tag :native
  test "a photographed printed page (scikit-image): the six complete lines under 15 % CER" do
    {:ok, gt} = Vapor.JSON.decode(File.read!(Path.join(@page, "page.json")))
    {:ok, r} = OCR.read(picture(Path.join(@page, "page.png")))
    hyp = r.lines |> Enum.take(length(gt["lines"])) |> Enum.map_join("\n", & &1.text)
    assert OCR.cer(hyp, Enum.join(gt["lines"], "\n")) < 0.15
  end

  @tag :torch
  test "per-frame logits = an independent PyTorch implementation of the checkpoint" do
    {:ok, m} = OCR.default()
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(@held_out, "labels.json")))
    {file, _} = meta["lines"] |> Enum.sort() |> hd()
    [line | _] = picture(Path.join(@held_out, file)) |> Segment.gray() |> Segment.ink() |> Segment.components() |> Segment.lines()
    rows = line |> Segment.line_bitmap() |> Segment.frames()
    [t, width] = rows.shape
    frames = rows |> Tensor.to_floats() |> Enum.chunk_every(width)

    {size, program} = Enum.find(m.programs, fn {s, _} -> s >= t end)
    out = Vapor.Modal.Runner.run(program, Vapor.Lock.Adapters.Encoder.input(m.spec, rows, size), worker: OCR.worker())
    got = out.row_logits |> Tensor.to_floats() |> Enum.take(t * tuple_size(m.labels))

    script = File.read!(Path.expand("../python/ocr_reference.py", __DIR__))
    {:ok, ref} = Vapor.JSON.decode(Vapor.TestHelpers.py!(script, [m.dir], Vapor.JSON.encode(%{frames: frames})))
    want = List.flatten(ref["logits"])

    scale = want |> Enum.map(&abs/1) |> Enum.max()
    assert (Enum.zip_with(got, want, &abs(&1 - &2)) |> Enum.max()) / scale < 1.0e-5
  end
end
