defmodule Vapor.VisionTest do
  @moduledoc "The geometry half of OCR (no model), CTC decoding, error rates, and the perceptron topology."
  use ExUnit.Case, async: true
  alias Vapor.{Lock, Tensor}
  alias Vapor.Vision.{OCR, Segment}
  alias Vapor.Runtime.Oracle

  # a page drawn by hand: two lines of "glyphs" (ink blocks), a word gap, an
  # i-like glyph (dot above a stem) and an illumination gradient across it
  defp page do
    {w, h} = {120, 60}
    blocks =
      [{10, 12, 15, 22}, {18, 12, 23, 22}, {26, 12, 31, 22}, {44, 12, 49, 22}, {52, 15, 54, 22}, {52, 11, 54, 13}, # line 1: "abc d i"
       {10, 38, 15, 48}, {18, 38, 23, 48}, {36, 38, 41, 48}]                                                   # line 2: "ab c"

    ink = MapSet.new(for {x0, y0, x1, y1} <- blocks, x <- x0..x1, y <- y0..y1, do: {x, y})

    gray =
      for y <- 0..(h - 1), x <- 0..(w - 1), into: <<>> do
        paper = 230 - div(x, 2)
        <<if(MapSet.member?(ink, {x, y}), do: max(paper - 150, 0), else: paper)>>
      end

    %{w: w, h: h, gray: gray}
  end

  test "Sauvola ink, components, lines, merged columns and spaces" do
    g = page()
    lines = g |> Segment.ink() |> Segment.components() |> Segment.lines()
    assert length(lines) == 2
    [l1, l2] = lines
    # the dot over the stem joins it: 5 glyphs on line 1, a space after the third
    assert length(l1.glyphs) == 5 and l1.spaces == [2]
    assert Enum.at(l1.glyphs, 4).parts == 2
    assert length(l2.glyphs) == 3 and l2.spaces == [1]
    assert length(hd(Segment.features(l1))) == Segment.width()
  end

  test "line bitmaps: baseline on row 22, median glyph 12 px tall; frames are 32×8 windows every 2 columns" do
    [l1, _] = page() |> Segment.ink() |> Segment.components() |> Segment.lines()
    bm = Segment.line_bitmap(l1)
    assert bm.h == 32 and byte_size(bm.data) == 32 * bm.w
    row = fn y -> for x <- 0..(bm.w - 1), do: :binary.at(bm.data, y * bm.w + x) end
    assert Enum.sum(row.(21)) > 0 and Enum.sum(row.(23)) == 0
    f = Segment.frames(bm)
    assert f.shape == [div(bm.w - 8, 2) + 1, 256]
  end

  test "CTC greedy: argmax per frame, repeats collapsed, blanks dropped" do
    labels = {"", "a", "b"}
    hot = fn k -> for i <- 0..2, do: if(i == k, do: 5.0, else: 0.0) end
    rows = Enum.map([1, 1, 0, 1, 2, 2, 0, 0, 2], hot)
    assert OCR.ctc_greedy(rows, 9, labels) |> Enum.map_join(& &1.char) == "aabb"
    assert [%{frames: [0, 1]} | _] = OCR.ctc_greedy(rows, 9, labels)
  end

  test "character and word error rates" do
    assert OCR.cer("kitten", "sitting") == 3 / 7
    assert OCR.cer("same", "same") == 0.0
    assert OCR.wer("o gato subiu", "o gato desceu") == 1 / 3
  end

  test "the perceptron topology: admitted, the program = the formula, wrong shapes refused" do
    cfg = %{"model_type" => "vapor_mlp", "in_width" => 16, "hidden_sizes" => [32], "out_width" => 3, "hidden_act" => "relu"}
    ws = %{"layers.0.weight" => Tensor.random(:f32, [32, 16], 1), "layers.0.bias" => Tensor.random(:f32, [32], 2),
           "layers.1.weight" => Tensor.random(:f32, [3, 32], 3), "layers.1.bias" => Tensor.random(:f32, [3], 4)}
    assert {:ok, spec, ws} = Lock.from_map(cfg, ws)
    assert spec.interface == :map and spec.family == "vapor_mlp"
    {:ok, p} = Lock.build(spec, ws, rows: 2)
    x = Tensor.random(:f32, [2, 16], 5)
    got = Oracle.eval_program(p, %{rows: x}).out |> Tensor.to_floats() |> Enum.chunk_every(3)

    mat = fn t -> t |> Tensor.to_floats() |> Enum.chunk_every(List.last(t.shape)) end
    aff = fn rows, w, b -> Enum.map(rows, fn r -> Enum.zip_with(mat.(w), Tensor.to_floats(b), fn wr, bv -> Enum.zip_reduce(r, wr, bv, fn a, c, s -> s + a * c end) end) end) end
    h = aff.(mat.(x), ws["layers.0.weight"], ws["layers.0.bias"]) |> Enum.map(fn r -> Enum.map(r, &max(&1, 0.0)) end)
    want = aff.(h, ws["layers.1.weight"], ws["layers.1.bias"])
    for {a, b} <- Enum.zip(List.flatten(got), List.flatten(want)), do: assert_in_delta(a, b, 1.0e-4)

    assert {:error, %Vapor.Rejection{node: {:weight, "layers.1.weight"}}} = Lock.from_map(cfg, Map.put(ws, "layers.1.weight", Tensor.random(:f32, [4, 32], 3)))
    assert {:error, %Vapor.Rejection{node: {:config, "hidden_sizes"}}} = Lock.from_map(Map.put(cfg, "hidden_sizes", [30]), ws)
  end

  test "linear algebra: Cholesky solves, non-positive pivots refused" do
    a = Vapor.Linalg.from_rows([[4, 2, 0.6], [2, 5, 1], [0.6, 1, 3]])
    {:ok, l} = Vapor.Linalg.cholesky(a)
    x = Vapor.Linalg.chol_solve(l, [1.0, 2.0, 3.0]) |> Tuple.to_list()
    back = for r <- Vapor.Linalg.to_rows(a), do: Enum.zip_reduce(r, x, 0.0, fn p, q, s -> s + p * q end)
    for {b, want} <- Enum.zip(back, [1.0, 2.0, 3.0]), do: assert_in_delta(b, want, 1.0e-12)
    assert {:error, :not_positive_definite, 1} = Vapor.Linalg.cholesky(Vapor.Linalg.from_rows([[1, 2], [2, 1]]))
  end
end
