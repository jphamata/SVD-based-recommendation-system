defmodule Vapor.MathTest do
  @moduledoc """
  `Vapor.Vision.Math` on formulas set in Computer Modern and STIX — type
  families never used for its templates (`priv/quality/math`, rendered by
  `test/python/math_render.py`): the LaTeX read token for token, against
  the control of the same symbols read flat, left to right, without the
  geometry of fractions, radicals and scripts.
  """
  use ExUnit.Case, async: true
  alias Vapor.Vision.{Math, OCR}

  @dir Path.expand("../../priv/quality/math/test", __DIR__)

  defp cases do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(@dir, "truth.json")))

    for {f, t} <- Enum.sort(meta["items"]) do
      {:ok, pic} = Vapor.Docs.Pictures.read(:png, File.read!(Path.join(@dir, f)))
      {:ok, r} = Math.read(pic.image)
      {t["latex"], r}
    end
  end

  defp ter(pairs), do: Enum.sum(for({t, p} <- pairs, do: OCR.levenshtein(Math.tokens(p), Math.tokens(t)))) / Enum.sum(for({t, _} <- pairs, do: length(Math.tokens(t))))

  test "unseen type families: few token errors, most formulas exact; flat reading (the control) is far worse" do
    cs = cases()
    read = for {t, r} <- cs, do: {t, r.latex}
    flat = for {t, r} <- cs, do: {t, r.symbols |> Enum.sort_by(fn %{box: {x0, _, _, _}} -> x0 end) |> Enum.map_join(& &1.latex)}
    exact = Enum.count(read, fn {t, p} -> t == p end)
    assert ter(read) < 0.08, "token error #{ter(read)}"
    assert exact >= 0.45 * length(cs), "#{exact}/#{length(cs)} exact"
    assert ter(flat) > 3 * ter(read), "flat #{ter(flat)} vs #{ter(read)}"
  end

  test "structure: fractions of fractions, radicals over fractions, scripts on both sides, limits" do
    {:ok, meta} = Vapor.JSON.decode(File.read!(Path.join(@dir, "truth.json")))
    cs = cases()
    kinds = for {t, r} <- cs, t == r.latex, k <- ["\\frac", "\\sqrt", "_{", "^{", "\\sum", "\\int"], String.contains?(t, k), uniq: true, do: k
    assert Enum.sort(kinds) == Enum.sort(["\\frac", "\\sqrt", "_{", "^{", "\\sum", "\\int"]), inspect(kinds)
    assert map_size(meta["items"]) == length(cs)
  end

  test "an empty picture is no formula" do
    blank = Vapor.Modal.Image.new(40, 20, 1, List.duplicate(1.0, 800))
    assert Math.read(blank) == {:error, :empty}
  end
end
