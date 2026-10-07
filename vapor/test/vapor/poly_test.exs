defmodule Vapor.PolyTest do
  @moduledoc """
  Certified polynomial approximations (`Vapor.Poly`) — the form every
  non-linearity must take under CKKS: the proved bound holds at thousands of
  random points (against the binary64 function), and it
  is tight (within a small factor of the error actually observed).
  """
  use ExUnit.Case, async: true
  alias Vapor.Poly

  # a very accurate approximation needs a finer grid for the remainder term to stay below its error
  for {f, iv, d, grid} <- [{:sigmoid, {-8, 8}, 15, 4096}, {:tanh, {-4, 4}, 15, 4096}, {:exp, {-1, 1}, 8, 32_768}, {:sigmoid, {-8, 8}, 7, 4096}] do
    test "#{f} on #{inspect(iv)}, degree #{d}" do
      %{bound: bound, depth: depth} = p = Poly.approx(unquote(f), unquote(Macro.escape(iv)), unquote(d), grid: unquote(grid))
      {a, b} = unquote(Macro.escape(iv))
      :rand.seed(:exsss, {1, 2, 3})
      xs = [a * 1.0, b * 1.0, 0.0] ++ for(_ <- 1..4000, do: a + (b - a) * :rand.uniform())
      observed = xs |> Enum.map(&abs(Poly.float_f(unquote(f), &1) - Poly.eval(p, &1))) |> Enum.max()
      # binary64 evaluation adds ~1e-15 on top of the proved bound
      assert observed <= bound + 1.0e-14, "observed #{observed} > bound #{bound}"
      assert bound <= 4 * observed + 1.0e-12, "bound #{bound} is loose (observed #{observed})"
      assert depth == ceil(:math.log2(unquote(d) + 1))
      IO.puts("\n  #{unquote(f)} #{inspect({a, b})} deg #{unquote(d)}: certified |error| ≤ #{Float.round(bound, 12)} (observed #{Float.round(observed, 12)}), depth #{depth}")
    end
  end
end
