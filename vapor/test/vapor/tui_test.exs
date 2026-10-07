defmodule Vapor.TUITest do
  @moduledoc "The terminal console's interpreter, without a terminal: plain text when there is no colour, both languages, refusals as messages."
  use ExUnit.Case, async: true
  alias Vapor.TUI

  setup do
    {:ok, st: TUI.new(color: false)}
  end

  test "help, languages, unknown commands and quitting", %{st: st} do
    {help, _} = TUI.eval("help", st)
    assert help =~ "read FILE" and help =~ "draw DIGIT"
    {_, pt} = TUI.eval("lang pt", st)
    assert {msg, _} = TUI.eval("frobnicate", pt)
    assert msg =~ "comando desconhecido"
    assert {:quit, "bye"} = TUI.eval("quit", st)
    assert {[], ^st} = TUI.eval("", st)
  end

  test "draw: the digit in the terminal (no ANSI without colour), read back, with its distance to training", %{st: st} do
    {out, _} = TUI.eval("draw 3 1", st)
    text = IO.iodata_to_binary(out)
    refute text =~ "\e["
    assert text =~ "classifier reads 3"
    assert text |> String.split("\n") |> Enum.take(8) |> Enum.all?(&(String.length(&1) == 16))
  end

  test "the noise gate: real prose is signal, repetition is noise", %{st: st} do
    {a, _} = TUI.eval("quality O motor compila o programa, confere cada passo e executa no substrato nativo sem perder um bit da saída.", st)
    assert a =~ "signal"
    {b, _} = TUI.eval("quality " <> String.duplicate("bla ", 40), st)
    assert b == "noise"
  end

  test "refusals are messages, not crashes", %{st: st} do
    {out, _} = TUI.eval("read /nonexistent.png", st)
    assert IO.iodata_to_binary(out) =~ "error"
    {out, _} = TUI.eval("search anything", st)
    assert out =~ "library is empty"
    {out, _} = TUI.eval("draw 12", st)
    assert out =~ "0 to 9"
  end

  @tag :native
  test "read: a printed line with its confidence; a picture without text says so instead of inventing", %{st: st} do
    {out, _} = TUI.eval("read " <> Path.expand("../../priv/quality/ocr/00002.png", __DIR__), st)
    assert IO.iodata_to_binary(out) =~ "monotonicity"
    {out, _} = TUI.eval("read " <> Path.expand("../fixtures/docs/gray.png", __DIR__), %{st | lang: :pt})
    assert IO.iodata_to_binary(out) =~ "gray.png · nenhum texto encontrado"
  end
end
