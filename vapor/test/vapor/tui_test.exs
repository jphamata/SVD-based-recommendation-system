defmodule Vapor.TUITest do
  @moduledoc "The terminal console's interpreter, without a terminal: plain text when there is no colour, both languages, refusals as messages."
  use ExUnit.Case, async: true
  alias Vapor.TUI

  setup do
    {:ok, st: TUI.new(color: false)}
  end

  test "help, languages, unknown commands and quitting", %{st: st} do
    {help, _} = TUI.eval("help", st)
    assert help =~ "read FILE" and help =~ "every vapor verb"
    {_, pt} = TUI.eval("lang pt", st)
    assert {msg, _} = TUI.eval("frobnicate", pt)
    assert msg =~ "comando desconhecido"
    assert {:quit, "bye"} = TUI.eval("quit", st)
    assert {[], ^st} = TUI.eval("", st)
  end

  test "every other line is the shared interpreter: vapor verbs, pipes, sequences — as in the console's terminal", %{st: st} do
    dir = Path.join(System.tmp_dir!(), "tui-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    f = Path.join(dir, "nums.txt")
    {_, st} = TUI.eval("echo 1e16 1 -1e16 > #{f}", st)
    {out, st} = TUI.eval("amalgam #{f} | cat", st)
    assert {:ok, %{"amalgam" => %{"value" => "1.0"}}} = Vapor.JSON.decode(out)
    {out, _} = TUI.eval("wzn abjad ميزان", st)
    assert out =~ "108"
    File.rm_rf!(dir)
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
  end

  @tag :native
  test "read: a printed line with its confidence; a picture without text says so instead of inventing", %{st: st} do
    {out, _} = TUI.eval("read " <> Path.expand("../../priv/quality/ocr/00002.png", __DIR__), st)
    assert IO.iodata_to_binary(out) =~ "monotonicity"
    {out, _} = TUI.eval("read " <> Path.expand("../fixtures/docs/gray.png", __DIR__), %{st | lang: :pt})
    assert IO.iodata_to_binary(out) =~ "gray.png · nenhum texto encontrado"
  end
end
