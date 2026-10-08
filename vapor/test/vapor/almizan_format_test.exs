defmodule Vapor.AlmizanFormatTest do
  @moduledoc """
  `vapor wzn fmt` and the language server's formatting (`Vapor.Almizan.Format`):
  the canonical form keeps every comment (the printer alone dropped them —
  "format document" used to delete them), never changes the program's
  hash, is idempotent, and is the same tree in either script.
  """
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO
  alias Vapor.Almizan
  alias Vapor.Almizan.Format

  @files Path.wildcard(Path.expand("../../priv/almizan/*.wzn", __DIR__))

  defp comments(text), do: for(l <- String.split(text, "\n"), [_, c] <- [String.split(l, ";", parts: 2)], do: String.trim(String.trim_leading(c, ";")))
  defp hash(text), do: (fn {:ok, m} -> Almizan.hash(m) end).(Almizan.parse(text))

  test "every shipped file: idempotent, the same hash, every comment kept" do
    assert length(@files) >= 5

    for f <- @files do
      text = File.read!(f)
      {:ok, canon} = Format.format(text)
      assert Format.format(canon) == {:ok, canon}, f
      assert hash(canon) == hash(text), f
      assert Enum.sort(comments(canon)) == Enum.sort(comments(text)), f
    end
  end

  test "a messy file: canonical layout, the inner comment moved above, fractions in lowest terms, the trailer kept" do
    messy = """
    ;; the kinetic part
    (claim kinetic (root H-s-b)   (wazn fail)
       (inputs (v q))   ; a remark inside
     (body (* 2/4 v v)))
    ; and a note at the end
    """

    {:ok, canon} = Format.format(messy)
    assert String.starts_with?(canon, ";; the kinetic part\n; a remark inside\n(claim kinetic")
    assert canon =~ "(* 1/2 v v)"
    assert String.ends_with?(canon, "\n\n; and a note at the end\n")
    assert hash(canon) == hash(messy)
    refute Format.canonical?(messy)
    assert Format.canonical?(canon)
  end

  test "the two scripts are one tree: Latin → Arabic → Latin is the Latin form, comments and hash kept" do
    text = File.read!(Path.expand("../../priv/almizan/oscillator.wzn", __DIR__))
    {:ok, latin} = Format.format(text, :latin)
    {:ok, arabic} = Format.format(latin, :arabic)
    assert arabic =~ ~r/\p{Arabic}/u
    assert Format.format(arabic, :latin) == {:ok, latin}
    assert hash(arabic) == hash(latin)
    assert Enum.sort(comments(arabic)) == Enum.sort(comments(text))
  end

  test "vapor wzn fmt: --check, --write, refusal of what does not parse" do
    dir = Path.join(System.tmp_dir!(), "vapor-fmt-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    p = Path.join(dir, "k.wzn")
    File.write!(p, "(claim k (root H-s-b) (wazn fail) (inputs (v q)) (body (* 2/4 v v)))  ; x\n")

    try do
      capture_io(:stderr, fn -> assert Vapor.Main.run(["wzn", "fmt", p, "--check"]) == 1 end)
      assert Vapor.Main.run(["wzn", "fmt", p, "--write"]) == 0
      assert capture_io(fn -> assert Vapor.Main.run(["wzn", "fmt", p, "--check"]) == 0 end) == ""
      assert File.read!(p) =~ "; x\n(claim k"
      File.write!(p, "(claim k (root")
      capture_io(:stderr, fn -> assert Vapor.Main.run(["wzn", "fmt", p]) == 3 end)
    after
      File.rm_rf!(dir)
    end
  end

  test "the language server's formatting and lens keep the comments" do
    st = %{docs: %{}, shutdown: false}
    uri = "file:///tmp/k.wzn"
    text = "; keep me\n(claim k (root H-s-b) (wazn fail) (inputs (v q)) (body (* 2/4 v v)))\n"
    {_, st} = Vapor.LSP.handle(%{"method" => "textDocument/didOpen", "params" => %{"textDocument" => %{"uri" => uri, "languageId" => "almizan", "version" => 1, "text" => text}}}, st)
    {[reply], st} = Vapor.LSP.handle(%{"method" => "textDocument/formatting", "id" => 7, "params" => %{"textDocument" => %{"uri" => uri}}}, st)
    assert [%{"newText" => new}] = reply["result"]
    assert new =~ "; keep me\n(claim k" and new =~ "1/2"
    {[_, apply], _} = Vapor.LSP.handle(%{"method" => "workspace/executeCommand", "id" => 8, "params" => %{"command" => "vapor.almizan.toArabic", "arguments" => [uri]}}, st)
    [edit] = apply["params"]["edit"]["changes"][uri]
    assert edit["newText"] =~ "; keep me" and edit["newText"] =~ ~r/\p{Arabic}/u
  end
end
