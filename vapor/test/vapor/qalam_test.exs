defmodule Vapor.QalamTest do
  @moduledoc """
  Al-Qalam (`Vapor.Qalam`): the editor's pure core driven by keys, the
  balance in its gutter, the Merkle undo tree, and one end-to-end session
  through a real pseudo-terminal.
  """
  use ExUnit.Case, async: true
  alias Vapor.Qalam

  @root Path.expand("../..", __DIR__)
  @osc Path.join(@root, "priv/almizan/oscillator.wzn")

  defp ed(text), do: Qalam.new(text)
  defp lines(st), do: st.lines

  test "bytes from a terminal become keys: arrows, escape, enter, backspace, control" do
    assert Qalam.keys("a\e[A\eOB\e[3~\e\r\x7f\x01é") == ["a", :up, :down, :delete, :esc, :enter, :backspace, {:ctrl, ?a}, "é"]
  end

  test "editing as vi does: motions, counts, insert, delete, yank and paste, join" do
    st = ed("one\ntwo\nthree\n")
    assert lines(Qalam.feed(st, "jddp")) == ["one", "three", "two"]
    assert lines(Qalam.feed(st, "2ddP")) == ["one", "two", "three"]
    assert lines(Qalam.feed(st, "Ahi\e")) == ["onehi", "two", "three"]
    assert lines(Qalam.feed(st, "o4\e")) == ["one", "4", "two", "three"]
    assert lines(Qalam.feed(st, "J")) == ["one two", "three"]
    assert lines(Qalam.feed(st, "lx$x")) == ["o", "two", "three"]
    assert lines(Qalam.feed(st, "yyGp")) == ["one", "two", "three", "one"]
    # w on the last word stays; b at a line's start goes to the previous line's last word
    s = Qalam.feed(st, "2jwbih\e")
    assert {s.row, lines(s)} == {1, ["one", "htwo", "three"]}
    assert Qalam.feed(ed("(foo bar-baz) qux"), "ww").col == 5
    assert Qalam.feed(ed("(foo bar-baz) qux"), "$b").col == 14
    # insert mode: enter keeps the indentation, backspace joins lines
    s = Qalam.feed(ed("  (a b)"), "A\rc\e")
    assert lines(s) == ["  (a b)", "  c"]
    assert lines(Qalam.feed(s, "I\x7f\x7f\x7f\e")) == ["  (a b)c"]
  end

  test "the balance in the gutter: a damping coefficient walked to zero turns ✗ into ✓" do
    {:ok, st} = Qalam.open(@osc)
    st = Qalam.refresh(st)
    damped = Enum.find_index(st.lines, &String.contains?(&1, "claim damped-energy"))
    energy = Enum.find_index(st.lines, &String.contains?(&1, "claim energy"))
    assert {"✓", _} = st.marks[energy]
    assert {"✗", why} = st.marks[damped]
    assert why =~ "refuted" and why =~ "at "
    assert st.summary == "1 proved · 1 refuted"

    s = Qalam.feed(st, ["/", "1", "/", "1", "0", :enter, {:ctrl, ?x}])
    assert Enum.at(s.lines, s.row) =~ "(* 0/10 v)"
    assert {"✓", _} = s.marks[damped]
    assert s.summary == "2 proved"
    # K: the verdict's detail on the claim's line
    assert Qalam.feed(s, "#{damped + 1}GK").msg =~ "damped-energy: proved"
    # one more step is a damping that pumps energy in: refuted again
    assert {"✗", _} = Qalam.feed(s, [{:ctrl, ?x}]).marks[damped]
  end

  test "scrubbing: integers by one, fractions by their unit, Arabic-Indic digits kept, identifiers left alone" do
    assert lines(Qalam.feed(ed("(* x2 7)"), [{:ctrl, ?a}])) == ["(* x2 8)"]
    assert lines(Qalam.feed(ed("(* 1/3 v)"), ["5", {:ctrl, ?a}])) == ["(* 6/3 v)"]
    assert lines(Qalam.feed(ed("(* ١/٢ ع ع)"), [{:ctrl, ?x}])) == ["(* ٠/٢ ع ع)"]
    assert lines(Qalam.feed(ed("(- x 1)"), [{:ctrl, ?x}, {:ctrl, ?x}])) == ["(- x -1)"]
    assert Qalam.feed(ed("(a b)"), [{:ctrl, ?a}]).msg =~ "no number"
  end

  test "the Merkle undo tree: undo, a new branch, the old one kept, every state reachable in order" do
    st = ed("abc\n")
    a = Qalam.feed(st, "x")
    assert lines(a) == ["bc"]
    b = Qalam.feed(a, "uiz\e")
    assert lines(b) == ["zabc"]
    assert Qalam.tree(b) == %{states: 3, branches: 2, at: b.at}
    # g- walks back in time, onto the abandoned branch
    assert lines(Qalam.feed(b, "g-")) == ["bc"]
    assert lines(Qalam.feed(b, "g-g-")) == ["abc"]
    # a state's identity is its history, not only its text: deleting then retyping is a new state
    c = Qalam.feed(st, "xia\e")
    assert lines(c) == lines(st) and c.at != st.at
    # and the same history gives the same identity, in any editor
    assert Qalam.feed(ed("abc\n"), "uiz\e").at == Qalam.feed(st, "uiz\e").at
    assert lines(Qalam.feed(a, "u\x12")) == ["bc"]
  end

  test ":fmt keeps comments; :ar and :la change the script, not the tree" do
    {:ok, st} = Qalam.open(@osc)
    messy = st |> Qalam.feed("2Gi   \e")
    f = Qalam.feed(messy, ":fmt\r")
    assert Qalam.text(f) == File.read!(@osc)
    ar = Qalam.feed(f, ":ar\r")
    assert Enum.any?(ar.lines, &String.contains?(&1, "دعوى")) and Enum.any?(ar.lines, &String.starts_with?(&1, ";"))
    {:ok, m1} = Vapor.Almizan.parse(Qalam.text(f))
    {:ok, m2} = Vapor.Almizan.parse(Qalam.text(ar))
    assert Vapor.Almizan.hash(m1) == Vapor.Almizan.hash(m2)
    assert Qalam.text(Qalam.feed(ar, ":la\r")) == File.read!(@osc)
    assert Qalam.feed(ed("plain"), ":fmt\r").msg =~ "Almizan"
  end

  test "structure: % finds the matching bracket across lines, [[ and ]] the top-level forms" do
    {:ok, st} = Qalam.open(@osc)
    s = Qalam.feed(st, "]]")
    assert Enum.at(s.lines, s.row) =~ "(claim kinetic"
    m = Qalam.feed(s, "%")
    assert {m.row, String.at(Enum.at(m.lines, m.row), m.col)} == {s.row + 2, ")"}
    assert Qalam.feed(m, "%") |> then(&{&1.row, &1.col}) == {s.row, 0}
    assert Enum.at(Qalam.feed(s, "]]").lines, Qalam.feed(s, "]]").row) =~ "(claim energy"
    # a bracket inside a comment is not code
    c = Qalam.feed(%{ed("(a ; (\n b)") | kind: :almizan}, "%")
    assert {c.row, c.col} == {1, 2}
  end

  @tag :tmp_dir
  test "writing: :q refuses to lose changes, :wq writes atomically with the final newline, a new file is made", %{tmp_dir: d} do
    p = Path.join(d, "a.txt")
    File.write!(p, "x\ny\n")
    {:ok, st} = Qalam.open(p)
    s = Qalam.feed(st, "dd:q\r")
    assert not s.quit and s.msg =~ "unsaved"
    s = Qalam.feed(s, ":wq\r")
    assert s.quit and File.read!(p) == "y\n"
    assert Qalam.feed(st, ":q\r").quit
    {:ok, n} = Qalam.open(Path.join(d, "new.wzn"))
    assert n.msg == "new file"
    Qalam.feed(n, "i(claim k (root H-s-b) (wazn fail) (inputs (v q)) (body v))\e:x\r")
    assert File.read!(Path.join(d, "new.wzn")) =~ "(claim k"
    assert Path.wildcard(Path.join(d, "*.qalam*")) == []
  end

  test "the screen: the cursor stays visible, the gutter beside its claim, the status line says where and what" do
    {:ok, st} = Qalam.open(@osc)
    st = Qalam.refresh(st)
    {st2, rows, {r, c}} = Qalam.render(Qalam.feed(st, "G$"), {8, 30})
    assert length(rows) == 8 and r in 0..6 and c in 2..29
    assert st2.top > 0
    {_, rows, _} = Qalam.render(st, {40, 100})
    assert Enum.any?(rows, &String.starts_with?(&1, "✓ (claim energy"))
    assert Enum.any?(rows, &String.starts_with?(&1, "✗ (claim damped-energy"))
    assert List.last(rows) =~ ~r/^NORMAL  oscillator\.wzn  1:1  1 proved · 1 refuted/
    {_, rows, _} = Qalam.render(Qalam.feed(st, ":wq"), {10, 40})
    assert List.last(rows) == ":wq"
  end

  @tag :python
  @tag :tmp_dir
  test "a real terminal: bin/vapor qalam in a pseudo-terminal, keys in, the file written", %{tmp_dir: d} do
    p = Path.join(d, "t.txt")
    File.write!(p, "keep\ndrop\n")
    driver = Path.join(d, "drive.py")

    File.write!(driver, ~S"""
    import os, pty, select, sys, time
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.argv[1], [sys.argv[1], "qalam", sys.argv[2]])
    def drain(t):
        out, end = b"", time.time() + t
        while time.time() < end:
            if select.select([fd], [], [], 0.05)[0]:
                try: out += os.read(fd, 65536)
                except OSError: break
        return out
    screen = b""
    deadline = time.time() + 90
    while b"NORMAL" not in screen and time.time() < deadline:
        screen += drain(0.5)
    for k in [b"j", b"d", b"d", b"Z", b"Z"]:
        os.write(fd, k); drain(0.2)
    drain(2)
    _, status = os.waitpid(pid, 0)
    sys.stdout.write("NORMAL seen\n" if b"NORMAL" in screen else "no screen\n")
    sys.stdout.write("exit %d\n" % os.WEXITSTATUS(status))
    """)

    {out, 0} = System.cmd(Vapor.TestHelpers.python(), [driver, Path.join(@root, "bin/vapor"), p], env: [{"MIX_ENV", "test"}], stderr_to_stdout: true)
    assert out =~ "NORMAL seen" and out =~ "exit 0"
    assert File.read!(p) == "keep\n"
  end
end
