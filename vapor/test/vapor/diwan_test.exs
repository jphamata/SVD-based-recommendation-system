defmodule Vapor.DiwanTest do
  use ExUnit.Case, async: true
  alias Vapor.Diwan, as: D

  defp run(st, line), do: D.eval(line, st)
  defp out!(st, line), do: (fn {r, st} -> {r.out, st} end).(run(st, line))

  test "lexing: quotes, escapes, pipes, redirection and sequences" do
    assert {:ok, [%{argv: ["echo", "a b", "c|d", "e;f"]}], nil} = D.parse(~s(echo "a b" 'c|d' e\\;f))
    assert {:ok, [%{argv: ["cat"], stdin: "in.txt"}, %{argv: ["head", "2"]}], {:write, "out.txt"}} = D.parse("cat < in.txt | head 2 > out.txt")
    assert {:ok, [{_, nil}, {_, {:append, "log"}}]} = D.parse_line("echo one ; echo two >> log")
    assert {:error, _} = D.parse(~s(echo "unclosed))
    assert {:error, _} = D.parse("echo a >")
    assert {:error, _} = D.parse("a | < f b")
    assert {:error, _} = D.parse("a > f | b")
    assert {:error, _} = D.parse("| b")
  end

  test "a jailed session: files are its own; vapor verbs read them; the server's files do not exist" do
    st = D.new()
    {r, st} = run(st, ~s(echo "2^100 + 1" > big.alb))
    assert r.code == 0 and r.out == ""
    {out, st} = out!(st, "ls")
    assert out =~ "big.alb"
    {r, st} = run(st, ~s(alembic -e "2^100"))
    assert r.out =~ "1267650600228229401496703205376" and r.code == 0
    {r, _} = run(st, "alembic -e x /etc/passwd")
    assert r.code == 3 and r.err =~ "no such file in this session"
    {r, _} = run(st, "cat /etc/passwd")
    assert r.code == 1
    {r, _} = run(st, "echo x > ../escape")
    assert r.err =~ "not a file name"
  end

  test "pipes: every stage but the last writes JSON; the last writes for people" do
    st = D.new()
    {_, st} = run(st, "echo 1e16 1 -1e16 > nums")
    {piped, st} = out!(st, "amalgam nums | cat")
    assert {:ok, %{"amalgam" => %{"value" => "1.0"}}} = Vapor.JSON.decode(piped)
    {human, _} = out!(st, "amalgam nums")
    refute match?({:ok, %{}}, Vapor.JSON.decode(human))
    {r, _} = run(D.new(tty: false), "amalgam nums")
    assert r.code == 3
  end

  test "builtins: cat, head, wc, cp, mv, rm, history; sequences run in order" do
    st = D.new()
    {_, st} = run(st, "echo a > f ; echo b >> f ; echo c >> f")
    {out, st} = out!(st, "cat f | head 2")
    assert out == "a\nb\n"
    {out, st} = out!(st, "cat f | wc")
    assert out =~ ~r/^3 3 /
    {_, st} = run(st, "cp f g ; mv g h ; rm f")
    {out, st} = out!(st, "ls")
    assert out =~ "h" and not (out =~ ~r/\sf\n/)
    {r, st} = run(st, "rm nope")
    assert r.code == 1
    {out, _} = out!(st, "history")
    assert out =~ "rm nope"
  end

  test "--measure (a program on the server) is refused in a jail" do
    {:ok, st} = D.write_file(D.new(), "p.alb", Vapor.Athanor.Examples.get("golomb").text, :write)
    {r, _} = run(st, ~s{athanor run p.alb --measure "echo 1"})
    assert r.code == 3 and r.err =~ "--measure"
  end

  test "a runaway command is stopped by its deadline or its heap ceiling; the session goes on" do
    {r, st} = run(D.new(timeout: 30), "cupel --n 128 --k 256 --trials 24")
    assert r.code == 4 and r.err =~ "stopped"
    {r, _} = run(%{st | timeout: 120_000, heap_mb: 4}, "cupel --n 128 --k 256 --trials 24")
    assert r.code == 4 and r.err =~ "MB"
    {r, _} = run(st, "echo still here")
    assert r.out == "still here\n"
  end

  test "file limits in a jail" do
    st = D.new()
    big = String.duplicate("x", 9 * 1024 * 1024)
    assert {:error, msg} = D.write_file(st, "big", big, :write)
    assert msg =~ "MB"
    st = Enum.reduce(1..128, st, fn i, st -> {:ok, st} = D.write_file(st, "f#{i}", "x", :write); st end)
    assert {:error, msg} = D.write_file(st, "one-more", "x", :write)
    assert msg =~ "at most"
  end

  test "chat through the terminal: the same verbs as the command line, on the server's majlis" do
    dir = Path.join(System.tmp_dir!(), "diwan-majlis-#{System.unique_integer([:positive])}")
    {:ok, m} = Vapor.Majlis.start_link(dir: dir, backends: %{"echo" => %Vapor.MajlisTest.Echo{}})
    st = D.new(majlis: m, tty: false)
    {r, st} = run(st, "chat new --title notes")
    {:ok, %{"thread" => tid}} = Vapor.JSON.decode(r.out)
    {r, st} = run(st, "chat say #{tid} hello there")
    assert {:ok, %{"content" => "echo: hello there (1 msgs)"}} = Vapor.JSON.decode(r.out)
    {r, st} = run(st, "chat search hello")
    assert {:ok, [%{"threads" => [^tid]} | _]} = Vapor.JSON.decode(r.out)
    {r, _} = run(%{st | tty: true}, "chat show #{tid}")
    assert r.out =~ "hello there" and r.code == 0
    File.rm_rf!(dir)
  end

  test "unknown commands and bad usage say so with the conventional exit status" do
    {r, _} = run(D.new(), "frobnicate now")
    assert r.code == 2 and r.err =~ "unknown command"
    {r, _} = run(D.new(), "vapor help")
    assert r.out =~ "vapor"
  end
end
