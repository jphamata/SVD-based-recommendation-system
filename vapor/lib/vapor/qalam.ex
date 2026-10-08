defmodule Vapor.Qalam do
  @moduledoc """
  **Al-Qalam** (القلم, the reed pen): vapor's editor, in the terminal and in
  the suckless manner. It edits text and shows what vapor decides about it;
  everything else belongs to the file and the person (docs/EDITORS.md,
  "Al-Qalam").

      vapor qalam energy.wzn

  * **Modal**, a small subset of vi: `h j k l w b 0 ^ $ gg G` (and the
    arrows), counts (`3j`, `2dd`), `i a I A o O`, `x dd D yy p P J`, `u` and
    `Ctrl-R`, `/` and `n`, `:w :q :wq :x :q! ZZ`, `:N` to go to line N.
  * **The balance in the gutter.** In an Almizan file each claim carries its
    verdict on its first line: `✓` proved, `✗` refuted, `?` undecided. The
    file is rechecked after every change made outside insert mode, never
    while typing. `K` shows the verdict's detail (the decider, the
    counterexample). A line that does not parse is marked `!`, and so is
    the line of an Alembic load error.
  * **Scrubbable numbers.** `Ctrl-A` and `Ctrl-X` step the number under or
    after the cursor up and down (by the count, if one is typed): an integer
    by 1, an exact fraction by its own unit 1/d (`1/10` walks to `0/10`, then
    `-1/10`), in Latin or Arabic-Indic digits. The gutter answers at once, so
    a coefficient can be walked to where a law starts or stops holding.
  * **Structure.** `%` jumps to the matching bracket; `[[` and `]]` to the
    previous and next top-level form.
  * **Canon and lens.** `:fmt` rewrites an Almizan file in canonical form,
    comments kept (`Vapor.Almizan.Format`); `:ar` and `:la` show it in the
    Arabic or the Latin script: the same tree, the same hash.
  * **A Merkle undo tree.** A state's identity is the SHA-256 of its
    parent's identity and its text. Undoing and then changing starts a
    branch, and the old branch stays: `u` and `Ctrl-R` walk the tree, `g-`
    and `g+` walk every state in the order it was made, `:tree` counts them.

  What it does not do, on purpose: colours for syntax, plugins,
  configuration, the mouse, windows, the network. A wide character (CJK)
  counts as one column, and right-to-left text is laid out by the terminal.

  The core is pure. `feed/2` takes a state and keys and returns a state;
  `render/2` takes a state and a size and returns the screen. So the editor
  is tested without a terminal, and `run/2` is the loop around the two.
  """
  alias Vapor.{Alembic, Almizan}
  alias Vapor.Almizan.Format

  defstruct lines: [""], row: 0, col: 0, mode: :normal, path: nil, kind: :plain, eol: true,
            top: 0, left: 0, cmd: "", pending: "", count: nil, reg: [], search: nil, msg: "",
            marks: %{}, summary: "", stale: true, tree: %{}, at: nil, order: [], saved: [""],
            quit: false, color: false

  @check_ms 5_000

  # ================================================================ state

  @doc "A state for `text`, as if read from `path` (options: `path`, `color`)."
  def new(text, opts \\ []) do
    path = opts[:path]
    eol = text == "" or String.ends_with?(text, "\n")
    lines = text |> String.trim_trailing("\n") |> String.split("\n")
    st = %__MODULE__{lines: lines, path: path, kind: kind(path), eol: eol, saved: lines, color: opts[:color] || false}
    id = node_id(nil, lines)
    %{st | tree: %{id => %{lines: lines, parent: nil, cursor: {0, 0}}}, at: id, order: [id]}
  end

  @doc "Open `path` (a missing file is a new, empty buffer)."
  def open(path, opts \\ []) do
    case File.read(path) do
      {:ok, text} -> if String.valid?(text), do: {:ok, new(text, Keyword.put(opts, :path, path))}, else: {:error, "#{path}: not UTF-8 text"}
      {:error, :enoent} -> {:ok, %{new("", Keyword.put(opts, :path, path)) | msg: "new file"}}
      {:error, e} -> {:error, "#{path}: #{:file.format_error(e)}"}
    end
  end

  @doc "The buffer's text."
  def text(%__MODULE__{lines: lines, eol: eol}), do: Enum.join(lines, "\n") <> if(eol, do: "\n", else: "")

  defp kind(nil), do: :plain
  defp kind(path), do: (case Path.extname(path) do ".wzn" -> :almizan; ".nbq" -> :alembic; _ -> :plain end)

  # ================================================================ keys

  @doc """
  Bytes from a terminal → keys: graphemes, `:esc`, `:enter`, `:backspace`,
  `:tab`, `:delete`, the arrows (`:up` …), `:home`, `:end`, `{:ctrl, ?a}`.
  """
  def keys(bin), do: keys(bin, [])

  defp keys("", acc), do: Enum.reverse(acc)
  defp keys("\e[" <> <<c, rest::binary>>, acc) when c in ~c"ABCDHF", do: keys(rest, [arrow(c) | acc])
  defp keys("\eO" <> <<c, rest::binary>>, acc) when c in ~c"ABCDHF", do: keys(rest, [arrow(c) | acc])
  defp keys("\e[3~" <> rest, acc), do: keys(rest, [:delete | acc])
  defp keys("\e" <> rest, acc), do: keys(rest, [:esc | acc])
  defp keys("\r\n" <> rest, acc), do: keys(rest, [:enter | acc])
  defp keys(<<c, rest::binary>>, acc) when c in [?\r, ?\n], do: keys(rest, [:enter | acc])
  defp keys(<<c, rest::binary>>, acc) when c in [127, 8], do: keys(rest, [:backspace | acc])
  defp keys("\t" <> rest, acc), do: keys(rest, [:tab | acc])
  defp keys(<<c, rest::binary>>, acc) when c < 32, do: keys(rest, [{:ctrl, c + 96} | acc])

  defp keys(bin, acc) do
    case String.next_grapheme(bin) do
      {g, rest} -> keys(rest, [g | acc])
      nil -> Enum.reverse(acc)
    end
  end

  defp arrow(c), do: %{?A => :up, ?B => :down, ?C => :right, ?D => :left, ?H => :home, ?F => :end}[c]

  @doc "Apply keys (a list, or a string decoded by `keys/1`), then recheck if the text changed."
  def feed(st, keys) when is_binary(keys), do: feed(st, keys(keys))

  def feed(st, keys) do
    keys
    |> Enum.reduce_while(st, fn k, s -> s = key(s, k); if s.quit, do: {:halt, s}, else: {:cont, s} end)
    |> refresh()
  end

  # ---------------------------------------------------------------- insert

  defp key(%{mode: :insert} = st, k) do
    case k do
      :esc -> %{commit(st) | mode: :normal, col: max(st.col - 1, 0)}
      :enter ->
        {before, rest} = split(cur(st), st.col)
        indent = Regex.run(~r/^[ \t]*/, before) |> hd()
        %{set_lines(st, List.replace_at(st.lines, st.row, before) |> List.insert_at(st.row + 1, indent <> rest)) | row: st.row + 1, col: glen(indent)}
      :backspace ->
        cond do
          st.col > 0 -> {b, a} = split(cur(st), st.col); %{put_cur(st, drop_last(b) <> a) | col: st.col - 1}
          st.row > 0 ->
            prev = Enum.at(st.lines, st.row - 1)
            lines = st.lines |> List.replace_at(st.row - 1, prev <> cur(st)) |> List.delete_at(st.row)
            %{set_lines(st, lines) | row: st.row - 1, col: glen(prev)}
          true -> st
        end
      :tab -> ins(st, "  ")
      k when k in [:left, :right, :up, :down, :home, :end] -> move(st, k, true)
      :delete -> delete_char(st)
      g when is_binary(g) -> ins(st, g)
      _ -> st
    end
  end

  # ---------------------------------------------------------------- command line and search

  defp key(%{mode: m} = st, k) when m in [:command, :search] do
    case k do
      :esc -> %{st | mode: :normal, cmd: ""}
      :enter -> if m == :command, do: ex(%{st | mode: :normal}, String.trim(st.cmd)), else: find(%{st | mode: :normal, search: st.cmd}, st.cmd)
      :backspace -> if st.cmd == "", do: %{st | mode: :normal}, else: %{st | cmd: drop_last(st.cmd)}
      g when is_binary(g) -> %{st | cmd: st.cmd <> g}
      _ -> st
    end
  end

  # ---------------------------------------------------------------- normal

  defp key(%{mode: :normal, pending: p} = st, k) when p != "" do
    n = st.count || 1
    st = %{st | pending: "", count: nil}

    case {p, k} do
      {"g", "g"} -> %{st | row: if(n > 1, do: min(n - 1, last(st)), else: 0), col: 0}
      {"g", "-"} -> walk(st, -1)
      {"g", "+"} -> walk(st, 1)
      {"d", "d"} ->
        rows = Enum.slice(st.lines, st.row, n)
        rest = List.delete_at(st.lines, st.row) |> then(fn l -> Enum.reduce(2..n//1, l, fn _, acc -> List.delete_at(acc, st.row) end) end)
        rest = if rest == [], do: [""], else: rest
        st = %{set_lines(st, rest) | reg: rows}
        commit(%{st | row: min(st.row, last(st)), col: 0})
      {"y", "y"} -> %{st | reg: Enum.slice(st.lines, st.row, n), msg: "#{min(n, length(st.lines) - st.row)} line(s) yanked"}
      {"Z", "Z"} -> ex(st, "x")
      {"[", "["} -> form(st, -1)
      {"]", "]"} -> form(st, 1)
      _ -> st
    end
  end

  defp key(%{mode: :normal} = st, d) when d in ~w(1 2 3 4 5 6 7 8 9) or (d == "0" and st.count != nil),
    do: %{st | count: (st.count || 0) * 10 + String.to_integer(d)}

  defp key(%{mode: :normal} = st, k) do
    n = st.count || 1
    st = %{st | count: nil, msg: ""}

    case k do
      k when k in ["g", "d", "y", "Z", "[", "]"] -> %{st | pending: k, count: if(n > 1, do: n)}
      k when k in ["h", "j", "k", "l", "w", "b", :left, :right, :up, :down] -> Enum.reduce(1..n, st, fn _, s -> move(s, k, false) end)
      k when k in ["0", :home] -> %{st | col: 0}
      "^" -> %{st | col: glen(Regex.run(~r/^\s*/u, cur(st)) |> hd())}
      k when k in ["$", :end] -> %{st | col: max(glen(cur(st)) - 1, 0)}
      "G" -> %{st | row: if(st.count == nil and n == 1, do: last(st), else: min(n - 1, last(st))), col: 0}
      "i" -> %{st | mode: :insert}
      "a" -> %{st | mode: :insert, col: min(st.col + 1, glen(cur(st)))}
      "I" -> %{st | mode: :insert, col: glen(Regex.run(~r/^\s*/u, cur(st)) |> hd())}
      "A" -> %{st | mode: :insert, col: glen(cur(st))}
      "o" -> %{set_lines(st, List.insert_at(st.lines, st.row + 1, "")) | row: st.row + 1, col: 0, mode: :insert}
      "O" -> %{set_lines(st, List.insert_at(st.lines, st.row, "")) | col: 0, mode: :insert}
      k when k in ["x", :delete] -> commit(Enum.reduce(1..n, st, fn _, s -> delete_char(s) end) |> clamp())
      "D" -> {b, _} = split(cur(st), st.col); commit(clamp(put_cur(st, b)))
      "J" ->
        if st.row < last(st) do
          next = Enum.at(st.lines, st.row + 1) |> String.trim_leading()
          joined = String.trim_trailing(cur(st)) <> if(next == "", do: "", else: " " <> next)
          commit(%{set_lines(st, st.lines |> List.replace_at(st.row, joined) |> List.delete_at(st.row + 1)) | col: glen(cur(st))})
        else
          st
        end
      "p" -> if st.reg == [], do: st, else: commit(%{set_lines(st, List.insert_at(st.lines, st.row + 1, st.reg) |> List.flatten()) | row: st.row + 1, col: 0})
      "P" -> if st.reg == [], do: st, else: commit(%{set_lines(st, List.insert_at(st.lines, st.row, st.reg) |> List.flatten()) | col: 0})
      "u" -> Enum.reduce(1..n, st, fn _, s -> undo(s) end)
      {:ctrl, ?r} -> Enum.reduce(1..n, st, fn _, s -> redo(s) end)
      "%" -> match(st)
      {:ctrl, ?a} -> scrub(st, n)
      {:ctrl, ?x} -> scrub(st, -n)
      "/" -> %{st | mode: :search, cmd: ""}
      "n" -> if st.search, do: find(st, st.search), else: st
      ":" -> %{st | mode: :command, cmd: ""}
      "K" -> detail(refresh(st))
      _ -> st
    end
  end

  # ================================================================ edits

  defp cur(st), do: Enum.at(st.lines, st.row, "")
  defp last(st), do: length(st.lines) - 1
  defp glen(s), do: String.length(s)
  defp split(line, col), do: {String.slice(line, 0, col), String.slice(line, col..-1//1)}
  defp drop_last(s), do: String.slice(s, 0, max(glen(s) - 1, 0))
  defp put_cur(st, line), do: set_lines(st, List.replace_at(st.lines, st.row, line))
  defp set_lines(st, lines), do: %{st | lines: lines}

  defp ins(st, g) do
    {b, a} = split(cur(st), st.col)
    %{put_cur(st, b <> g <> a) | col: st.col + glen(g)}
  end

  defp delete_char(st) do
    line = cur(st)
    if st.col < glen(line), do: put_cur(st, String.slice(line, 0, st.col) <> String.slice(line, (st.col + 1)..-1//1)), else: st
  end

  # in normal mode the cursor sits on a character; in insert mode it may sit after the last
  defp clamp(st), do: %{st | row: st.row |> max(0) |> min(last(st)), col: st.col |> min(max(glen(cur(%{st | row: min(st.row, last(st))})) - 1, 0)) |> max(0)}

  defp move(st, k, insert?) do
    limit = fn s -> if insert?, do: glen(cur(s)), else: max(glen(cur(s)) - 1, 0) end

    case k do
      k when k in ["h", :left] -> %{st | col: max(st.col - 1, 0)}
      k when k in ["l", :right] -> %{st | col: min(st.col + 1, limit.(st))}
      k when k in ["j", :down] -> s = %{st | row: min(st.row + 1, last(st))}; %{s | col: min(st.col, limit.(s))}
      k when k in ["k", :up] -> s = %{st | row: max(st.row - 1, 0)}; %{s | col: min(st.col, limit.(s))}
      :home -> %{st | col: 0}
      :end -> %{st | col: limit.(st)}
      "w" -> word(st, 1)
      "b" -> word(st, -1)
    end
  end

  # the start of the next (or previous) word: a run of letters and digits, or of other non-space characters
  defp word(st, dir) do
    here = starts(cur(st))

    case if(dir > 0, do: Enum.find(here, &(&1 > st.col)), else: here |> Enum.reverse() |> Enum.find(&(&1 < st.col))) do
      nil ->
        rows = if dir > 0, do: (st.row + 1)..last(st)//1, else: (st.row - 1)..0//-1

        Enum.find_value(rows, st, fn r ->
          case starts(Enum.at(st.lines, r)) do
            [] -> nil
            cs -> %{st | row: r, col: if(dir > 0, do: hd(cs), else: List.last(cs))}
          end
        end)

      c ->
        %{st | col: c}
    end
  end

  defp starts(line) do
    cls = line |> String.graphemes() |> Enum.map(&class/1)
    for {{prev, c}, i} <- Enum.with_index(Enum.zip([:space | cls], cls)), c != :space, prev != c, do: i
  end

  defp class(g) do
    cond do
      g =~ ~r/^\s$/u -> :space
      g =~ ~r/^[\p{L}\p{N}_\-]$/u -> :word
      true -> :punct
    end
  end

  # the first match after the cursor, wrapping to the first in the buffer
  defp find(st, pat) do
    hits = for {line, r} <- Enum.with_index(st.lines), {b, _} <- :binary.matches(line, pat), do: {r, glen(binary_part(line, 0, b))}

    case Enum.find(hits, &(&1 > {st.row, st.col})) || List.first(hits) do
      {r, c} -> %{st | row: r, col: c, msg: "/" <> pat}
      nil -> %{st | msg: "not found: " <> pat}
    end
  end

  # ---------------------------------------------------------------- structure

  @open ~w|( [|
  @close ~w|) ]|

  defp match(st) do
    line = cur(st)
    gs = String.graphemes(line)

    case Enum.find_index(Enum.drop(gs, st.col), &(&1 in @open or &1 in @close)) do
      nil -> st
      off ->
        c = st.col + off
        g = Enum.at(gs, c)
        dir = if g in @open, do: 1, else: -1
        chars = flat_code(st)
        i = Enum.find_index(chars, fn {r, cc, _} -> r == st.row and cc == c end)
        seq = if dir > 0, do: Enum.drop(chars, i), else: chars |> Enum.take(i + 1) |> Enum.reverse()

        found =
          Enum.reduce_while(seq, 0, fn {r, cc, x}, depth ->
            depth = cond do x in @open -> depth + dir; x in @close -> depth - dir; true -> depth end
            if depth == 0, do: {:halt, {r, cc}}, else: {:cont, depth}
          end)

        case found do
          {r, cc} -> %{st | row: r, col: cc}
          _ -> %{st | msg: "no matching bracket"}
        end
    end
  end

  # brackets outside comments (`;` in Almizan, `#` in Alembic), with their positions
  defp flat_code(st) do
    comment = %{almizan: ";", alembic: "#"}[st.kind]

    for {l, r} <- Enum.with_index(st.lines),
        code = (if comment, do: l |> String.split(comment, parts: 2) |> hd(), else: l),
        {g, c} <- Enum.with_index(String.graphemes(code)), g in @open or g in @close,
        do: {r, c, g}
  end

  defp form(st, dir) do
    top? = fn l -> if st.kind == :almizan, do: String.starts_with?(l, "("), else: l =~ ~r/^[^\s#;]/u end
    rows = if dir > 0, do: (st.row + 1)..last(st)//1, else: (st.row - 1)..0//-1

    case Enum.find(rows, &top?.(Enum.at(st.lines, &1))) do
      nil -> st
      r -> %{st | row: r, col: 0}
    end
  end

  # ---------------------------------------------------------------- scrubbing

  @ar ~w(٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩)

  defp scrub(st, delta) do
    line = cur(st)
    re = ~r/(?<![\p{L}\p{N}_])-?[0-9٠-٩]+(?:\/[0-9٠-٩]+)?/u

    hit =
      Regex.scan(re, line, return: :index)
      |> Enum.map(fn [{b, len}] -> {glen(binary_part(line, 0, b)), glen(binary_part(line, b, len)), binary_part(line, b, len)} end)
      |> Enum.find(fn {c, len, _} -> c + len > st.col end)

    case hit do
      nil -> %{st | msg: "no number on this line after the cursor"}
      {c, len, lit} ->
        arabic? = String.contains?(lit, @ar)
        # a fraction keeps its denominator, so a walk keeps its unit
        out =
          case String.split(latin(lit), "/") do
            [n] -> Integer.to_string(String.to_integer(n) + delta)
            [n, d] -> "#{String.to_integer(n) + delta}/#{d}"
          end
          |> then(&if(arabic?, do: arabic(&1), else: &1))
        {b, _} = split(line, c)
        a = String.slice(line, (c + len)..-1//1)
        commit(%{put_cur(st, b <> out <> a) | col: c + glen(out) - 1})
    end
  end

  defp latin(s), do: Enum.reduce(Enum.with_index(@ar), s, fn {d, i}, acc -> String.replace(acc, d, Integer.to_string(i)) end)
  defp arabic(s), do: Enum.reduce(Enum.with_index(@ar), s, fn {d, i}, acc -> String.replace(acc, Integer.to_string(i), d) end)

  # ================================================================ the Merkle undo tree

  defp node_id(parent, lines), do: :crypto.hash(:sha256, [parent || "", 0, Enum.join(lines, "\n")]) |> Base.encode16(case: :lower) |> binary_part(0, 16)

  # a change becomes a node, the child of the state it changed
  defp commit(st) do
    if st.lines == st.tree[st.at].lines do
      st
    else
      id = node_id(st.at, st.lines)
      tree = Map.put(st.tree, id, %{lines: st.lines, parent: st.at, cursor: {st.row, st.col}})
      %{st | tree: tree, at: id, order: st.order ++ [id], stale: true}
    end
  end

  defp goto_node(st, id, cursor) do
    {r, c} = cursor
    clamp(%{st | lines: st.tree[id].lines, at: id, row: r, col: c, stale: true})
  end

  defp undo(st) do
    node = st.tree[st.at]
    if node.parent, do: goto_node(st, node.parent, node.cursor), else: %{st | msg: "at the oldest state"}
  end

  # the newest child of this state
  defp redo(st) do
    case Enum.filter(st.order, &(st.tree[&1].parent == st.at)) |> List.last() do
      nil -> %{st | msg: "at the newest state of this branch"}
      id -> goto_node(st, id, st.tree[id].cursor)
    end
  end

  defp walk(st, dir) do
    i = Enum.find_index(st.order, &(&1 == st.at)) + dir

    if i < 0 or i >= length(st.order) do
      %{st | msg: "no state #{if dir < 0, do: "before", else: "after"} this one"}
    else
      id = Enum.at(st.order, i)
      goto_node(st, id, st.tree[id].cursor)
    end
  end

  @doc "The undo tree: `%{states, branches, at}` (a branch is a state with no child)."
  def tree(st) do
    parents = st.tree |> Map.values() |> Enum.map(& &1.parent) |> MapSet.new()
    %{states: map_size(st.tree), branches: Enum.count(Map.keys(st.tree), &(not MapSet.member?(parents, &1))), at: st.at}
  end

  # ================================================================ ex commands

  defp ex(st, cmd) do
    case cmd do
      "w" -> write(st)
      "q" -> if dirty?(st), do: %{st | msg: "unsaved changes (:w to write, :q! to leave them)"}, else: %{st | quit: true}
      "q!" -> %{st | quit: true}
      c when c in ["wq", "x"] -> st = write(st); if dirty?(st), do: st, else: %{st | quit: true}
      "fmt" -> canon(st, nil)
      "ar" -> canon(st, :arabic)
      "la" -> canon(st, :latin)
      "check" -> refresh(%{st | stale: true})
      "tree" -> t = tree(st); %{st | msg: "#{t.states} states, #{t.branches} branch(es), at #{t.at}"}
      "" -> st
      c ->
        case Integer.parse(c) do
          {n, ""} -> %{st | row: n |> Kernel.-(1) |> max(0) |> min(last(st)), col: 0}
          _ -> %{st | msg: "not a command: :" <> c}
        end
    end
  end

  defp dirty?(st), do: st.lines != st.saved

  defp write(%{path: nil} = st), do: %{st | msg: "no file name"}

  defp write(st) do
    tmp = st.path <> ".qalam#{System.unique_integer([:positive])}"

    with :ok <- File.write(tmp, text(st)), :ok <- File.rename(tmp, st.path) do
      %{st | saved: st.lines, msg: "written: #{st.path} (#{length(st.lines)} lines)"}
    else
      {:error, e} -> File.rm(tmp); %{st | msg: "not written: #{:file.format_error(e)}"}
    end
  end

  defp canon(%{kind: :almizan} = st, proj) do
    case Format.format(Enum.join(st.lines, "\n") <> "\n", proj) do
      {:ok, t} -> commit(clamp(%{st | lines: t |> String.trim_trailing("\n") |> String.split("\n"), eol: true, msg: if(proj, do: "the #{proj} script: the same tree", else: "canonical form")}))
      {:error, why} -> %{st | msg: "not formatted: " <> why}
    end
  end

  defp canon(st, _), do: %{st | msg: "formatting is for Almizan files (.wzn)"}

  # ================================================================ the balance

  @doc "Recheck the file if it changed since the last check: the gutter's marks and the summary."
  def refresh(%{stale: false} = st), do: st

  def refresh(st) do
    text = Enum.join(st.lines, "\n") <> "\n"

    {marks, summary} =
      case Vapor.Hermetic.seal(fn -> verdicts(st.kind, text) end, timeout: @check_ms) do
        {:ok, v} -> v
        {:error, :timeout} -> {%{}, "the check took over #{div(@check_ms, 1000)} s (:check to try again)"}
        {:error, e} -> {%{}, "the check failed: #{inspect(e)}"}
      end

    %{st | marks: marks, summary: summary, stale: false}
  end

  defp verdicts(:almizan, text) do
    case Almizan.verdict_lines(text, depth: 12) do
      {:ok, rs} ->
        marks =
          for r <- rs, r.verdict != "none", into: %{} do
            cex = if r[:counterexample], do: " — at " <> Enum.map_join(r.counterexample, ", ", fn {k, v} -> "#{k} = #{Almizan.show(v)}" end), else: ""
            {r.line - 1, {mark(r.verdict), "#{r.claim}: #{r.verdict} (#{r.decider}) #{r.detail}#{cex}"}}
          end

        counts = rs |> Enum.reject(&(&1.verdict == "none")) |> Enum.frequencies_by(& &1.verdict)
        {marks, ["proved", "refuted", "unknown"] |> Enum.filter(&counts[&1]) |> Enum.map_join(" · ", &"#{counts[&1]} #{&1}")}

      {:error, why, line} -> {%{(line - 1) => {"!", why}}, why}
    end
  end

  defp verdicts(:alembic, text) do
    case Alembic.load(text) do
      {:ok, _} -> {%{}, "loads"}
      {:error, %{message: m, line: l}} -> {%{(max(l, 1) - 1) => {"!", m}}, "line #{l}: #{m}"}
      {:error, other} -> {%{0 => {"!", Alembic.format_error(other)}}, Alembic.format_error(other)}
    end
  end

  defp verdicts(:plain, _), do: {%{}, ""}

  defp mark("proved"), do: "✓"
  defp mark("refuted"), do: "✗"
  defp mark(_), do: "?"

  defp detail(st) do
    case st.marks[st.row] do
      {_, d} -> %{st | msg: d}
      nil -> %{st | msg: "nothing decided on this line"}
    end
  end

  # ================================================================ the screen

  @doc """
  The screen for a terminal of `{rows, cols}`: `{state, lines, {row, col}}`,
  the state scrolled so the cursor is visible, one string per row (the last
  row is the status line), and the cursor's position on the screen.
  """
  def render(st, {rows, cols}) do
    body = max(rows - 1, 1)
    width = max(cols - 2, 1)
    top = cond do st.row < st.top -> st.row; st.row >= st.top + body -> st.row - body + 1; true -> st.top end
    left = cond do st.col < st.left -> st.col; st.col >= st.left + width -> st.col - width + 1; true -> st.left end
    st = %{st | top: top, left: left}

    text_rows =
      for i <- 0..(body - 1) do
        r = top + i

        case Enum.at(st.lines, r) do
          nil -> "~"
          line -> gutter(st, r) <> " " <> String.slice(line, left, width)
        end
      end

    {st, text_rows ++ [status(st, cols)], {st.row - top, st.col - left + 2}}
  end

  defp gutter(st, r) do
    case st.marks[r] do
      nil -> " "
      {m, _} -> if st.color, do: color(m) <> m <> "\e[0m", else: m
    end
  end

  defp color("✓"), do: "\e[32m"
  defp color("✗"), do: "\e[31m"
  defp color("!"), do: "\e[1;31m"
  defp color(_), do: "\e[33m"

  defp status(st, cols) do
    line =
      case st.mode do
        :command -> ":" <> st.cmd
        :search -> "/" <> st.cmd
        m ->
          name = if st.path, do: Path.basename(st.path), else: "[no name]"
          tag = if m == :insert, do: "INSERT", else: "NORMAL"
          info = Enum.reject([st.msg, st.summary], &(&1 in [nil, ""])) |> List.first() || ""
          "#{tag}  #{name}#{if dirty?(st), do: " [+]", else: ""}  #{st.row + 1}:#{st.col + 1}  #{info}"
      end

    line = String.slice(line, 0, cols)
    if st.color, do: "\e[7m" <> String.pad_trailing(line, cols) <> "\e[0m", else: line
  end

  # ================================================================ the terminal

  @doc "Edit `path` on this terminal. Returns `:ok` or `{:error, why}`."
  def run(path, opts \\ []) do
    with true <- tty?() || {:error, "qalam needs a terminal"},
         {:ok, st} <- open(path, Keyword.put_new(opts, :color, System.get_env("NO_COLOR") in [nil, ""])) do
      case :shell.start_interactive({:noshell, :raw}) do
        r when r in [:ok, {:error, :already_started}] -> :ok
      end

      parent = self()
      reader = spawn_link(fn -> read_loop(parent) end)
      IO.write("\e[?1049h")

      try do
        loop(st)
      after
        Process.unlink(reader)
        Process.exit(reader, :kill)
        IO.write("\e[0m\e[2J\e[H\e[?1049l")
      end
    end
  end

  defp tty?, do: match?({:ok, true}, :io.getopts(:standard_io) |> Keyword.fetch(:terminal))

  defp read_loop(parent) do
    case IO.getn(:stdio, "", 1) do
      c when is_binary(c) -> send(parent, {:qalam, c}); read_loop(parent)
      _ -> send(parent, {:qalam, :eof})
    end
  end

  defp loop(st) do
    st = refresh(st)
    size = {(case :io.rows() do {:ok, r} -> r; _ -> 24 end), (case :io.columns() do {:ok, c} -> c; _ -> 80 end)}
    {st, rows, {r, c}} = render(st, size)
    IO.write(["\e[?25l\e[H", Enum.map_intersperse(rows, "\r\n", &[&1, "\e[K"]), "\e[#{r + 1};#{c + 1}H\e[?25h"])

    if st.quit do
      :ok
    else
      case next_input() do
        :eof -> :ok
        bin -> st |> then(&Enum.reduce_while(keys(bin), &1, fn k, s -> s = key(s, k); if s.quit, do: {:halt, s}, else: {:cont, s} end)) |> loop()
      end
    end
  end

  # an escape alone is a key; followed at once by more, a sequence (an arrow)
  defp next_input do
    receive do
      {:qalam, :eof} -> :eof
      {:qalam, "\e"} -> "\e" <> more(30)
      {:qalam, c} -> c
    end
  end

  defp more(ms) do
    receive do
      {:qalam, c} when is_binary(c) -> c <> more(5)
    after
      ms -> ""
    end
  end
end
