defmodule Vapor.Mizan.Syntax do
  @moduledoc """
  Al-Mizān's two surface forms over one neutral tree (docs/MIZAN.md §2).

  The tree is the program; text is a **projection** of it. Two projections
  are defined, and each is a bijection with the tree — `read(print(t)) == t`
  for every tree, tested on random trees:

    * **Latin** (ASCII): keywords in English, roots in Buckwalter
      transliteration (`H-f-Z`), Arabic identifiers as `@` + Buckwalter
      (`@twAzn` is توازن);
    * **Arabic**: keywords in Arabic (`دعوى`, `جذر`, `وزن`, `برهان`, `تنفيذ`…),
      roots in Arabic letters (`ح-ف-ظ`), numerals in Arabic-Indic digits.

  A file may be written in either; its identity (`Vapor.Mizan.hash/1`) is the
  SHA-256 of the canonical encoding of the tree, so the same program has the
  same hash in both scripts — the property the manifesto wanted from storing
  Arabic on disk, obtained without forcing any script on anyone.

  Reading is RTL-agnostic: Unicode text is stored in logical order, so the
  same S-expression reader serves both; only the printer chooses words.
  """

  # ---------------------------------------------------------------- tables

  # keyword: Latin ⇄ Arabic (clause heads, wazn names, proof kinds, types, logic)
  @kw [
    {"claim", "دعوى"}, {"root", "جذر"}, {"wazn", "وزن"}, {"inputs", "مدخلات"}, {"field", "حقل"},
    {"box", "صندوق"}, {"step", "خطوة"}, {"init", "بداية"}, {"invariant", "ثابت"}, {"proof", "برهان"},
    {"body", "تنفيذ"}, {"import", "استيراد"}, {"as", "باسم"},
    {"fail", "فاعل"}, {"maful", "مفعول"}, {"burhan", "برهان"},
    {"conserved", "محفوظ"}, {"nonneg", "غير-سالب"}, {"pos", "موجب"}, {"identity", "متطابقة"}, {"bounded", "محدود"},
    {"q", "نسبي"}, {"int", "صحيح"}, {"f64", "عائم٦٤"}, {"f32", "عائم٣٢"}, {"bool", "منطقي"},
    {"and", "و"}, {"or", "أو"}, {"not", "ليس"}, {"if", "إذا"}, {"true", "صواب"}, {"false", "خطأ"}
  ]

  # operator words accepted on input in the Arabic projection (the manifesto's جمع); printed as symbols
  @ar_ops %{"جمع" => "+", "طرح" => "-", "ضرب" => "*", "قسمة" => "/", "أس" => "^"}

  @roots %{"hsb" => {"H-s-b", "ح-س-ب"}, "hfz" => {"H-f-Z", "ح-ف-ظ"}, "nql" => {"n-q-l", "ن-ق-ل"}, "ktb" => {"k-t-b", "ك-ت-ب"}}

  @buckwalter [
    {"'", "ء"}, {"|", "آ"}, {">", "أ"}, {"&", "ؤ"}, {"<", "إ"}, {"}", "ئ"}, {"A", "ا"}, {"b", "ب"}, {"p", "ة"}, {"t", "ت"},
    {"v", "ث"}, {"j", "ج"}, {"H", "ح"}, {"x", "خ"}, {"d", "د"}, {"*", "ذ"}, {"r", "ر"}, {"z", "ز"}, {"s", "س"}, {"$", "ش"},
    {"S", "ص"}, {"D", "ض"}, {"T", "ط"}, {"Z", "ظ"}, {"E", "ع"}, {"g", "غ"}, {"_", "ـ"}, {"f", "ف"}, {"q", "ق"}, {"k", "ك"},
    {"l", "ل"}, {"m", "م"}, {"n", "ن"}, {"h", "ه"}, {"w", "و"}, {"Y", "ى"}, {"y", "ي"}, {"F", "ً"}, {"N", "ٌ"}, {"K", "ٍ"},
    {"a", "َ"}, {"u", "ُ"}, {"i", "ِ"}, {"~", "ّ"}, {"o", "ْ"}, {"`", "ٰ"}, {"{", "ٱ"}
  ]

  @lat2ar Map.new(@kw)
  @ar2lat Map.new(@kw, fn {l, a} -> {a, l} end)
  @bw_to_ar Map.new(@buckwalter)
  @ar_to_bw Map.new(@buckwalter, fn {b, a} -> {a, b} end)
  @digits_ar ~w(٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩)

  @doc "The root names: internal id → {Latin, Arabic}."
  def roots, do: @roots

  @doc "Latin keyword → Arabic keyword."
  def arabic(kw), do: Map.get(@lat2ar, kw, kw)

  # ------------------------------------------------------------------ read

  @doc """
  Text (either projection) → S-expressions: `{:list, [..], line}`, `{:atom,
  text, line}`, `{:num, {n, d}, line}`. Comments run from `;` to the end of
  the line.
  """
  def read(text) when is_binary(text) do
    with true <- String.valid?(text) || {:error, "the file is not UTF-8"},
         {:ok, toks} <- tokens(text, 1, [], "") do
      forms(toks, [])
    end
  end

  defp tokens("", _line, acc, cur), do: {:ok, Enum.reverse(flush(acc, cur))}
  defp tokens("\n" <> r, line, acc, cur), do: tokens(r, line + 1, flush(acc, cur, line), "")
  defp tokens(";" <> r, line, acc, cur) do
    rest = case String.split(r, "\n", parts: 2) do [_, more] -> "\n" <> more; [_] -> "" end
    tokens(rest, line, flush(acc, cur, line), "")
  end

  defp tokens("(" <> r, line, acc, cur), do: tokens(r, line, [{:open, line} | flush(acc, cur, line)], "")
  defp tokens(")" <> r, line, acc, cur), do: tokens(r, line, [{:close, line} | flush(acc, cur, line)], "")
  defp tokens(<<c::utf8, r::binary>>, line, acc, cur) when c in [?\s, ?\t, ?\r, 0x200F, 0x200E, 0x061C], do: tokens(r, line, flush(acc, cur, line), "")
  defp tokens(<<c::utf8, r::binary>>, line, acc, cur), do: tokens(r, line, acc, cur <> <<c::utf8>>)

  defp flush(acc, cur, line \\ 0)
  defp flush(acc, "", _line), do: acc
  defp flush(acc, cur, line), do: [{:word, cur, line} | acc]

  defp forms([], acc), do: {:ok, Enum.reverse(acc)}

  defp forms(toks, acc) do
    case form(toks) do
      {:ok, f, rest} -> forms(rest, [f | acc])
      {:error, _} = e -> e
    end
  end

  defp form([{:open, line} | rest]), do: list(rest, line, [])
  defp form([{:close, line} | _]), do: {:error, "line #{line}: a ) with no ("}
  defp form([{:word, w, line} | rest]), do: {:ok, atom_or_number(w, line), rest}

  defp list([], line, _acc), do: {:error, "line #{line}: a ( that is never closed"}
  defp list([{:close, _} | rest], line, acc), do: {:ok, {:list, Enum.reverse(acc), line}, rest}

  defp list(toks, line, acc) do
    case form(toks) do
      {:ok, f, rest} -> list(rest, line, [f | acc])
      e -> e
    end
  end

  defp atom_or_number(w, line) do
    ascii = latin_digits(w)

    cond do
      Regex.match?(~r/^-?\d+$/, ascii) -> {:num, {String.to_integer(ascii), 1}, line}
      Regex.match?(~r/^-?\d+\/\d+$/, ascii) ->
        [n, d] = String.split(ascii, "/")
        d = String.to_integer(d)
        if d == 0, do: {:atom, w, line}, else: {:num, Vapor.Logic.LP.q(String.to_integer(n), d), line}

      Regex.match?(~r/^-?\d+\.\d+$/, ascii) ->
        [i, f] = String.split(String.trim_leading(ascii, "-"), ".")
        sign = if String.starts_with?(ascii, "-"), do: -1, else: 1
        {:num, Vapor.Logic.LP.q(sign * String.to_integer(i <> f), Integer.pow(10, String.length(f))), line}

      true ->
        {:atom, w, line}
    end
  end

  defp latin_digits(w), do: Enum.reduce(Enum.with_index(@digits_ar), w, fn {d, i}, acc -> String.replace(acc, d, Integer.to_string(i)) end)
  defp arabic_digits(w), do: Enum.reduce(Enum.with_index(@digits_ar), w, fn {d, i}, acc -> String.replace(acc, Integer.to_string(i), d) end)

  @doc "Which projection a text is written in: `:arabic` if any keyword is Arabic, else `:latin`."
  def projection(text), do: if(Regex.match?(~r/\p{Arabic}/u, text), do: :arabic, else: :latin)

  # --------------------------------------------------- words in either script

  @doc "A keyword read from either projection (nil if the word is not one)."
  def keyword(w) do
    cond do
      Map.has_key?(@lat2ar, w) -> w
      Map.has_key?(@ar2lat, w) -> @ar2lat[w]
      true -> nil
    end
  end

  @doc "An operator symbol (Arabic operator words are accepted on input)."
  def operator(w), do: Map.get(@ar_ops, w, w)

  @doc "A root name from either projection → its id, or nil."
  def root(w) do
    Enum.find_value(@roots, fn {id, {l, a}} -> if w in [l, a], do: id end)
  end

  @doc """
  An identifier as written → its canonical form: Arabic-script identifiers
  stay Arabic; `@`-prefixed Buckwalter is read as the Arabic it stands for;
  plain ASCII identifiers stay ASCII. `{:ok, name}` or `{:error, why}`.
  """
  def ident("@" <> bw) do
    chars = String.graphemes(bw)

    if bw != "" and Enum.all?(chars, &(Map.has_key?(@bw_to_ar, &1) or &1 == "-" or &1 =~ ~r/^\d$/)),
      do: {:ok, Enum.map_join(chars, fn c -> Map.get(@bw_to_ar, c, c) |> then(&if(&1 =~ ~r/^\d$/, do: arabic_digits(&1), else: &1)) end)},
      else: {:error, "@#{bw}: not Buckwalter transliteration"}
  end

  def ident(w) do
    cond do
      Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_\-]*$/, w) -> {:ok, w}
      arabic_ident?(w) -> {:ok, w}
      true -> {:error, "#{inspect(w)} is not a name (Latin letters, digits, - and _, or Arabic letters; not mixed)"}
    end
  end

  defp arabic_ident?(w) do
    chars = String.graphemes(w)
    chars != [] and Enum.all?(chars, &(Map.has_key?(@ar_to_bw, &1) or &1 == "-" or &1 in @digits_ar)) and
      Map.has_key?(@ar_to_bw, hd(chars))
  end

  @doc "An identifier in the Latin projection."
  def ident_latin(name) do
    if Regex.match?(~r/^[A-Za-z_][A-Za-z0-9_\-]*$/, name),
      do: name,
      else: "@" <> Enum.map_join(String.graphemes(name), fn c -> Map.get(@ar_to_bw, c) || latin_digits(c) end)
  end

  # ------------------------------------------------------------------ print

  @doc "A module tree printed in one projection (`:latin` or `:arabic`)."
  def print(%{"decls" => decls}, proj) do
    Enum.map_join(decls, "\n\n", &decl(&1, proj)) <> "\n"
  end

  defp kw(w, :latin), do: w
  defp kw(w, :arabic), do: arabic(w)

  defp id(n, :latin), do: n |> String.split(".") |> Enum.map_join(".", &ident_latin/1)
  defp id(n, :arabic), do: n

  defp num({n, 1}, :latin), do: Integer.to_string(n)
  defp num({n, d}, :latin), do: "#{n}/#{d}"
  defp num(q, :arabic), do: arabic_digits(num(q, :latin))

  defp decl(%{"import" => h, "as" => name}, p), do: "(#{kw("import", p)} #{h} #{kw("as", p)} #{id(name, p)})"

  defp decl(%{"claim" => name} = c, p) do
    {lat, ar} = @roots[c["root"]]
    head = "(#{kw("claim", p)} #{id(name, p)} (#{kw("root", p)} #{if p == :latin, do: lat, else: ar}) (#{kw("wazn", p)} #{kw(c["wazn"], p)})"

    clauses =
      [
        c["inputs"] != [] && "(#{kw("inputs", p)} " <> Enum.map_join(c["inputs"], " ", fn [v, t] -> "(#{id(v, p)} #{kw(t, p)})" end) <> ")",
        c["field"] && "(#{kw("field", p)} " <> Enum.map_join(c["field"], " ", fn [v, e] -> "(#{id(v, p)} #{expr(e, p)})" end) <> ")",
        c["box"] && "(#{kw("box", p)} " <> Enum.map_join(c["box"], " ", fn [v, lo, hi] -> "(#{id(v, p)} #{num(lo, p)} #{num(hi, p)})" end) <> ")",
        c["init"] && "(#{kw("init", p)} #{expr(c["init"], p)})",
        c["step"] && "(#{kw("step", p)} " <> Enum.map_join(c["step"], " ", fn [v, e] -> "(#{id(v, p)} #{expr(e, p)})" end) <> ")",
        c["invariant"] && "(#{kw("invariant", p)} #{expr(c["invariant"], p)})",
        c["proof"] && "(#{kw("proof", p)} #{proof(c["proof"], p)})",
        "(#{kw("body", p)} #{expr(c["body"], p)})"
      ]
      |> Enum.reject(&(&1 in [nil, false]))

    head <> "\n  " <> Enum.join(clauses, "\n  ") <> ")"
  end

  defp proof(%{"kind" => k}, p) when k in ["conserved", "nonneg", "pos", "invariant"], do: kw(k, p)
  defp proof(%{"kind" => "identity", "rhs" => r}, p), do: "(#{kw("identity", p)} #{expr(r, p)})"
  defp proof(%{"kind" => "bounded", "lo" => lo, "hi" => hi}, p), do: "(#{kw("bounded", p)} #{num(lo, p)} #{num(hi, p)})"

  @doc "An expression printed in one projection."
  def expr(["q", n, d], p), do: num({n, d}, p)
  def expr(["b", true], p), do: kw("true", p)
  def expr(["b", false], p), do: kw("false", p)
  def expr(["v", name], p), do: id(name, p)
  def expr(["op", op, args], p) when op in ["and", "or", "not", "if"], do: "(" <> Enum.join([kw(op, p) | Enum.map(args, &expr(&1, p))], " ") <> ")"
  def expr(["op", op, args], p), do: "(" <> Enum.join([op | Enum.map(args, &expr(&1, p))], " ") <> ")"
  def expr(["call", f, args], p), do: "(" <> Enum.join([id(f, p) | Enum.map(args, &expr(&1, p))], " ") <> ")"
end
