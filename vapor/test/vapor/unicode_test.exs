defmodule Vapor.UnicodeTest do
  @moduledoc "NFC/NFKC (UAX #15) against Python's unicodedata, over every code point and random mark sequences."
  use ExUnit.Case, async: true
  alias Vapor.Unicode
  import Vapor.TestHelpers

  test "the primary-composite table has the standard's pairs for its Unicode version; blocking by a class-0 mark is respected" do
    # the count depends on the Unicode data OTP ships (15.x: 941; 16.0, OTP 28: 961, as Python 3.14's unicodedata counts them)
    expected = %{{15, 0} => 941, {15, 1} => 941, {16, 0} => 961}[:unicode_util.spec_version()]
    assert expected != nil, "no recorded count for Unicode #{Unicode.version()}: count it with Python's unicodedata"
    assert Unicode.composites() == expected
    # и, THAI YAMAKKAN (class 0), COMBINING DIAERESIS: no composition across the mark
    assert Unicode.nfc("и๎̈") == "и๎̈"
    assert Unicode.nfc("ӥ") == "ӥ"
    assert Unicode.nfc("é") == "é"
    assert Unicode.nfc("각") == "각"
    assert Unicode.nfkc("ﬁ ½ Ⅻ") == "fi 1⁄2 XII"
    assert Unicode.nfc("plain ascii") == "plain ascii"
  end

  @tag :python
  @tag timeout: 600_000
  test "equal to Python's unicodedata on every code point and on random mark sequences" do
    :rand.seed(:exsss, {5, 5, 5})
    singles = for cp <- Enum.concat(0x20..0xD7FF, 0xE000..0x2FFFF), do: <<cp::utf8>>
    marks = Enum.concat([0x300..0x36F, [0x0E4E, 0x0E48, 0x05B0, 0x0591, 0x1AB0, 0x20D0, 0x302A, 0x0345, 0x093C, 0x094D]])
    bases = ~c"aeiouAEOUnzcsyиеЕΑΩ" ++ [0x1100, 0x1161, 0x11A8, 0xAC00, 0x0E01, 0x05D0, 0x0915]

    mixes =
      for _ <- 1..20_000 do
        for _ <- 1..:rand.uniform(6), into: "" do
          <<(if :rand.uniform(2) == 1, do: Enum.random(bases), else: Enum.random(marks))::utf8>>
        end
      end

    texts = singles ++ mixes
    script = """
    import sys, unicodedata
    ts = sys.stdin.buffer.read().decode("utf-8").split("\\x00")[:-1]
    out = []
    for t in ts:
        info = ""
        if len(t) == 1:
            info = "%d;%s" % (unicodedata.combining(t), unicodedata.decomposition(t))
        out.append(unicodedata.normalize("NFC", t) + "\\x00" + unicodedata.normalize("NFKC", t) + "\\x00" + info)
    sys.stdout.buffer.write((unicodedata.unidata_version + "\\x02" + "\\x01".join(out)).encode("utf-8"))
    """

    [pyver, body] = py!(script, [], Enum.map_join(texts, &(&1 <> <<0>>))) |> String.split(<<2>>)
    got = String.split(body, <<1>>)
    assert length(got) == length(texts)
    otp = :unicode_util.spec_version() |> Tuple.to_list() |> Enum.join(".")

    # The algorithm is what is tested; the character data are OTP's. When
    # Python's Unicode version differs from OTP's, code points whose data
    # differ between the two versions (new characters, new decompositions)
    # are set aside and counted: there the normal forms differ by data, not
    # by algorithm. With equal versions nothing is set aside.
    same_version = String.starts_with?(pyver <> ".0", otp) or pyver == otp

    drift? = fn t, info ->
      case {String.to_charlist(t), info} do
        {[cp], info} when info != "" ->
          [ccc, dec] = String.split(info, ";", parts: 2)
          %{ccc: occ, canon: canon, compat: compat} = :unicode_util.lookup(cp)
          String.to_integer(ccc) != occ or (dec == "") != (canon == [] and compat == [])

        _ ->
          false
      end
    end

    {bad, drifted} =
      texts
      |> Enum.zip(got)
      |> Enum.reject(fn {t, line} -> [c, k, _] = String.split(line, <<0>>); Unicode.nfc(t) == c and Unicode.nfkc(t) == k end)
      |> Enum.split_with(fn {t, line} -> same_version or not drift?.(t, List.last(String.split(line, <<0>>))) end)

    if drifted != [], do: IO.puts("\n  Unicode #{otp} (OTP) vs #{pyver} (Python): #{length(drifted)} code points differ by data, set aside")
    assert bad == [], "#{length(bad)} differences, e.g. #{inspect(Enum.take(bad, 3))}"
  end
end
