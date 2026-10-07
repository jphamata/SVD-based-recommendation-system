defmodule Vapor.Rebis.GCM do
  @moduledoc """
  AES (FIPS-197) and AES-GCM (NIST SP 800-38D) written from the field
  arithmetic of `Vapor.Rebis.Field`: the S-box is the GF(2⁸) inverse plus an
  affine map, MixColumns is multiplication by a fixed polynomial over
  GF(2⁸), and the tag is GHASH in GF(2¹²⁸). Nothing is a table copied from
  a standard; everything is checked against OTP's `:crypto` (OpenSSL) in
  the tests.

  This is the **reference** an accelerated kernel answers to (the
  carry-less multiply as an instruction in the five emitters is the next
  step, docs/TODO.md) — and the algebra behind the round's claim about
  "esquecimento" is stated plainly: destroying the key destroys the
  ability to decrypt, which is a property of AES, not of any data
  structure around it.
  """
  import Bitwise
  alias Vapor.Rebis.Field

  @sbox (for x <- 0..255, do: Field.sbox(x)) |> List.to_tuple()

  @doc "The S-box as a tuple (computed at compile time from the field)."
  def sbox_table, do: @sbox

  # ------------------------------------------------------------ the cipher

  @doc "Expand a 16-, 24- or 32-byte key into round keys (list of 16-byte binaries)."
  def expand(key) when byte_size(key) in [16, 24, 32] do
    nk = div(byte_size(key), 4)
    nr = nk + 6
    words = for <<w::32 <- key>>, do: w

    words =
      Enum.reduce(nk..(4 * (nr + 1) - 1)//1, words, fn i, ws ->
        prev = List.last(ws)
        t =
          cond do
            rem(i, nk) == 0 -> bxor(sub_word(rot_word(prev)), rcon(div(i, nk)) <<< 24)
            nk > 6 and rem(i, nk) == 4 -> sub_word(prev)
            true -> prev
          end

        ws ++ [bxor(Enum.at(ws, i - nk), t)]
      end)

    words |> Enum.chunk_every(4) |> Enum.map(fn ws -> for w <- ws, into: <<>>, do: <<w::32>> end)
  end

  defp rot_word(w), do: (w <<< 8 ||| w >>> 24) &&& 0xFFFF_FFFF
  defp sub_word(w), do: for(<<b <- <<w::32>> >>, into: <<>>, do: <<elem(@sbox, b)>>) |> :binary.decode_unsigned()
  defp rcon(i), do: Field.pow(2, i - 1, Field.aes_poly())

  @doc "Encrypt one 16-byte block."
  def encrypt_block(round_keys, <<_::128>> = block) do
    [k0 | rest] = round_keys
    {mid, [last]} = Enum.split(rest, -1)
    s = xor16(block, k0)
    s = Enum.reduce(mid, s, fn k, s -> s |> sub_bytes() |> shift_rows() |> mix_columns() |> xor16(k) end)
    s |> sub_bytes() |> shift_rows() |> xor16(last)
  end

  defp xor16(<<a::128>>, <<b::128>>), do: <<bxor(a, b)::128>>
  defp sub_bytes(s), do: for(<<b <- s>>, into: <<>>, do: <<elem(@sbox, b)>>)

  # state is column-major: byte 4c + r is row r, column c
  defp shift_rows(s) do
    t = :binary.bin_to_list(s) |> List.to_tuple()
    for c <- 0..3, r <- 0..3, into: <<>>, do: <<elem(t, 4 * rem(c + r, 4) + r)>>
  end

  defp mix_columns(s) do
    for <<a0, a1, a2, a3 <- s>>, into: <<>> do
      <<bxor(bxor(m2(a0), m3(a1)), bxor(a2, a3)), bxor(bxor(a0, m2(a1)), bxor(m3(a2), a3)),
        bxor(bxor(a0, a1), bxor(m2(a2), m3(a3))), bxor(bxor(m3(a0), a1), bxor(a2, m2(a3)))>>
    end
  end

  defp m2(b), do: Field.mul(b, 2, Field.aes_poly())
  defp m3(b), do: bxor(m2(b), b)

  # ---------------------------------------------------------------- GCM

  @doc """
  AES-GCM encryption: `{ciphertext, tag}` (16-byte tag). `iv` of 12 bytes
  (the recommended size) or any other length (then GHASH-derived `J₀`).
  """
  def encrypt(key, iv, plaintext, aad \\ "") do
    rk = expand(key)
    <<h::128>> = encrypt_block(rk, <<0::128>>)
    j0 = j0(h, iv)
    ct = ctr(rk, inc32(j0), plaintext)
    {ct, tag(rk, h, j0, aad, ct)}
  end

  @doc "AES-GCM decryption: `{:ok, plaintext}` or `:error` when the tag does not authenticate."
  def decrypt(key, iv, ciphertext, aad, tag) do
    rk = expand(key)
    <<h::128>> = encrypt_block(rk, <<0::128>>)
    j0 = j0(h, iv)
    if constant_eq(tag(rk, h, j0, aad, ciphertext), tag), do: {:ok, ctr(rk, inc32(j0), ciphertext)}, else: :error
  end

  defp j0(_h, <<iv::binary-size(12)>>), do: iv <> <<1::32>>

  defp j0(h, iv) do
    s = pad16(iv) <> <<0::64, bit_size(iv)::64>>
    <<Field.ghash(h, s)::128>>
  end

  defp tag(rk, h, j0, aad, ct) do
    s = pad16(aad) <> pad16(ct) <> <<bit_size(aad)::64, bit_size(ct)::64>>
    <<t::128>> = encrypt_block(rk, j0)
    <<bxor(Field.ghash(h, s), t)::128>>
  end

  defp ctr(rk, cb, data) do
    {out, _} =
      data
      |> chunks16()
      |> Enum.map_reduce(cb, fn chunk, cb ->
        <<ks::binary-size(byte_size(chunk)), _::binary>> = encrypt_block(rk, cb)
        {:crypto.exor(chunk, ks), inc32(cb)}
      end)

    IO.iodata_to_binary(out)
  end

  defp chunks16(<<>>), do: []
  defp chunks16(<<c::binary-size(16), rest::binary>>), do: [c | chunks16(rest)]
  defp chunks16(c), do: [c]

  defp inc32(<<pre::96, ctr::32>>), do: <<pre::96, (ctr + 1) &&& 0xFFFF_FFFF::32>>

  defp pad16(b) do
    r = rem(byte_size(b), 16)
    if r == 0, do: b, else: b <> :binary.copy(<<0>>, 16 - r)
  end

  defp constant_eq(a, b) when byte_size(a) == byte_size(b),
    do: :crypto.exor(a, b) |> :binary.bin_to_list() |> Enum.reduce(0, &bor/2) == 0

  defp constant_eq(_, _), do: false
end
