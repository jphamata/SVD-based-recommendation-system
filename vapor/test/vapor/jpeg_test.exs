defmodule Vapor.JPEGTest do
  @moduledoc """
  The JPEG decoder against libjpeg-turbo (through Pillow): the same pixels,
  bit for bit — committed fixtures with Pillow's pixel digests (always), and
  every layout of `test/python/jpeg_fixtures.py` (real photographs,
  4:4:4/4:2:2/4:2:0/4:4:0/4:1:1, progressive with successive approximation,
  restart markers, optimised tables, Adobe RGB, odd sizes) when Python with
  Pillow is present.
  """
  use ExUnit.Case, async: true
  alias Vapor.Docs.JPEG
  import Vapor.TestHelpers

  defp fx(name), do: Path.expand("../fixtures/docs/#{name}", __DIR__)
  defp digest(bin), do: Base.encode16(:crypto.hash(:sha256, bin), case: :lower)

  test "committed fixtures: baseline and progressive decode to Pillow's pixels (SHA-256)" do
    assert {:ok, %{width: 64, height: 48, mode: "RGB", progressive: false} = a} = JPEG.decode(File.read!(fx("scene.jpg")))
    assert digest(a.pixels) == "ef114fad3707daa5ec37ab16a550c8bcf0a4fb2d9fd8dec701fb04372497791f"
    assert {:ok, %{progressive: true, sampling: [{2, 2}, {1, 1}, {1, 1}]} = b} = JPEG.decode(File.read!(fx("scene_prog.jpg")))
    assert digest(b.pixels) == "91bd3506eeee1e6324e504f2f07209849d5f70bf6a47b38fee0394fc41eb5ae7"
  end

  test "what is not decoded is refused with a reason, never garbage" do
    assert {:error, "not a JPEG" <> _} = JPEG.decode("GIF89a")
    # SOF9 (arithmetic coding), SOF3 (lossless), a 12-bit frame
    soi = <<0xFF, 0xD8>>
    sof = fn m, p -> <<0xFF, m, 11::16, p, 8::16, 8::16, 1, 1, 0x11, 0>> end
    assert {:error, "arithmetic" <> _} = JPEG.decode(soi <> sof.(0xC9, 8))
    assert {:error, "lossless" <> _} = JPEG.decode(soi <> sof.(0xC3, 8))
    assert {:error, "12-bit" <> _} = JPEG.decode(soi <> sof.(0xC1, 12))
    assert {:error, "no scan"} = JPEG.decode(soi <> sof.(0xC0, 8) <> <<0xFF, 0xD9>>)
    # a file cut inside its scan still yields what was decoded (as libjpeg)
    full = File.read!(fx("scene.jpg"))
    assert {:ok, %{width: 64}} = JPEG.decode(binary_part(full, 0, byte_size(full) - 200))
  end

  @tag :python
  @tag timeout: 600_000
  test "every layout = Pillow (libjpeg-turbo), pixel for pixel" do
    if python?(["PIL", "numpy", "sklearn"]) do
      dir = Path.join(System.tmp_dir!(), "vapor-jpeg-#{System.unique_integer([:positive])}")
      py!(File.read!(Path.expand("../python/jpeg_fixtures.py", __DIR__)), [dir])
      {:ok, idx} = Vapor.JSON.decode(File.read!(Path.join(dir, "index.json")))
      assert map_size(idx) >= 33

      for {name, [w, h, _mode]} <- idx do
        assert {:ok, img} = JPEG.decode(File.read!(Path.join(dir, name <> ".jpg"))), name
        assert {img.width, img.height} == {w, h}, name
        assert img.pixels == File.read!(Path.join(dir, name <> ".raw")), "#{name}: pixels differ from Pillow's"
      end

      for f <- Path.wildcard(Path.join(dir, "refuse_*.jpg")), do: assert({:error, _} = JPEG.decode(File.read!(f)))
      File.rm_rf!(dir)
    end
  end
end
