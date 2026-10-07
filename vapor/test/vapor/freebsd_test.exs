defmodule Vapor.FreeBSDTest do
  @moduledoc """
  The FreeBSD worker, as far as a Linux host can check it: it
  cross-compiles for x86-64 and AArch64 (zig, FreeBSD's libc headers), the
  result is a FreeBSD executable (its ABI note, FreeBSD's run-time linker), and it imports what the port says it
  uses — Capsicum (`cap_enter`, `cap_rights_limit`, `__cap_rights_init`),
  the `_umtx_op` futex, `openat` through the kept directory descriptor and
  `mprotect` for W^X code pages — and none of Darwin's JIT calls. (The
  Linux seccomp filter is installed by raw system calls, invisible in an
  import table: its absence is by conditional compilation, not checked
  here.) Running it needs a FreeBSD host (docs/TODO.md).
  """
  use ExUnit.Case, async: false

  @moduletag :zig
  @moduletag timeout: 600_000

  for target <- ["x86_64-freebsd", "aarch64-freebsd"] do
    test "#{target}: builds, is a FreeBSD ELF, imports Capsicum and _umtx_op" do
      out = Path.join(System.tmp_dir!(), "vapor-fbsd-#{unquote(target)}-#{System.unique_integer([:positive])}")
      native = Path.expand("../../native", __DIR__)
      {log, code} = System.cmd("zig", ["build", "-Dtarget=#{unquote(target)}", "-Doptimize=ReleaseFast", "--prefix", out], cd: native, stderr_to_stdout: true)
      assert code == 0, log
      bin = File.read!(Path.join([out, "bin", "vapor-worker"]))
      # a FreeBSD executable: its ABI note (OS/ABI byte stays SYSV) and FreeBSD's run-time linker
      assert <<0x7F, "ELF", 2, 1, 1, _::binary>> = bin
      assert :binary.match(bin, "FreeBSD") != :nomatch and :binary.match(bin, "/libexec/ld-elf.so.1") != :nomatch
      for sym <- ~w(cap_enter cap_rights_limit __cap_rights_init _umtx_op openat mprotect), do: assert(:binary.match(bin, sym) != :nomatch, sym)
      for sym <- ~w(pthread_jit_write_protect_np __ulock_wait), do: assert(:binary.match(bin, sym) == :nomatch, sym)
    end
  end
end
