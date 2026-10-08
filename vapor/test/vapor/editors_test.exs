defmodule Vapor.EditorsTest do
  @moduledoc "The editor packages, checked where they can be without the editors themselves."
  use ExUnit.Case, async: true
  @root Path.expand("../..", __DIR__)

  @tag :node
  test "VS Code: manifests parse; every TextMate pattern compiles; both scripts and Alembic are tokenized" do
    {out, 0} = System.cmd(System.find_executable("node") || flunk("node"), [Path.join(@root, "test/js/editors_check.mjs"), @root])
    {:ok, r} = Vapor.JSON.decode(out)
    assert r["files"] == 5 and r["patterns"] > 15
    for key <- ["keyword.declaration.almizan", "keyword.other.clause.almizan", "entity.name.tag.root.almizan", "storage.modifier.wazn.almizan"] do
      assert key in r["hits"]["latin"] and key in r["hits"]["arabic"], key
    end
    assert "entity.name.function.alembic" in r["hits"]["alembic"]
  end

  test "Vim: the filetype is detected and Arabic keywords are highlighted as keywords" do
    vim = System.find_executable("vim")
    if vim do
      out = Path.join(System.tmp_dir!(), "vimout-#{System.unique_integer([:positive])}.txt")
      args = ["-Es", "-u", "NONE", "-N", "--cmd", "set encoding=utf-8", "-c", "set rtp+=#{@root}/editors/nvim", "-c", "syntax on", "-c", "filetype on",
              "-c", "runtime ftdetect/vapor.vim", "-c", "edit #{@root}/priv/almizan/oscillator-ar.wzn", "-c", "redir! > #{out}",
              "-c", "echo &filetype synIDattr(synID(2, 2, 1), 'name') synIDattr(synID(3, 9, 1), 'name')", "-c", "redir END", "-c", "qa!"]
      {_, 0} = System.cmd(vim, args)
      assert File.read!(out) |> String.split() == ["almizan", "almizanDecl", "almizanClause"]
      File.rm(out)
    end
  end

  test "Emacs and Neovim packages are present and name the server command" do
    assert File.read!(Path.join(@root, "editors/emacs/vapor-mode.el")) =~ ~s("vapor" "lsp")
    assert File.read!(Path.join(@root, "editors/nvim/lua/vapor/init.lua")) =~ ~s({ "vapor", "lsp" })
  end
end
