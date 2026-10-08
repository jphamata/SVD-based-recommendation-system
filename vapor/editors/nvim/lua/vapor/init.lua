-- vapor for Neovim (0.10+): filetypes and the language server, with the built-in client.
--   require("vapor").setup({ cmd = { "vapor", "lsp" } })
local M = {}

function M.setup(opts)
  opts = opts or {}
  local cmd = opts.cmd or { "vapor", "lsp" }
  vim.filetype.add({ extension = { wzn = "almizan", nbq = "alembic" } })
  vim.api.nvim_create_autocmd("FileType", {
    pattern = { "almizan", "alembic" },
    callback = function(args)
      vim.bo[args.buf].commentstring = (vim.bo[args.buf].filetype == "almizan") and "; %s" or "# %s"
      vim.lsp.start({
        name = "vapor",
        cmd = cmd,
        root_dir = vim.fs.root(args.buf, { ".git", "mix.exs" }) or vim.fn.getcwd(),
        -- the verdicts beside the claims (the server's inlay hints)
        on_attach = function(_, buf)
          if vim.lsp.inlay_hint then vim.lsp.inlay_hint.enable(true, { bufnr = buf }) end
        end,
      })
    end,
  })
  -- the projection, as a command: the same program in the other script
  for _, which in ipairs({ "Arabic", "Latin" }) do
    vim.api.nvim_create_user_command("Almizan" .. which, function()
      vim.lsp.buf.execute_command({ command = "vapor.almizan.to" .. which, arguments = { vim.uri_from_bufnr(0) } })
    end, {})
  end
end

return M
