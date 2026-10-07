" Alembic syntax for Vim and Neovim
scriptencoding utf-8
if exists("b:current_syntax") | finish | endif
syntax match alembicComment "#.*$"
syntax region alembicString start=+"+ skip=+\\.+ end=+"+
syntax keyword alembicKeyword if then else let in for match fn
syntax keyword alembicSpace space minimize maximize claim find holdout describe neighbor violation measured init moves play winner player
syntax keyword alembicConst true false nil inf pi
syntax match alembicNumber "\v<\d+(\.\d+)?([eE][-+]?\d+)?>"
syntax match alembicDef "^\s*\h\w*\ze\s*(.*)\s*="
highlight default link alembicComment Comment
highlight default link alembicString String
highlight default link alembicKeyword Keyword
highlight default link alembicSpace Statement
highlight default link alembicConst Constant
highlight default link alembicNumber Number
highlight default link alembicDef Function
setlocal commentstring=#\ %s
let b:current_syntax = "alembic"
