" Al-Mizān syntax for Vim and Neovim (both scripts)
scriptencoding utf-8
if exists("b:current_syntax") | finish | endif
syntax match mizanComment ";.*$"
syntax keyword mizanDecl claim import دعوى استيراد
syntax keyword mizanClause root wazn inputs field box step init invariant proof body جذر وزن مدخلات حقل صندوق خطوة بداية ثابت برهان تنفيذ
syntax keyword mizanWazn fail maful burhan فاعل مفعول
syntax keyword mizanType q int f64 f32 bool نسبي صحيح منطقي
syntax keyword mizanProof conserved nonneg pos identity bounded محفوظ موجب متطابقة محدود
syntax keyword mizanLogic and or not if true false أو ليس إذا صواب خطأ
syntax match mizanRoot "\(H-s-b\|H-f-Z\|n-q-l\|k-t-b\|ح-س-ب\|ح-ف-ظ\|ن-ق-ل\|ك-ت-ب\)"
syntax match mizanNumber "\v<-?\d+([./]\d+)?>"
syntax match mizanNumber "[٠-٩]\+\([/.][٠-٩]\+\)\?"
highlight default link mizanComment Comment
highlight default link mizanDecl Keyword
highlight default link mizanClause Statement
highlight default link mizanWazn StorageClass
highlight default link mizanType Type
highlight default link mizanProof Special
highlight default link mizanLogic Operator
highlight default link mizanRoot Constant
highlight default link mizanNumber Number
setlocal commentstring=;\ %s
let b:current_syntax = "mizan"
