" Almizan syntax for Vim and Neovim (both scripts)
scriptencoding utf-8
if exists("b:current_syntax") | finish | endif
syntax match almizanComment ";.*$"
syntax keyword almizanDecl claim import دعوى استيراد
syntax keyword almizanClause root wazn inputs field box step init invariant proof body graph جذر وزن مدخلات حقل صندوق خطوة بداية ثابت برهان تنفيذ مخطط
syntax keyword almizanWazn fail maful burhan فاعل مفعول
syntax keyword almizanType q int f64 f32 bool نسبي صحيح منطقي
syntax keyword almizanProof conserved nonneg pos identity bounded identifiable adjustment separated محفوظ موجب متطابقة محدود معرف تعديل منفصل
syntax keyword almizanLogic and or not if true false do independent أو ليس إذا صواب خطأ افعل مستقل
syntax match almizanRoot "\(H-s-b\|H-f-Z\|n-q-l\|k-t-b\|ح-س-ب\|ح-ف-ظ\|ن-ق-ل\|ك-ت-ب\|s-b-b\|س-ب-ب\)"
syntax match almizanEdge "<\?->"
syntax match almizanNumber "\v<-?\d+([./]\d+)?>"
syntax match almizanNumber "[٠-٩]\+\([/.][٠-٩]\+\)\?"
highlight default link almizanComment Comment
highlight default link almizanDecl Keyword
highlight default link almizanClause Statement
highlight default link almizanWazn StorageClass
highlight default link almizanType Type
highlight default link almizanProof Special
highlight default link almizanLogic Operator
highlight default link almizanRoot Constant
highlight default link almizanNumber Number
highlight default link almizanEdge Operator
setlocal commentstring=;\ %s
let b:current_syntax = "almizan"
