;;; vapor-mode.el --- Almizan and Alembic, with vapor's language server -*- lexical-binding: t; -*-

;; Usage:
;;   (add-to-list 'load-path "/path/to/vapor/editors/emacs")
;;   (require 'vapor-mode)
;;   M-x eglot   in a .wzn or .nbq buffer (or (add-hook 'almizan-mode-hook #'eglot-ensure))

;;; Code:

(defgroup vapor nil "vapor's languages." :group 'languages)

(defcustom vapor-program '("vapor" "lsp")
  "The language server command."
  :type '(repeat string) :group 'vapor)

(defconst almizan-font-lock-keywords
  `((,(regexp-opt '("claim" "import" "دعوى" "استيراد") 'symbols) . font-lock-keyword-face)
    (,(regexp-opt '("root" "wazn" "inputs" "field" "box" "step" "init" "invariant" "proof" "body" "graph"
                    "جذر" "وزن" "مدخلات" "حقل" "صندوق" "خطوة" "بداية" "ثابت" "برهان" "تنفيذ" "مخطط") 'symbols) . font-lock-builtin-face)
    (,(regexp-opt '("fail" "maful" "burhan" "فاعل" "مفعول") 'symbols) . font-lock-type-face)
    (,(regexp-opt '("q" "int" "f64" "f32" "bool" "نسبي" "صحيح" "منطقي") 'symbols) . font-lock-type-face)
    (,(regexp-opt '("conserved" "nonneg" "pos" "identity" "bounded" "identifiable" "adjustment" "separated"
                    "محفوظ" "موجب" "متطابقة" "محدود" "معرف" "تعديل" "منفصل") 'symbols) . font-lock-preprocessor-face)
    (,(regexp-opt '("do" "independent" "افعل" "مستقل") 'symbols) . font-lock-keyword-face)
    (,(regexp-opt '("H-s-b" "H-f-Z" "n-q-l" "k-t-b" "s-b-b" "ح-س-ب" "ح-ف-ظ" "ن-ق-ل" "ك-ت-ب" "س-ب-ب")) . font-lock-constant-face))
  "Highlighting for Almizan in both scripts.")

;;;###autoload
(define-derived-mode almizan-mode lisp-data-mode "Almizan"
  "Major mode for Almizan (.wzn), in Latin or Arabic script."
  (setq-local comment-start "; ")
  (setq-local bidi-paragraph-direction nil)
  (setq-local font-lock-defaults '(almizan-font-lock-keywords)))

(defconst alembic-font-lock-keywords
  `((,(regexp-opt '("if" "then" "else" "let" "in" "for" "match" "fn") 'symbols) . font-lock-keyword-face)
    (,(regexp-opt '("space" "minimize" "maximize" "claim" "find" "holdout" "describe" "neighbor" "violation" "measured") 'symbols) . font-lock-builtin-face)
    ("^\\s-*\\([A-Za-z_][A-Za-z0-9_]*\\)\\s-*(.*)\\s-*=" 1 font-lock-function-name-face)))

;;;###autoload
(define-derived-mode alembic-mode prog-mode "Alembic"
  "Major mode for Alembic (.nbq), vapor's problem language."
  (setq-local comment-start "# ")
  (modify-syntax-entry ?# "<" alembic-mode-syntax-table)
  (modify-syntax-entry ?\n ">" alembic-mode-syntax-table)
  (setq-local font-lock-defaults '(alembic-font-lock-keywords)))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.wzn\\'" . almizan-mode))
;;;###autoload
(add-to-list 'auto-mode-alist '("\\.nbq\\'" . alembic-mode))

(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs `((almizan-mode alembic-mode) . ,vapor-program)))

(defun almizan-to-arabic () "Show this program in Arabic script (same tree, same hash)." (interactive) (almizan--project "vapor.almizan.toArabic"))
(defun almizan-to-latin () "Show this program in Latin script (same tree, same hash)." (interactive) (almizan--project "vapor.almizan.toLatin"))
(defun almizan--project (cmd)
  (eglot-execute-command (eglot--current-server-or-lose) cmd (vector (eglot--path-to-uri (buffer-file-name)))))

(provide 'vapor-mode)
;;; vapor-mode.el ends here
