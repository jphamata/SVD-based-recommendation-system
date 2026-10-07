;;; vapor-mode.el --- Al-Mizān and Alembic, with vapor's language server -*- lexical-binding: t; -*-

;; Usage:
;;   (add-to-list 'load-path "/path/to/vapor/editors/emacs")
;;   (require 'vapor-mode)
;;   M-x eglot   in a .wzn or .alb buffer (or (add-hook 'mizan-mode-hook #'eglot-ensure))

;;; Code:

(defgroup vapor nil "vapor's languages." :group 'languages)

(defcustom vapor-program '("vapor" "lsp")
  "The language server command."
  :type '(repeat string) :group 'vapor)

(defconst mizan-font-lock-keywords
  `((,(regexp-opt '("claim" "import" "دعوى" "استيراد") 'symbols) . font-lock-keyword-face)
    (,(regexp-opt '("root" "wazn" "inputs" "field" "box" "step" "init" "invariant" "proof" "body"
                    "جذر" "وزن" "مدخلات" "حقل" "صندوق" "خطوة" "بداية" "ثابت" "برهان" "تنفيذ") 'symbols) . font-lock-builtin-face)
    (,(regexp-opt '("fail" "maful" "burhan" "فاعل" "مفعول") 'symbols) . font-lock-type-face)
    (,(regexp-opt '("q" "int" "f64" "f32" "bool" "نسبي" "صحيح" "منطقي") 'symbols) . font-lock-type-face)
    (,(regexp-opt '("conserved" "nonneg" "pos" "identity" "bounded" "محفوظ" "موجب" "متطابقة" "محدود") 'symbols) . font-lock-preprocessor-face)
    (,(regexp-opt '("H-s-b" "H-f-Z" "n-q-l" "k-t-b" "ح-س-ب" "ح-ف-ظ" "ن-ق-ل" "ك-ت-ب")) . font-lock-constant-face))
  "Highlighting for Al-Mizān in both scripts.")

;;;###autoload
(define-derived-mode mizan-mode lisp-data-mode "Mizān"
  "Major mode for Al-Mizān (.wzn), in Latin or Arabic script."
  (setq-local comment-start "; ")
  (setq-local bidi-paragraph-direction nil)
  (setq-local font-lock-defaults '(mizan-font-lock-keywords)))

(defconst alembic-font-lock-keywords
  `((,(regexp-opt '("if" "then" "else" "let" "in" "for" "match" "fn") 'symbols) . font-lock-keyword-face)
    (,(regexp-opt '("space" "minimize" "maximize" "claim" "find" "holdout" "describe" "neighbor" "violation" "measured") 'symbols) . font-lock-builtin-face)
    ("^\\s-*\\([A-Za-z_][A-Za-z0-9_]*\\)\\s-*(.*)\\s-*=" 1 font-lock-function-name-face)))

;;;###autoload
(define-derived-mode alembic-mode prog-mode "Alembic"
  "Major mode for Alembic (.alb), vapor's problem language."
  (setq-local comment-start "# ")
  (modify-syntax-entry ?# "<" alembic-mode-syntax-table)
  (modify-syntax-entry ?\n ">" alembic-mode-syntax-table)
  (setq-local font-lock-defaults '(alembic-font-lock-keywords)))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.wzn\\'" . mizan-mode))
;;;###autoload
(add-to-list 'auto-mode-alist '("\\.alb\\'" . alembic-mode))

(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs `((mizan-mode alembic-mode) . ,vapor-program)))

(defun mizan-to-arabic () "Show this program in Arabic script (same tree, same hash)." (interactive) (mizan--project "vapor.mizan.toArabic"))
(defun mizan-to-latin () "Show this program in Latin script (same tree, same hash)." (interactive) (mizan--project "vapor.mizan.toLatin"))
(defun mizan--project (cmd)
  (eglot-execute-command (eglot--current-server-or-lose) cmd (vector (eglot--path-to-uri (buffer-file-name)))))

(provide 'vapor-mode)
;;; vapor-mode.el ends here
