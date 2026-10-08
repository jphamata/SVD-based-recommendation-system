import Lean
/-!
# Lean → Elixir extraction

Reads the *elaborated* definitions of the proven functions — exactly the
terms the theorems quantify over — and prints them as Elixir. Supported
fragment: `Nat`/`Int`/`Bool` arithmetic and comparisons (with Lean's total
semantics: truncated `Nat` subtraction, `x / 0 = 0`, Euclidean `Int` `%`),
`if`/`decide`, `let`, lambdas, pairs, and the list combinators `all`, `any`,
`map`, `foldr`, `range`, `contains`, `length`, `cons`/`nil`. Calls to other
`Vapor` definitions are extracted transitively; partial applications are
eta-expanded. Anything outside the fragment is a hard error, never a silent
approximation.

Trusted computing base: this printer (~200 lines) and a six-line Elixir
prelude for Lean's total division/modulo. Both are differential-tested:
`Main.lean` evaluates every extracted function *in Lean* on generated inputs
and emits an ExUnit suite that the Elixir code must pass.
-/
open Lean Meta

namespace VaporExtract

structure St where
  pending : Array Name := #[]
  done : NameSet := {}
  helpers : NameSet := {}
  names : Std.HashMap FVarId String := {}
  counter : Nat := 0

abbrev M := StateRefT St MetaM

def snake (s : String) : String := Id.run do
  let mut out := ""
  let mut prev := false
  for c in s.toList do
    if c.isUpper then
      if prev then out := out.push '_'
      out := out.push c.toLower
      prev := false
    else
      out := out.push c
      prev := c.isLower || c.isDigit
  return out

def elixirName (n : Name) : String := snake n.getString!

def fresh (base : String) : M String := do
  let s ← get
  set { s with counter := s.counter + 1 }
  let clean := (snake base).map (fun c => if c.isAlphanum then c else '_')
  let clean := if clean.isEmpty || !clean.front.isLower then "v" ++ clean else clean
  return s!"{clean}_{s.counter}"

def helper (h : Name) : M Unit := modify fun s => { s with helpers := s.helpers.insert h }

def isVapor (n : Name) : Bool := (`Vapor).isPrefixOf n

def typeName (e : Expr) : String :=
  match e.getAppFn with
  | .const n _ => n.toString
  | _ => "?"

partial def arityOf (n : Name) : MetaM Nat := do
  let info ← getConstInfo n
  forallTelescope info.type fun xs _ => pure xs.size

mutual

partial def lam (e : Expr) : M String := do
  lambdaTelescope e fun xs body => do
    let mut ps := #[]
    for x in xs do
      let d ← x.fvarId!.getDecl
      let nm ← fresh d.userName.toString
      modify fun s => { s with names := s.names.insert x.fvarId! nm }
      ps := ps.push nm
    let b ← ex body
    return s!"fn {", ".intercalate ps.toList} -> {b} end"

partial def ex (e : Expr) : M String := do
  match e with
  | .mdata _ b => ex b
  | .lit (.natVal n) => return toString n
  | .fvar id =>
      match (← get).names[id]? with
      | some n => return n
      | none => throwError "extract: unbound variable {e}"
  | .lam .. => lam e
  | .letE n t v b _ => do
      let rhs ← ex v
      withLetDecl n t v fun x => do
        let nm ← fresh n.toString
        modify fun s => { s with names := s.names.insert x.fvarId! nm }
        let body ← ex (b.instantiate1 x)
        return s!"({nm} = {rhs}; {body})"
  | .const n _ =>
      if n == ``Bool.true then return "true"
      else if n == ``Bool.false then return "false"
      else if n == ``List.nil then return "[]"
      else if isVapor n then call n #[]
      else throwError "extract: unsupported constant {n}"
  | .app .. => app e
  | _ => throwError "extract: unsupported expression {e}"

partial def call (n : Name) (args : Array Expr) : M String := do
  let arity ← arityOf n
  unless (← get).done.contains n || (← get).pending.contains n do
    modify fun s => { s with pending := s.pending.push n }
  let as ← args.mapM ex
  if args.size ≥ arity then
    return s!"{elixirName n}({", ".intercalate as.toList})"
  else
    -- partial application: eta-expand
    let missing := (List.range (arity - args.size)).map (fun i => s!"eta_{i}")
    let all := as.toList ++ missing
    return s!"fn {", ".intercalate missing} -> {elixirName n}({", ".intercalate all}) end"

partial def bin (op : String) (args : Array Expr) : M String := do
  let a ← ex args[args.size - 2]!
  let b ← ex args[args.size - 1]!
  return s!"({a} {op} {b})"

partial def app (e : Expr) : M String := do
  let f := e.getAppFn
  let args := e.getAppArgs
  let last (k : Nat) : Expr := args[args.size - k]!
  match f with
  | .fvar _ => do
      let fs ← ex f
      let as ← args.mapM ex
      return s!"{fs}.({", ".intercalate as.toList})"
  | .const n _ =>
    match n with
    | ``OfNat.ofNat => ex args[1]!
    | ``Nat.cast | ``NatCast.natCast | ``Int.ofNat | ``IntCast.intCast => ex (last 1)
    | ``HAdd.hAdd => bin "+" args
    | ``HMul.hMul => bin "*" args
    | ``HSub.hSub =>
        if typeName args[0]! == "Nat" then do
          return s!"max({← ex (last 2)} - {← ex (last 1)}, 0)"
        else bin "-" args
    | ``HMod.hMod => do
        let h := if typeName args[0]! == "Nat" then `nat_mod else `int_emod
        helper h
        return s!"{h}({← ex (last 2)}, {← ex (last 1)})"
    | ``HDiv.hDiv => do
        unless typeName args[0]! == "Nat" do throwError "extract: only Nat division is supported"
        helper `nat_div
        return s!"nat_div({← ex (last 2)}, {← ex (last 1)})"
    | ``HPow.hPow => return s!"Integer.pow({← ex (last 2)}, {← ex (last 1)})"
    | ``Neg.neg => return s!"(-{← ex (last 1)})"
    | ``LT.lt => bin "<" args
    | ``LE.le => bin "<=" args
    | ``GT.gt => bin ">" args
    | ``GE.ge => bin ">=" args
    | ``Eq => bin "==" args
    | ``Ne => bin "!=" args
    | ``BEq.beq => bin "==" args
    | ``bne => bin "!=" args
    | ``And | ``and => bin "and" args
    | ``Or | ``or => bin "or" args
    | ``Not | ``not => return s!"(not {← ex (last 1)})"
    | ``Decidable.decide => ex args[0]!
    | ``ite => return s!"(if {← ex args[1]!}, do: {← ex args[3]!}, else: {← ex args[4]!})"
    | ``cond => return s!"(if {← ex args[1]!}, do: {← ex args[2]!}, else: {← ex args[3]!})"
    | ``Prod.mk => return s!"\{{← ex (last 2)}, {← ex (last 1)}}"
    | ``Prod.fst => return s!"elem({← ex (last 1)}, 0)"
    | ``Prod.snd => return s!"elem({← ex (last 1)}, 1)"
    | ``List.cons => return s!"[{← ex (last 2)} | {← ex (last 1)}]"
    | ``List.nil => return "[]"
    | ``List.all => return s!"Enum.all?({← ex (last 2)}, {← ex (last 1)})"
    | ``List.any => return s!"Enum.any?({← ex (last 2)}, {← ex (last 1)})"
    | ``List.map => return s!"Enum.map({← ex (last 1)}, {← ex (last 2)})"
    | ``List.foldr => return s!"List.foldr({← ex (last 1)}, {← ex (last 2)}, {← ex (last 3)})"
    | ``List.length => return s!"length({← ex (last 1)})"
    | ``List.contains | ``List.elem => do
        -- `List.contains l a` and `List.elem a l`
        if n == ``List.contains then return s!"Enum.member?({← ex (last 2)}, {← ex (last 1)})"
        else return s!"Enum.member?({← ex (last 1)}, {← ex (last 2)})"
    | ``List.range => do helper `range_list; return s!"range_list({← ex (last 1)})"
    | _ =>
        if isVapor n then call n args
        else throwError "extract: unsupported function {n} in {e}"
  | _ => throwError "extract: unsupported application head {f}"

end

/-- Extract one definition as an Elixir `def`. -/
def defn (n : Name) (doc : String) : M String := do
  let info ← getConstInfo n
  let some v := info.value? | throwError "extract: {n} has no definitional value"
  lambdaTelescope v fun xs body => do
    let mut ps := #[]
    for x in xs do
      let d ← x.fvarId!.getDecl
      let nm ← fresh d.userName.toString
      modify fun s => { s with names := s.names.insert x.fvarId! nm }
      ps := ps.push nm
    let b ← ex body
    let docline := if doc.isEmpty then s!"  @doc false\n" else s!"  @doc \"\"\"\n  {doc}\n\n  Extracted from `{n}`.\n  \"\"\"\n"
    return s!"{docline}  def {elixirName n}({", ".intercalate ps.toList}) do\n    {b}\n  end\n"

def prelude (h : Name) : String :=
  match h.toString with
  | "nat_mod" => "  # Lean: n % 0 = n\n  defp nat_mod(a, 0), do: a\n  defp nat_mod(a, b), do: rem(a, b)\n"
  | "nat_div" => "  # Lean: n / 0 = 0\n  defp nat_div(_a, 0), do: 0\n  defp nat_div(a, b), do: div(a, b)\n"
  | "int_emod" => "  # Lean: Int.emod (Euclidean), a % 0 = a\n  defp int_emod(a, 0), do: a\n  defp int_emod(a, b), do: Integer.mod(a, abs(b))\n"
  | "range_list" => "  defp range_list(n), do: Enum.to_list(0..(n - 1)//1)\n"
  | _ => ""

/-- Extract the given roots (with their docs) and everything they call. -/
def extractAll (roots : List (Name × String)) : M String := do
  modify fun s => { s with pending := roots.toArray.map (·.1) }
  let docs : Std.HashMap Name String := roots.foldl (fun m (n, d) => m.insert n d) {}
  let mut out := #[]
  repeat
    let s ← get
    if s.pending.isEmpty then break
    let n := s.pending[0]!
    set { s with pending := s.pending.eraseIdxIfInBounds 0, done := s.done.insert n }
    out := out.push (← defn n (docs.getD n ""))
  let helpers := (← get).helpers.toList.map (prelude ·)
  return "\n".intercalate out.toList ++ "\n" ++ "".intercalate helpers

def run (env : Environment) (roots : List (Name × String)) : IO String := do
  let ctx : Core.Context := { fileName := "<vapor-extract>", fileMap := default, maxHeartbeats := 0 }
  let st : Core.State := { env }
  let (s, _) ← ((extractAll roots).run' {} |>.run' {} {}).toIO ctx st
  return s

end VaporExtract
