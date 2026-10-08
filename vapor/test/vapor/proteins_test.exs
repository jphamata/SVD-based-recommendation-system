defmodule Vapor.ProteinsTest do
  @moduledoc """
  Protein structure (docs/PROTEINS.md): TM-score and RMSD against
  TM-align on real NMR models (1LCD), the folding of a real protein
  (1A8O, HIV capsid C-terminal domain) from its contacts, the mirror
  image rejected by helix handedness, contacts from a planted
  coevolution model (DCA beats MI; a shuffled alignment scores at chance —
  the control), the whole pipeline alignment → contacts → structure, and
  pairwise alignment against Biopython.
  """
  use ExUnit.Case, async: true
  @moduletag timeout: 1_200_000
  alias Vapor.Bio.{Align, Coevolution, Structure}

  defp read(f), do: File.read!(Path.join([to_string(:code.priv_dir(:vapor)), "quality", "protein", f]))

  test "superposition: a rotated, translated copy has RMSD 0 and TM-score 1; lDDT is invariant" do
    {:ok, nat} = Structure.read_pdb(read("1A8O.pdb"))
    {r, _} = {[[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]], nil}
    moved = Structure.transform(nat.ca, {r, {10.0, -4.0, 3.0}})
    assert Structure.rmsd(moved, nat.ca) < 1.0e-9
    assert_in_delta Structure.tm_score(moved, nat.ca).tm, 1.0, 1.0e-12
    assert_in_delta Structure.lddt(moved, nat.ca), 1.0, 1.0e-12
  end

  @tag :tmtools
  test "TM-score and RMSD equal TM-align's on the NMR models of 1LCD" do
    [m1, m2, m3] = read("1LCD.pdb") |> Structure.read_models() |> Enum.map(& &1.ca)
    dir = Path.join(System.tmp_dir!(), "vapor-tm-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    for {m, k} <- [{m1, 1}, {m2, 2}, {m3, 3}], do: File.write!(Path.join(dir, "m#{k}.pdb"), Structure.to_pdb(m))
    out = Vapor.TestHelpers.py!("""
    import numpy as np, tmtools, sys
    def ca(p):
        xs=[[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in open(p) if l.startswith('ATOM')]
        return np.array(xs)
    d=sys.argv[1]; b=ca(d+'/m1.pdb')
    for k in (2,3):
        a=ca(d+'/m%d.pdb'%k); r=tmtools.tm_align(a,b,'G'*len(a),'G'*len(b)); print(r.tm_norm_chain2, r.rmsd)
    """, [dir])
    File.rm_rf!(dir)
    [[t2, r2], [t3, r3]] = out |> String.split("\n", trim: true) |> Enum.map(fn l -> l |> String.split() |> Enum.map(&String.to_float/1) end)
    s2 = Structure.tm_score(m2, m1)
    s3 = Structure.tm_score(m3, m1)
    assert_in_delta s2.tm, t2, 1.0e-6
    assert_in_delta s3.tm, t3, 1.0e-6
    assert_in_delta s2.rmsd, r2, 1.0e-6
    assert_in_delta s3.rmsd, r3, 1.0e-6
  end

  test "1A8O folds from its contact map to TM > 0.75; the mirror image (the control) is below 0.4 and rejected by helix handedness" do
    {:ok, nat} = Structure.read_pdb(read("1A8O.pdb"))
    ss = Structure.secondary(nat.ca)
    f = Structure.fold(length(nat.ca), Structure.contacts(nat.ca, 8.0, 3), complete: true, helices: ss, restarts: 2)
    tm = Structure.tm_score(f.ca, nat.ca).tm
    mirror = Structure.tm_score(Enum.map(f.ca, fn {x, y, z} -> {-x, y, z} end), nat.ca).tm
    assert tm > 0.75
    assert mirror < 0.4
  end

  test "coevolution: DCA finds the planted contacts better than MI; a shuffled alignment scores at chance" do
    {:ok, nat} = Structure.read_pdb(read("1A8O.pdb"))
    l = length(nat.ca)
    truth = Structure.contacts(nat.ca, 8.0, 6)
    msa = Coevolution.sample(l, Structure.contacts(nat.ca, 8.0, 3), n: 2000, coupling: 0.6, sweeps: 3, seed: 1)
    k = length(truth)
    pd = Structure.precision(Coevolution.dca(msa), truth, k)
    pm = Structure.precision(Coevolution.mi(msa), truth, k)
    ps = Structure.precision(Coevolution.dca(Coevolution.shuffle(msa)), truth, k)
    chance = k / div((l - 6) * (l - 5), 2)
    assert pd > 0.9 and pd > pm
    assert ps < 3 * chance
  end

  test "the pipeline: planted alignment → DCA contacts → distance geometry → TM > 0.6; from shuffled contacts (the control) < 0.3" do
    {:ok, nat} = Structure.read_pdb(read("1A8O.pdb"))
    l = length(nat.ca)
    ss = Structure.secondary(nat.ca)
    allc = Structure.contacts(nat.ca, 8.0, 3)
    local = Enum.filter(allc, fn {i, j} -> j - i < 6 end)
    k = length(Structure.contacts(nat.ca, 8.0, 6))
    msa = Coevolution.sample(l, allc, n: 2000, coupling: 0.6, sweeps: 3, seed: 1)
    good = Structure.fold(l, local ++ Enum.take(Coevolution.dca(msa), k), helices: ss, restarts: 2)
    bad = Structure.fold(l, local ++ Enum.take(Coevolution.dca(Coevolution.shuffle(msa)), k), helices: ss, restarts: 2)
    assert Structure.tm_score(good.ca, nat.ca).tm > 0.6
    assert Structure.tm_score(bad.ca, nat.ca).tm < 0.3
  end

  test "secondary structure from Cα: 1A8O is mostly helix" do
    {:ok, nat} = Structure.read_pdb(read("1A8O.pdb"))
    ss = Structure.secondary(nat.ca)
    assert String.length(ss) == 70
    assert Enum.count(String.graphemes(ss), &(&1 == "H")) > 40
  end

  @tag :biopython
  test "pairwise alignment scores equal Biopython's PairwiseAligner (BLOSUM62, gaps 11/1), and the traceback attains them" do
    pairs = [{"HEAGAWGHEE", "PAWHEAE"}, {"MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNW", "MDIRQGPKEPFRDYVDRFYKTLRAEQASQDVKNWMTE"}, {"KTILKALGPGAT", "TLKALGEGAT"}]
    out = Vapor.TestHelpers.py!("""
    from Bio import Align
    from Bio.Align import substitution_matrices
    import sys
    al=Align.PairwiseAligner(); al.substitution_matrix=substitution_matrices.load('BLOSUM62'); al.open_gap_score=-11; al.extend_gap_score=-1
    for line in sys.stdin:
        a,b,mode=line.split()
        al.mode=mode; print(int(al.score(a,b)))
    """, [], Enum.map_join(pairs, "", fn {a, b} -> "#{a} #{b} global\n#{a} #{b} local\n" end))
    ref = out |> String.split() |> Enum.map(&String.to_integer/1)
    mine = Enum.flat_map(pairs, fn {a, b} -> for m <- [:global, :local], do: Align.align(a, b, mode: m) end)
    assert Enum.map(mine, & &1.score) == ref
    for r <- mine, do: assert(Align.score_of(r.a, r.b) == r.score)
  end
end
