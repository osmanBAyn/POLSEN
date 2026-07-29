"""
retro.py -- rule-based polymer retrosynthesis.

WHY THIS EXISTS
---------------
A from-scratch seq2seq (T5) trained on synthetic data does not generalise to the
novel, often unusual polymers a generative GA produces (distribution shift). But
polymers -- unlike arbitrary molecules -- are made by a SMALL set of well-defined
polymerisation reactions. So the reliable approach is TEMPLATE DISCONNECTION:
recognise the characteristic backbone linkage and cut it to recover the real
monomers. This works for any polymer that contains a known linkage, regardless of
whether a neural model ever saw something like it.

METHOD
------
The repeat unit has two '*' connection points.
  1. Find the BACKBONE (shortest path between the two '*') and tag those atoms;
     this is what distinguishes a real polymerisation linkage from a pendant group
     (e.g. PMMA's side-chain ester must NOT be treated as the backbone linkage).
  2. CYCLISE the unit (bond the two backbone ends, drop the '*') so a linkage that
     spans the '*' boundary (nylon-6, PLA, PCL ...) becomes an internal, detectable
     bond.
  3. Cut every backbone carbonyl->heteroatom (C(=O)-O / C(=O)-N) bond at once and
     cap each acyl carbon with -OH. The resulting fragments are the real monomers.
For an all-carbon backbone (chain-growth) there is no such linkage, so we instead
rebuild the C=C of the olefin monomer.

Every function is defensive and returns [] / None on failure.
"""

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def _mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _canon(mol):
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def is_valid_monomer(smi):
    """A monomer prediction is trustworthy only if it parses to a real molecule."""
    if not smi or not isinstance(smi, str):
        return False
    m = Chem.MolFromSmiles(smi.strip())
    return m is not None and m.GetNumHeavyAtoms() >= 2


# carbonyl carbon single-bonded to O or N (the cleavable acyl-heteroatom bond)
_ACYL_HETERO = Chem.MolFromSmarts("[CX3](=[OX1])[#7,#8]")
# ether C-O-C (O not part of a carbonyl/ester) -> polyether / poly(arylene ether).
# Includes AROMATIC carbons so aryl ethers (PPO/PPE-type diaryl ethers) are covered.
_ETHER = Chem.MolFromSmarts("[#6][OX2;!$([OX2][CX3]=[OX1])][#6]")
# sulfur analogues
_THIOESTER = Chem.MolFromSmarts("[CX3](=[OX1])[SX2]")          # -C(=O)-S- -> acid + thiol
_THIOETHER = Chem.MolFromSmarts("[#6][SX2;!$([SX2][CX3]=[OX1])][#6]")  # C-S-C -> dithiol
_DISULFIDE = Chem.MolFromSmarts("[SX2][SX2]")                  # -S-S- -> dithiol
_IMINE = Chem.MolFromSmarts("[CX3]=[NX2]")                     # azomethine -> aldehyde + amine


def _cyclize_and_tag(smi):
    """
    Return (cyclic_mol, ok). Backbone atoms (on the path between the two '*') are
    tagged with atom-map-number 1 so we can tell backbone linkages from pendants.
    """
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    try:
        path = set(Chem.GetShortestPath(mol, stars[0], stars[1]))
    except Exception:
        return None
    for i in path:
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() != 0:
            a.SetAtomMapNum(1)

    nbrs = []
    for s in stars:
        ns = mol.GetAtomWithIdx(s).GetNeighbors()
        if len(ns) != 1:
            return None
        nbrs.append(ns[0].GetIdx())

    rw = Chem.RWMol(mol)
    if nbrs[0] != nbrs[1] and rw.GetBondBetweenAtoms(nbrs[0], nbrs[1]) is None:
        rw.AddBond(nbrs[0], nbrs[1], Chem.BondType.SINGLE)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def _backbone_acyl_bonds(mol):
    """Bonds (carbonylC, heteroatom) to cut: backbone carbonyl -> single O/N."""
    bonds = []
    for match in mol.GetSubstructMatches(_ACYL_HETERO):
        c_idx, _o_dbl, x_idx = match
        if (mol.GetAtomWithIdx(c_idx).GetAtomMapNum() == 1
                and mol.GetAtomWithIdx(x_idx).GetAtomMapNum() == 1):
            bonds.append((c_idx, x_idx))
    return bonds


def _backbone_ether_bonds(mol):
    """One cleavable C-O bond per backbone ether oxygen (deduped by the O atom)."""
    bonds, seen = [], set()
    for match in mol.GetSubstructMatches(_ETHER):
        c1, o, c2 = match
        if o in seen:
            continue
        if all(mol.GetAtomWithIdx(i).GetAtomMapNum() == 1 for i in (c1, o, c2)):
            seen.add(o)
            bonds.append((c1, o))   # cut c1-O, cap c1 with -OH; O keeps c2 -> HO-c2
    return bonds


def _classify(mol, bonds):
    """Name the polymerisation family from the environment of the cut carbonyls."""
    n_ester = n_amide = n_carbonate = n_urethane = n_urea = n_imide = 0
    for c_idx, _x in bonds:
        c = mol.GetAtomWithIdx(c_idx)
        hetero = [nb for nb in c.GetNeighbors()
                  if nb.GetAtomMapNum() == 1 and nb.GetAtomicNum() in (7, 8)
                  and mol.GetBondBetweenAtoms(c_idx, nb.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
        n_o = sum(1 for h in hetero if h.GetAtomicNum() == 8)
        n_n = sum(1 for h in hetero if h.GetAtomicNum() == 7)
        if n_o >= 2:
            n_carbonate += 1
        elif n_o == 1 and n_n == 1:
            n_urethane += 1
        elif n_n >= 2:
            n_urea += 1
        elif n_n == 1:
            # imide if this N is shared by two carbonyls
            n_atom = next((h for h in hetero if h.GetAtomicNum() == 7), None)
            if n_atom is not None and sum(1 for nb in n_atom.GetNeighbors()
                                          if nb.GetSymbol() == "C" and any(
                                              b.GetBondType() == Chem.BondType.DOUBLE and
                                              b.GetOtherAtom(nb).GetAtomicNum() == 8
                                              for b in nb.GetBonds())) >= 2:
                n_imide += 1
            else:
                n_amide += 1
        elif n_o == 1:
            n_ester += 1
    for name, cnt in [("imide", n_imide), ("urea", n_urea), ("urethane", n_urethane),
                      ("carbonate", n_carbonate), ("amide", n_amide), ("ester", n_ester)]:
        if cnt:
            return name
    return None


_META = {
    "imide":     ("Polyimide", "Poly-condensation (cyclo-imidisation)", ["aromatic tetra-acid / dianhydride", "diamine"]),
    "urea":      ("Polyurea", "Step-growth (isocyanate + amine)", ["diamine", "carbonyl source"]),
    "urethane":  ("Polyurethane", "Poly-addition (isocyanate + alcohol)", ["diol", "diamine", "carbonyl source"]),
    "carbonate": ("Polycarbonate", "Condensation / transesterification", ["diol", "carbonyl source (phosgene/CO2)"]),
    "ester":     ("Polyester", "Step-growth condensation", ["dicarboxylic acid", "diol"]),
    "amide":     ("Polyamide", "Step-growth condensation", ["dicarboxylic acid", "diamine"]),
}


def _cut_and_cap(mol, bonds, cap=8):
    """Cut each (atom_to_cap, atom_to_free) bond and cap the first atom with a new
    atom of atomic number `cap` (8=O -> -OH for acids/diols; 16=S -> -SH for dithiols)."""
    rw = Chem.RWMol(mol)
    for c_idx, x_idx in bonds:
        if rw.GetBondBetweenAtoms(c_idx, x_idx) is not None:
            rw.RemoveBond(c_idx, x_idx)
            o = rw.AddAtom(Chem.Atom(cap))
            rw.AddBond(c_idx, o, Chem.BondType.SINGLE)
    m = rw.GetMol()
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return []
    out = []   # ALL fragments (not deduped) so the caller can both dedupe for
    for frag in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True):  # display and count atoms
        cs = _canon(frag)
        if cs:
            out.append(cs)
    return out


def _ab_monomer(smi):
    """
    AB (self-condensing) monomer recovery: cap the two '*' directly instead of
    cyclising. Used for single-unit condensation monomers where cyclisation would
    create an impossible bridge -- e.g. poly(arylene ether) from a phenol (both '*'
    on one aromatic ring), or a hydroxy-/amino-acid.

    A carbonyl end is capped to -COOH; a heteroatom (O/N) or aryl/C end is capped
    with H (-> phenol/alcohol/amine/Ar-H). Requires at least one condensation-type
    (heteroatom or carbonyl) end, so vinyl polymers are left to the olefin path.
    """
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None

    condensation_end = False
    for s in stars:
        nb = mol.GetAtomWithIdx(s).GetNeighbors()
        if not nb:
            return None
        n = nb[0]
        if n.GetAtomicNum() in (7, 8):
            condensation_end = True
        elif n.GetAtomicNum() == 6 and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(n).GetAtomicNum() == 8
                for b in n.GetBonds()):
            n.SetAtomMapNum(99)          # carbonyl carbon -> will get -OH (acid)
            condensation_end = True
    if not condensation_end:
        return None                      # all-carbon ends -> not condensation (leave to vinyl)

    rw = Chem.RWMol(mol)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    for a in list(rw.GetAtoms()):
        if a.GetAtomMapNum() == 99:
            a.SetAtomMapNum(0)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(a.GetIdx(), o, Chem.BondType.SINGLE)
    return _canon(rw.GetMol())


def _classify_ab(smi):
    """Name the AB polymer family from the functional groups of its monomer."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None

    def has(sm):
        p = Chem.MolFromSmarts(sm)
        return p is not None and m.HasSubstructMatch(p)

    phenol = has("[OX2H]c")
    alcohol = has("[OX2H][CX4]")
    acid = has("C(=O)[OX2H]")
    amine = has("[NX3;H1,H2]")
    if acid and alcohol:
        return ("Polyester (AB / hydroxy-acid)", "Self-condensation of a hydroxy-acid", ["hydroxy-acid"])
    if acid and amine:
        return ("Polyamide (AB / amino-acid)", "Self-condensation of an amino-acid", ["amino-acid"])
    if phenol:
        return ("Poly(arylene ether)", "Oxidative coupling / SNAr of a phenol", ["phenol"])
    if alcohol or amine or acid:
        return ("Condensation (AB monomer)", "Self-condensation of an AB monomer", ["AB monomer"])
    return None


def _siloxane_monomer(smi):
    """Silicone: backbone contains Si; recover the silanediol (R2Si(OH)2) monomer."""
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    if not any(a.GetAtomicNum() == 14 for a in mol.GetAtoms()):   # needs silicon
        return None
    for s in stars:
        n = mol.GetAtomWithIdx(s).GetNeighbors()[0]
        if n.GetAtomicNum() == 14:
            n.SetAtomMapNum(98)          # Si end -> cap with -OH (silanol)
    rw = Chem.RWMol(mol)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    for a in list(rw.GetAtoms()):
        if a.GetAtomMapNum() == 98:
            a.SetAtomMapNum(0)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(a.GetIdx(), o, Chem.BondType.SINGLE)
    return _canon(rw.GetMol())


def _aryl_coupling_monomer(smi):
    """Aryl-aryl backbone (both '*' on aromatic C): recover the dihaloarene monomer."""
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    nbrs = [mol.GetAtomWithIdx(s).GetNeighbors()[0] for s in stars
            if mol.GetAtomWithIdx(s).GetNeighbors()]
    if len(nbrs) != 2 or not all(n.GetIsAromatic() and n.GetAtomicNum() == 6 for n in nbrs):
        return None
    for n in nbrs:
        n.SetAtomMapNum(97)              # aryl end -> cap with Br (coupling handle)
    rw = Chem.RWMol(mol)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    for a in list(rw.GetAtoms()):
        if a.GetAtomMapNum() == 97:
            a.SetAtomMapNum(0)
            br = rw.AddAtom(Chem.Atom(35))
            rw.AddBond(a.GetIdx(), br, Chem.BondType.SINGLE)
    return _canon(rw.GetMol())


def _vinyl_monomer(smi):
    """Chain-growth: rebuild the C=C double bond of the olefin monomer."""
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    nbrs = [mol.GetAtomWithIdx(s).GetNeighbors()[0].GetIdx()
            for s in stars if mol.GetAtomWithIdx(s).GetNeighbors()]
    if len(nbrs) != 2 or nbrs[0] == nbrs[1]:
        return None
    # only a genuine C=C addition monomer -- not Si-O (siloxane) etc.
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in nbrs):
        return None
    rw = Chem.RWMol(mol)
    b = rw.GetBondBetweenAtoms(nbrs[0], nbrs[1])
    if b is None or b.GetBondType() != Chem.BondType.SINGLE:
        return None
    b.SetBondType(Chem.BondType.DOUBLE)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    return _canon(rw.GetMol())


def _diene_monomer(smi):
    """
    1,4-addition diene rubber: a -CH2-CH=CH-CH2- backbone (4 backbone carbons, one
    internal C=C) comes from a 1,3-diene. Shift the bonds so both ends become C=C
    -> the diene monomer (butadiene / isoprene / chloroprene ...).
    """
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    ns = [mol.GetAtomWithIdx(s).GetNeighbors()[0] for s in stars
          if mol.GetAtomWithIdx(s).GetNeighbors()]
    if len(ns) != 2 or any(n.GetAtomicNum() != 6 for n in ns):
        return None
    c0, c1 = ns[0].GetIdx(), ns[1].GetIdx()
    try:
        path = Chem.GetShortestPath(mol, c0, c1)
    except Exception:
        return None
    if len(path) != 4:
        return None
    a, b, c, d = path
    ab = mol.GetBondBetweenAtoms(a, b)
    bc = mol.GetBondBetweenAtoms(b, c)
    cd = mol.GetBondBetweenAtoms(c, d)
    if not (ab and bc and cd):
        return None
    if not (ab.GetBondType() == Chem.BondType.SINGLE
            and bc.GetBondType() == Chem.BondType.DOUBLE
            and cd.GetBondType() == Chem.BondType.SINGLE):
        return None
    rw = Chem.RWMol(mol)
    rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
    rw.GetBondBetweenAtoms(b, c).SetBondType(Chem.BondType.SINGLE)
    rw.GetBondBetweenAtoms(c, d).SetBondType(Chem.BondType.DOUBLE)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    return _canon(rw.GetMol())


def _addition_monomers(smi):
    """
    Saturated all-carbon backbone -> the vinyl (olefin) monomer(s) of an addition
    (co)polymer. The backbone (path between the two '*') is split into consecutive
    C-C pairs -- each pair is one monomer whose C=C was opened during polymerisation.
    Even-length backbones only (odd = ambiguous). Homopolymers dedupe to one monomer
    (e.g. *CCCC* -> ethylene, *CC(C)CC(C)* -> propene); a mixed backbone yields the
    distinct comonomers (e.g. *CCC(F)(F)C(*)(C)F -> ethylene + a fluoro-olefin).
    """
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    ns = [mol.GetAtomWithIdx(s).GetNeighbors() for s in stars]
    if not ns[0] or not ns[1]:
        return None
    c0, c1 = ns[0][0].GetIdx(), ns[1][0].GetIdx()
    if c0 == c1:
        return None
    try:
        path = list(Chem.GetShortestPath(mol, c0, c1))
    except Exception:
        return None
    L = len(path)
    if L < 4 or L % 2 != 0:              # 2 is handled by _vinyl_monomer; odd is ambiguous
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path):
        return None
    for i in range(L - 1):
        b = mol.GetBondBetweenAtoms(path[i], path[i + 1])
        if b is None or b.GetBondType() != Chem.BondType.SINGLE:
            return None                  # any double bond -> not a saturated addition backbone

    # tag path atoms so we can find them after removing the '*' atoms
    for k, idx in enumerate(path):
        mol.GetAtomWithIdx(idx).SetAtomMapNum(k + 1)
    rw = Chem.RWMol(mol)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    p = {a.GetAtomMapNum(): a.GetIdx() for a in rw.GetAtoms() if a.GetAtomMapNum()}
    p = [p[k + 1] for k in range(L)]
    for k in range(1, L - 1, 2):         # cut the inter-monomer bonds
        if rw.GetBondBetweenAtoms(p[k], p[k + 1]) is not None:
            rw.RemoveBond(p[k], p[k + 1])
    for k in range(0, L, 2):             # restore each monomer's C=C
        b = rw.GetBondBetweenAtoms(p[k], p[k + 1])
        if b is not None:
            b.SetBondType(Chem.BondType.DOUBLE)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    frags = [_canon(f) for f in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)]
    frags = [f for f in frags if f and is_valid_monomer(f)]   # ALL frags (not deduped)
    return frags if frags else None


def _backbone_thioester_bonds(mol):
    """Backbone -C(=O)-S- bonds (cut acyl-S, cap acyl C with -OH -> acid + thiol)."""
    bonds = []
    for c_idx, _o, s_idx in mol.GetSubstructMatches(_THIOESTER):
        if (mol.GetAtomWithIdx(c_idx).GetAtomMapNum() == 1
                and mol.GetAtomWithIdx(s_idx).GetAtomMapNum() == 1):
            bonds.append((c_idx, s_idx))
    return bonds


def _backbone_thioether_bonds(mol):
    """One cleavable C-S bond per backbone thioether/disulfide sulfur (-> dithiol)."""
    bonds, seen = [], set()
    for c1, s, c2 in mol.GetSubstructMatches(_THIOETHER):
        if s in seen:
            continue
        if all(mol.GetAtomWithIdx(i).GetAtomMapNum() == 1 for i in (c1, s, c2)):
            seen.add(s)
            bonds.append((c1, s))
    for s1, s2 in mol.GetSubstructMatches(_DISULFIDE):        # disulfide -> break S-S
        if s1 in seen or s2 in seen:
            continue
        if all(mol.GetAtomWithIdx(i).GetAtomMapNum() == 1 for i in (s1, s2)):
            seen.add(s1); seen.add(s2)
            bonds.append((s1, s2))
    return bonds


def _imine_route(smi):
    """Polyazomethine (backbone C=N) -> dialdehyde/diketone + diamine (retro-Schiff)."""
    cyc = _cyclize_and_tag(smi)
    if cyc is None:
        return None
    matches = [(c, n) for c, n in cyc.GetSubstructMatches(_IMINE)
               if cyc.GetAtomWithIdx(c).GetAtomMapNum() == 1
               and cyc.GetAtomWithIdx(n).GetAtomMapNum() == 1]
    if not matches:
        return None
    rw = Chem.RWMol(cyc)
    for c_idx, n_idx in matches:
        b = rw.GetBondBetweenAtoms(c_idx, n_idx)
        if b is None:
            continue
        rw.RemoveBond(c_idx, n_idx)          # break C=N
        o = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c_idx, o, Chem.BondType.DOUBLE)   # C=O (aldehyde/ketone); N -> amine (implicit H)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(0)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    frags = [_canon(f) for f in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)]
    return [f for f in frags if f and is_valid_monomer(f)] or None


def _romp_monomer(smi):
    """
    Unsaturated all-carbon backbone (>=1 C=C, longer than a diene) -> the cyclic olefin
    of a ROMP polymer. Cyclise the backbone ends into the ring (e.g. polyoctenamer
    *CC=CCCCCC* -> cyclooctene).
    """
    mol = _mol(smi)
    if mol is None:
        return None
    stars = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    ns = [mol.GetAtomWithIdx(s).GetNeighbors() for s in stars]
    if not ns[0] or not ns[1]:
        return None
    c0, c1 = ns[0][0].GetIdx(), ns[1][0].GetIdx()
    if c0 == c1:
        return None
    try:
        path = list(Chem.GetShortestPath(mol, c0, c1))
    except Exception:
        return None
    if len(path) < 5:                        # shorter unsaturated cases handled by _diene
        return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in path):
        return None
    has_cc = any(mol.GetBondBetweenAtoms(path[i], path[i + 1]).GetBondType() == Chem.BondType.DOUBLE
                 for i in range(len(path) - 1))
    if not has_cc:                           # saturated -> not a ROMP olefin (that's addition)
        return None
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(c0, c1) is None:
        rw.AddBond(c0, c1, Chem.BondType.SINGLE)
    for s in sorted(stars, reverse=True):
        rw.RemoveAtom(s)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return _canon(m)


def retro_decompose(smi, verified_only=False):
    """
    Return a list of retrosynthesis routes for a polymer repeat unit (SMILES with
    two '*'). Each route: {type, mechanism, monomers:[SMILES], monomer_roles:[..],
    method, verified:bool, verify_reason:str}. The first route is the best guess.

    The route is checked with verify_route(); `verified` says whether the recovered
    monomers are chemically consistent (right functional groups + atom conservation).
    With verified_only=True, an unverified route is suppressed (returns []).
    Returns [] when no disconnection is found.
    """
    if not smi:
        return []

    route = None
    cyc = _cyclize_and_tag(smi)

    # 1. Step-growth: cut BACKBONE carbonyl->heteroatom bonds (ignores pendants).
    if cyc is not None:
        bonds = _backbone_acyl_bonds(cyc)
        if bonds:
            cls = _classify(cyc, bonds)
            frags = [m for m in _cut_and_cap(cyc, bonds) if is_valid_monomer(m)]
            if cls and frags:
                name, mech, roles = _META[cls]
                route = {"type": name, "mechanism": mech,
                         "monomers": list(dict.fromkeys(frags)),  # unique, for display
                         "_frag_heavy": _total_heavy(frags),      # all fragments, for atom check
                         "monomer_roles": roles, "method": "rule-based backbone disconnection"}

    # 1a2. Polythioester backbone: -C(=O)-S- -> dicarboxylic acid + dithiol.
    if route is None and cyc is not None:
        tbonds = _backbone_thioester_bonds(cyc)
        if tbonds:
            frags = [m for m in _cut_and_cap(cyc, tbonds, cap=8) if is_valid_monomer(m)]
            if frags:
                route = {"type": "Polythioester",
                         "mechanism": "Step-growth (acid/thiol or thioacid + ...)",
                         "monomers": list(dict.fromkeys(frags)), "_frag_heavy": _total_heavy(frags),
                         "monomer_roles": ["dicarboxylic acid", "dithiol"],
                         "method": "rule-based backbone disconnection"}

    # 1b. Polyether / poly(arylene ether) backbone: cut ethers -> diols/phenols.
    if route is None and cyc is not None:
        ethers = _backbone_ether_bonds(cyc)
        if ethers:
            frags = [m for m in _cut_and_cap(cyc, ethers) if is_valid_monomer(m)]
            if frags:
                route = {"type": "Polyether",
                         "mechanism": "Condensation of diols / ring-opening of epoxide",
                         "monomers": list(dict.fromkeys(frags)),
                         "_frag_heavy": _total_heavy(frags),
                         "monomer_roles": ["diol (or epoxide)"],
                         "method": "rule-based backbone disconnection"}

    # 1b2. Polythioether / polysulfide backbone: C-S-C or S-S -> dithiol.
    if route is None and cyc is not None:
        sbonds = _backbone_thioether_bonds(cyc)
        if sbonds:
            frags = [m for m in _cut_and_cap(cyc, sbonds, cap=16) if is_valid_monomer(m)]
            if frags:
                route = {"type": "Polythioether / polysulfide",
                         "mechanism": "Dithiol + dihalide / thiol-ene / ring-opening",
                         "monomers": list(dict.fromkeys(frags)), "_frag_heavy": _total_heavy(frags),
                         "monomer_roles": ["dithiol"],
                         "method": "rule-based backbone disconnection"}

    # 1b3. Polyazomethine (backbone C=N) -> dialdehyde + diamine.
    if route is None:
        imine = _imine_route(smi)
        if imine:
            route = {"type": "Polyimine (azomethine)",
                     "mechanism": "Condensation of a dialdehyde/diketone + diamine (Schiff base)",
                     "monomers": list(dict.fromkeys(imine)), "_frag_heavy": _total_heavy(imine),
                     "monomer_roles": ["dialdehyde/diketone", "diamine"],
                     "method": "rule-based backbone disconnection"}

    # 1c. AB self-condensing monomer (phenol / hydroxy-acid / amino-acid).
    if route is None:
        ab = _ab_monomer(smi)
        if ab and is_valid_monomer(ab):
            cls = _classify_ab(ab)
            if cls:
                name, mech, roles = cls
                route = {"type": name, "mechanism": mech, "monomers": [ab],
                         "monomer_roles": roles, "method": "AB self-condensation disconnection"}

    # 1d. Silicone (Si-O backbone) -> silanediol.
    if route is None:
        silox = _siloxane_monomer(smi)
        if silox and is_valid_monomer(silox):
            route = {"type": "Polysiloxane (silicone)",
                     "mechanism": "Hydrolysis/condensation of a dichlorosilane (or ROP of a cyclosiloxane)",
                     "monomers": [silox], "monomer_roles": ["silanediol / dichlorosilane"],
                     "method": "AB self-condensation disconnection"}

    # 2. Chain-growth: rebuild the olefin (adjacent ends) or diene (1,4-addition).
    if route is None:
        vinyl = _vinyl_monomer(smi)
        if vinyl and is_valid_monomer(vinyl):
            route = {"type": "Vinyl / addition polymer",
                     "mechanism": "Chain-growth (radical / ionic addition)",
                     "monomers": [vinyl], "monomer_roles": ["olefin monomer"],
                     "method": "double-bond reconstruction"}
    if route is None:
        diene = _diene_monomer(smi)
        if diene and is_valid_monomer(diene):
            route = {"type": "Vinyl / diene addition polymer",
                     "mechanism": "1,4-addition of a 1,3-diene (rubber)",
                     "monomers": [diene], "monomer_roles": ["1,3-diene monomer"],
                     "method": "diene double-bond reconstruction"}
    if route is None:
        romp = _romp_monomer(smi)
        if romp and is_valid_monomer(romp):
            route = {"type": "Poly(alkenylene) (ROMP/ADMET)",
                     "mechanism": "Ring-opening metathesis of a cyclic olefin (or ADMET diene)",
                     "monomers": [romp], "monomer_roles": ["cyclic olefin"],
                     "method": "backbone cyclisation -> cyclic olefin"}
    if route is None:
        addn = _addition_monomers(smi)
        if addn:
            uniq = list(dict.fromkeys(addn))     # dedupe for display (homopolymer -> 1 monomer)
            n = len(uniq)
            route = {"type": "Vinyl / addition polymer" if n == 1 else "Vinyl / addition copolymer",
                     "mechanism": "Chain-growth addition of olefin monomer(s)",
                     "monomers": uniq, "_frag_heavy": _total_heavy(addn),  # all frags -> atom check
                     "monomer_roles": ["olefin monomer"] if n == 1 else ["olefin comonomers"],
                     "method": "addition backbone -> olefin(s)"}

    # 3. Aryl-aryl coupled backbone -> dihaloarene.
    if route is None:
        aryl = _aryl_coupling_monomer(smi)
        if aryl and is_valid_monomer(aryl):
            route = {"type": "Poly(arylene) (aryl-aryl coupling)",
                     "mechanism": "Transition-metal coupling (Suzuki / Yamamoto)",
                     "monomers": [aryl], "monomer_roles": ["dihaloarene / diboronic acid"],
                     "method": "aryl-aryl disconnection"}

    if route is None:
        return []

    ok, reason = verify_route(smi, route)
    route["verified"] = ok
    route["verify_reason"] = reason
    if verified_only and not ok:
        return []
    return [route]


# ---------------------------------------------------------------------------
# Route verification -- flags chemically inconsistent decompositions.
# ---------------------------------------------------------------------------
def _count_groups(mols, smarts):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return sum(len(m.GetSubstructMatches(patt)) for m in mols)


def _total_heavy(smiles_list):
    """Total heavy atoms across a fragment list (with repeats) -- for atom accounting."""
    total = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            total += m.GetNumHeavyAtoms()
    return total


def verify_route(polymer_smi, route):
    """
    Return (ok, reason). Checks that the recovered monomers are (a) valid, (b) carry
    the functional groups implied by the polymerisation type, and (c) conserve atoms
    (no atoms silently lost or duplicated). Used to catch wrong decompositions.
    """
    monomers = route.get("monomers") or []
    if not monomers or not all(is_valid_monomer(m) for m in monomers):
        return False, "invalid/empty monomer SMILES"
    mols = [Chem.MolFromSmiles(m) for m in monomers]

    cooh = _count_groups(mols, "C(=O)[OX2H]")
    oh_total = _count_groups(mols, "[OX2H]")          # every hydroxyl...
    oh = max(0, oh_total - cooh)                       # ...minus the acid -OH -> alcohol/phenol/enol/silanol
    nh = _count_groups(mols, "[NX3;H1,H2]")
    cc = _count_groups(mols, "[CX3]=[CX3]")
    t = route.get("type", "")

    if t.startswith("Polyester"):
        ok = cooh >= 1 and oh >= 1
        reason = "polyester monomers must carry -COOH and -OH"
    elif t.startswith("Polyamide"):
        ok = cooh >= 1 and nh >= 1
        reason = "polyamide monomers must carry -COOH and -NH"
    elif t == "Polyether" or "arylene ether" in t:
        ok = oh >= 1
        reason = "ether monomers must carry -OH / phenol"
    elif t.startswith("Polycarbonate"):
        ok = oh >= 1
        reason = "carbonate needs a diol"
    elif t.startswith("Polyurethane") or t.startswith("Polyurea"):
        ok = (oh + nh) >= 1
        reason = "urethane/urea needs -OH and/or -NH monomers"
    elif t.startswith("Polyimide"):
        ok = cooh >= 1
        reason = "imide route needs an acid/anhydride fragment"
    elif "Vinyl" in t:
        ok = cc >= 1
        reason = "vinyl monomer must contain C=C"
    elif "arylene) (aryl" in t:
        ok = _count_groups(mols, "[#6;a][F,Cl,Br,I]") >= 1
        reason = "aryl-coupling monomer must carry an aryl halide handle"
    elif t.startswith("Poly(alkenylene)"):
        ok = _count_groups(mols, "[CX3;R]=[CX3;R]") >= 1
        reason = "ROMP monomer must be a cyclic olefin"
    elif t.startswith("Polysiloxane"):
        ok = _count_groups(mols, "[Si]") >= 1
        reason = "siloxane monomer must contain Si"
    elif t.startswith("Polythioester"):
        ok = cooh >= 1 and _count_groups(mols, "[SX2H]") >= 1
        reason = "thioester monomers must carry -COOH and -SH"
    elif "thioether" in t or "polysulfide" in t:
        ok = _count_groups(mols, "[SX2H]") >= 1
        reason = "thioether/sulfide monomer must carry -SH"
    elif t.startswith("Polyimine"):
        ok = _count_groups(mols, "[CX3]=[OX1]") >= 1 and nh >= 1
        reason = "azomethine needs a carbonyl (aldehyde/ketone) + amine"
    else:
        ok, reason = True, "ok"

    if not ok:
        return False, reason

    # Atom conservation: the cut fragments should account for all polymer heavy
    # atoms (retro only ADDS caps, never drops atoms). Use the full (non-deduped)
    # fragment count when available, so a monomer that legitimately repeats across
    # the chain (e.g. PEO -> ethylene glycol) isn't mistaken for lost atoms.
    poly = Chem.MolFromSmiles(polymer_smi.replace("*", "[H]"))
    if poly is not None:
        p_heavy = poly.GetNumHeavyAtoms()
        m_heavy = route.get("_frag_heavy")
        if m_heavy is None:
            m_heavy = sum(m.GetNumHeavyAtoms() for m in mols)
        if m_heavy < p_heavy:
            return False, f"atoms lost ({m_heavy} < {p_heavy})"

    return True, "ok"
