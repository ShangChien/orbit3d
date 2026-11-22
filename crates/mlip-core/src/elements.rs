//! Full 118-element IUPAC periodic table metadata: symbols, atomic numbers,
//! standard atomic masses (amu / g·mol⁻¹), and covalent radii (Å).

/// Element symbols ordered by atomic number `Z - 1` (`Z = 1..=118`).
pub const ELEMENT_SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Standard atomic weights (in unified atomic mass units `u` / `amu`) for `Z = 1..=118`.
pub const ATOMIC_MASSES_AMU: [f64; 118] = [
    1.0080, 4.0026, 6.9400, 9.0122, 10.810, 12.011, 14.007, 15.999, 18.998, 20.180,
    22.990, 24.305, 26.982, 28.085, 30.974, 32.060, 35.450, 39.948, 39.098, 40.078,
    44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.380,
    69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.620, 88.906, 91.224,
    92.906, 95.950, 98.000, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,
    121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,
    145.00, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05,
    174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,
    204.38, 207.20, 208.98, 209.00, 210.00, 222.00, 223.00, 226.00, 227.00, 232.04,
    231.04, 238.03, 237.00, 244.00, 243.00, 247.00, 247.00, 251.00, 252.00, 257.00,
    258.00, 259.00, 266.00, 267.00, 268.00, 269.00, 270.00, 277.00, 278.00, 281.00,
    282.00, 285.00, 286.00, 289.00, 290.00, 293.00, 294.00, 294.00,
];

/// Pyykkö / Cordero single-bond covalent radii (Å) for `Z = 1..=118`.
pub const COVALENT_RADII_A: [f64; 118] = [
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.68, 1.68, 1.65, 1.67,
    1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29, 1.28,
    1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57,
];

/// Map atomic number `Z` (`1..=118`) to IUPAC symbol.
#[inline]
pub fn symbol_from_z(z: u8) -> Result<&'static str, String> {
    if (1..=118).contains(&z) {
        Ok(ELEMENT_SYMBOLS[(z - 1) as usize])
    } else {
        Err(format!("Invalid atomic number Z={z} (expected 1..=118)"))
    }
}

/// Map case-insensitive element symbol (e.g. `"C"`, `"Ir"`, `"fe"`) to atomic number `Z` (`1..=118`).
pub fn z_from_symbol(sym: &str) -> Result<u8, String> {
    let trimmed = sym.trim();
    for (idx, &cand) in ELEMENT_SYMBOLS.iter().enumerate() {
        if cand.eq_ignore_ascii_case(trimmed) {
            return Ok((idx + 1) as u8);
        }
    }
    Err(format!("Unknown element symbol '{sym}'"))
}

/// Standard atomic mass in `amu` for atomic number `Z` (`1..=118`).
#[inline]
pub fn mass_amu_from_z(z: u8) -> f64 {
    if (1..=118).contains(&z) {
        ATOMIC_MASSES_AMU[(z - 1) as usize]
    } else {
        12.011
    }
}

/// Single-bond covalent radius in `Å` for atomic number `Z` (`1..=118`).
#[inline]
pub fn covalent_radius_a_from_z(z: u8) -> f64 {
    if (1..=118).contains(&z) {
        COVALENT_RADII_A[(z - 1) as usize]
    } else {
        1.50
    }
}
