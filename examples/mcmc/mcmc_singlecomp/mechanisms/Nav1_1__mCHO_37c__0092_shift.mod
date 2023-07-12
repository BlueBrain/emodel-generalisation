: <> generated: 2022-Sep-20 13:05:01
: <> family: NaV
: <> type: HodgkinHuxleyAlphaBeta
: <> model: Eyring
: <> institution: Ecole polytechnique federale de Lausanne (EPFL)
: <> lab: Blue Brain Project (BBP)
: <> host_cell: CHO_FT
: <> ion_channel: Nav1.1_cop
: <> species: Mouse
: <> model_id: 0092
: <> temperature: 37c
: <> ljp_corrected: True

NEURON {
    SUFFIX Nav1_1_shift
    USEION na READ ena WRITE ina
    RANGE gna, ina, gnabar
	EXTERNAL v_shift__nav_family
}

: ========== VARIABLES ==========

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gnabar = 5.000000e-02 (S / cm2)
    rate_max = 1.000000e+03 (1 / ms)
    P01 = 9.970627e+00 (1 / ms)
    P02 = 2.965675e-02 (1 / mV)
    P03 = 1.058889e+00 (1 / ms)
    P04 = 7.261443e-02 (1 / mV)
    P05 = 2.704686e+00 (1 / ms)
    P06 = 2.192504e-02 (1 / mV)
    P07 = 4.983769e-03 (1 / ms)
    P08 = 6.466780e-02 (1 / mV)
	v_shift__nav_family = 0 (mV)
}

ASSIGNED {
    v (mV)
    ena (mV)
    ina (mA / cm2)
    gna (S / cm2)
    m_inf
    h_inf
    m_tau (ms)
    h_tau (ms)
    vcut_mAlpha (mV)
    vcut_hAlpha (mV)
    vcut_mBeta (mV)
    vcut_hBeta (mV)
}

: ========== FUNCTION DEFINITIONS ==========

FUNCTION rate (k1 (1 / ms), k2 (1 / mV), vm (mV)) (1 / ms) {
    rate = k1 * exp(k2 * vm)
}

FUNCTION mAlpha (vm (mV)) (1 / ms) {
    if (vm < vcut_mAlpha) {mAlpha = rate(P01, P02, vm)}
    else {mAlpha = rate_max}
}

FUNCTION mBeta (vm (mV)) (1 / ms) {
    if (vm > vcut_mBeta) {mBeta = rate(P03, -P04, vm)}
    else {mBeta = rate_max}
}

FUNCTION hAlpha (vm (mV)) (1 / ms) {
    if (vm < vcut_hAlpha) {hAlpha = rate(P05, P06, vm)}
    else {hAlpha = rate_max}
}

FUNCTION hBeta (vm (mV)) (1 / ms) {
    if (vm > vcut_hBeta) {hBeta = rate(P07, -P08, vm)}
    else {hBeta = rate_max}
}

FUNCTION mTau (vm (mV)) (ms) {
    mTau = 1 / (mAlpha(vm) + mBeta(vm))
}

FUNCTION hTau (vm (mV)) (ms) {
    hTau = 1 / (hAlpha(vm) + hBeta(vm))
}

FUNCTION mInf (vm (mV)) {
    mInf = mAlpha(vm) * mTau(vm)
}

FUNCTION hInf (vm (mV)) {
    hInf = hBeta(vm) * hTau(vm)
}

PROCEDURE rates() {
	LOCAL vm
	vm = v - v_shift__nav_family
    m_tau = mTau(vm)
    h_tau = hTau(vm)
    m_inf = mInf(vm)
    h_inf = hInf(vm)
}

: ========== PROGRAM ==========

STATE {
    m
    h
}

INITIAL {
    rates()
    m = m_inf
    h = h_inf

    vcut_mAlpha = log(rate_max / P01) / P02
    vcut_hAlpha = log(rate_max / P05) / P06
    vcut_mBeta = -log(rate_max / P03) / P04
    vcut_hBeta = -log(rate_max / P07) / P08
}

BREAKPOINT {
	LOCAL vm
    SOLVE states METHOD cnexp
    gna = gnabar * m * m * m * h
    vm = v - v_shift__nav_family
    ina = gna * (vm - ena)
}

DERIVATIVE states {
    rates()
    m' = (m_inf  - m) / m_tau
    h' = (h_inf  - h) / h_tau
}
