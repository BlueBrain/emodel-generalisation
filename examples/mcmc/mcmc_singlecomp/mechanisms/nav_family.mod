: Use to set parameters affecting an entire family of mechanisms
NEURON {
    SUFFIX nav_family
	RANGE v_shift
	GLOBAL v_shift_
}
UNITS { (mV) = (millivolt) }
PARAMETER { v_shift = 0 (mV) }
ASSIGNED { v_shift_ (mV) }
INITIAL { v_shift_ = v_shift }
