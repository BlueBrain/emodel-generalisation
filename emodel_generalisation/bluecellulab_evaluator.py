"""Compute the threshold and holding current using bluecellulab, adapted from BluePyThresh."""

import logging
import os
from copy import copy
from pathlib import Path

import bluecellulab
import efel
from bluecellulab.simulation.neuron_globals import NeuronGlobals
from bluepyparallel.evaluator import evaluate

from emodel_generalisation.utils import isolate

logger = logging.getLogger(__name__)
AXON_LOC = "self.axonal[1](0.5)._ref_v"


def calculate_threshold_current(cell, config, holding_current):
    """Calculate threshold current"""
    min_current_spike_count = run_spike_sim(
        cell,
        config,
        holding_current,
        config["min_threshold_current"],
    )
    logger.debug("min %s", min_current_spike_count)
    if min_current_spike_count > 0:
        logger.debug("Cell is firing spontaneously at min current, we divide by 2")
        if config["min_threshold_current"] == 0:
            return None
        config["max_threshold_current"] = copy(config["min_threshold_current"])
        config["min_threshold_current"] /= 2.0
        return calculate_threshold_current(cell, config, holding_current)

    max_current_spike_count = run_spike_sim(
        cell,
        config,
        holding_current,
        config["max_threshold_current"],
    )
    logger.debug("max %s", max_current_spike_count)
    if max_current_spike_count < 1:
        logger.debug("Cell is not firing at max current, we multiply by 2")
        config["min_threshold_current"] = copy(config["max_threshold_current"])
        config["max_threshold_current"] *= 1.2
        return calculate_threshold_current(cell, config, holding_current)

    return binsearch_threshold_current(
        cell,
        config,
        holding_current,
        config["min_threshold_current"],
        config["max_threshold_current"],
    )


def binsearch_threshold_current(cell, config, holding_current, min_current, max_current):
    """Binary search for threshold currents"""
    mid_current = (min_current + max_current) / 2
    if abs(max_current - min_current) < config.get("threshold_current_precision", 0.001):
        spike_count = run_spike_sim(
            cell,
            config,
            holding_current,
            max_current,
        )
        return max_current

    spike_count = run_spike_sim(
        cell,
        config,
        holding_current,
        mid_current,
    )
    if spike_count == 0:
        return binsearch_threshold_current(cell, config, holding_current, mid_current, max_current)

    return binsearch_threshold_current(cell, config, holding_current, min_current, mid_current)


def run_spike_sim(cell, config, holding_current, step_current):
    """Run simulation on a cell and compute number of spikes."""
    cell.add_step(0, config["step_stop"], holding_current)
    cell.add_step(config["step_start"], config["step_stop"], step_current)

    if config["spike_at_ais"]:
        cell.add_recordings(["neuron.h._ref_t", AXON_LOC], dt=cell.record_dt)

    sim = bluecellulab.Simulation()
    ng = NeuronGlobals.get_instance()
    ng.v_init = config["v_init"]
    ng.temperature = config["celsius"]

    sim.run(
        config["step_stop"],
        cvode=config.get("deterministic", True),
        dt=config.get("dt", 0.025),
    )

    time = cell.get_time()
    if config["spike_at_ais"]:
        voltage = cell.get_recording(AXON_LOC)
    else:
        voltage = cell.get_soma_voltage()

    if len(voltage) < 2:
        raise Exception("No voltage trace!")

    efel.reset()
    efel.setIntSetting("strict_stiminterval", True)
    spike_count_array = efel.getFeatureValues(
        [
            {
                "T": time,
                "V": voltage,
                "stim_start": [config["step_start"]],
                "stim_end": [config["step_stop"]],
            }
        ],
        ["Spikecount"],
    )[0]["Spikecount"]

    cell.persistent = []  # remove the step protocols for next run
    if spike_count_array is None or len(spike_count_array) != 1:
        raise Exception("Error during spike count calculation")
    return spike_count_array[0]


def set_cell_deterministic(cell, deterministic):
    """Disable stochasticity in ion channels"""
    deterministic = True
    for section in cell.cell.all:
        for compartment in section:
            for mech in compartment:
                mech_name = mech.name()
                if "Stoch" in mech_name:
                    if not deterministic:
                        deterministic = False
                    setattr(
                        section,
                        f"deterministic_{mech_name}",
                        1 if deterministic else 0,
                    )


def calculate_rmp_and_rin(cell, config):
    """Calculate rmp and input resistance from rmp."""
    cell.add_step(0, config["rin"]["step_stop"], 0)
    cell.add_step(
        config["rin"]["step_start"], config["rin"]["step_stop"], config["rin"]["step_amp"]
    )
    if config["rin"]["with_ttx"]:
        cell.enable_ttx()

    sim = bluecellulab.Simulation()
    ng = NeuronGlobals.get_instance()
    ng.v_init = config["v_init"]
    ng.temperature = config["celsius"]

    sim.run(
        config["rin"]["step_stop"],
        cvode=config.get("deterministic", True),
        dt=config.get("dt", 0.025),
    )

    time = cell.get_time()
    voltage = cell.get_soma_voltage()

    efel.reset()
    efel.setIntSetting("strict_stiminterval", True)
    trace = {
        "T": time,
        "V": voltage,
        "stim_start": [config["rin"]["step_start"]],
        "stim_end": [config["rin"]["step_stop"]],
        "stimulus_current": [config["rin"]["step_amp"]],
    }
    features = efel.getFeatureValues(
        [trace], ["spike_count", "voltage_base", "ohmic_input_resistance_vb_ssse"]
    )[0]

    rmp = 0
    if features["voltage_base"] is not None:
        rmp = features["voltage_base"][0]

    rin = -1.0
    if features["ohmic_input_resistance_vb_ssse"] is not None:
        rin = features["ohmic_input_resistance_vb_ssse"][0]

    if features["spike_count"] > 0:
        logger.warning("SPIKES! %s, %s", rmp, rin)
        return 0.0, -1.0
    return rmp, rin


def calculate_holding_current(cell, config):
    """Calculate holding current.

    adapted from: bluecellulab.tools.holding_current_subprocess,
    """
    vclamp = bluecellulab.neuron.h.SEClamp(0.5, sec=cell.soma)
    vclamp.rs = 0.01
    vclamp.dur1 = 2000
    vclamp.amp1 = config["holding_voltage"]

    simulation = bluecellulab.Simulation()
    ng = NeuronGlobals.get_instance()
    ng.v_init = config["v_init"]

    simulation.run(1000, cvode=config.get("deterministic", True), dt=config.get("dt", 0.025))

    return vclamp.i


def _current_evaluation(
    combo,
    protocol_config,
    emodels_hoc_dir,
    morphology_path="morphology_path",
    template_format="v6",
    only_rin=False,
):
    """Compute the threshold and holding currents."""
    cell_kwargs = {
        "template_path": str(Path(emodels_hoc_dir) / f"{combo['emodel']}.hoc"),
        "morphology_path": combo[morphology_path],
        "template_format": template_format,
        "emodel_properties": bluecellulab.circuit.EmodelProperties(
            holding_current=0,
            threshold_current=0,
            AIS_scaler=combo.get("@dynamics:AIS_scaler", None),
            soma_scaler=combo.get("@dynamics:soma_scaler", None),
        ),
    }
    cell = bluecellulab.Cell(**cell_kwargs)
    set_cell_deterministic(cell, protocol_config["deterministic"])
    if not only_rin:
        holding_current = calculate_holding_current(cell, protocol_config)
        threshold_current = calculate_threshold_current(cell, protocol_config, holding_current)
    rmp, rin = calculate_rmp_and_rin(cell, protocol_config)

    cell.delete()

    results = {"resting_potential": rmp, "input_resistance": rin}
    if not only_rin:
        results["holding_current"] = holding_current
        results["threshold_current"] = threshold_current
    return results


def _isolated_current_evaluation(*args, **kwargs):
    """Isolate current evaluation for full safety."""
    timeout = kwargs.pop("timeout", None)
    res = isolate(_current_evaluation, timeout=timeout)(*args, **kwargs)
    if res is None:
        res = {
            "resting_potential": 0.0,
            "input_resistance": -1.0,
        }
        if not kwargs.get("only_rin", False):
            res["holding_current"] = 0.0
            res["threshold_current"] = 0.0

    return res


def evaluate_currents(
    morphs_combos_df,
    protocol_config,
    emodels_hoc_dir,
    morphology_path="path",
    resume=False,
    db_url="eval_db.sql",
    parallel_factory=None,
    template_format="v6",
    timeout=1000,
    only_rin=False,
):
    """Compute the threshold and holding currents using bluecellulab."""
    new_columns = [["resting_potential", 0.0], ["input_resistance", 0.0]]
    if not only_rin:
        new_columns += [["holding_current", 0.0], ["threshold_current", 0.0]]

    return evaluate(
        morphs_combos_df,
        _isolated_current_evaluation,
        new_columns=new_columns,
        resume=resume,
        parallel_factory=parallel_factory,
        db_url=db_url,
        func_kwargs={
            "protocol_config": protocol_config,
            "emodels_hoc_dir": emodels_hoc_dir,
            "morphology_path": morphology_path,
            "template_format": template_format,
            "timeout": timeout,
            "only_rin": only_rin,
        },
        progress_bar=not os.environ.get("NO_PROGRESS", False),
    )
