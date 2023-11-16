"""Functions for morphology modifications in evaluator."""

# Copyright (c) 2022 EPFL-BBP, All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This work is licensed under a Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode
# or send a letter to Creative Commons, 171
# Second Street, Suite 300, San Francisco, California, 94105, USA.

import json
import logging
from functools import partial

import numpy as np

logger = logging.getLogger(__name__)
ZERO = 1e-6


def replace_axon_with_taper(sim=None, icell=None):
    """Replace axon with tappered axon initial segment"""

    L_target = 60  # length of stub axon
    nseg0 = 5  # number of segments for each of the two axon sections

    nseg_total = nseg0 * 2
    chunkSize = L_target / nseg_total

    diams = []

    count = 0
    for section in icell.axonal:
        L = section.L
        nseg = 1 + int(L / chunkSize / 2.0) * 2  # nseg to get diameter
        section.nseg = nseg

        for seg in section:
            count = count + 1
            diams.append(seg.diam)
            if count == nseg_total:
                break
        if count == nseg_total:
            break

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create axon[2]", icell)

    count = 0
    for _, section in enumerate(icell.axon):
        section.nseg = nseg_total // 2
        section.L = L_target / 2

        for seg in section:
            if count >= len(diams):
                break
            seg.diam = diams[count]
            count = count + 1

        icell.axonal.append(sec=section)
        icell.all.append(sec=section)

        if count >= len(diams):
            break
    # childsec.connect(parentsec, parentx, childx)
    icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
    icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = 1.0  # diams[count - 1] if len(diams) > 0 else 1.0
    icell.myelin[0].connect(icell.axon[1], 1.0, 0.0)


def taper_function(distance, strength, taper_scale, terminal_diameter, scale=1.0, dist_max=None):
    """Function to model tappered AIS."""
    value = strength * np.exp(-distance / taper_scale) + terminal_diameter * scale
    if dist_max is not None:
        # pylint: disable=invalid-unary-operand-type
        value -= strength * np.exp(-dist_max / taper_scale)
    return value


def synth_soma(sim=None, icell=None, params=None, scale=1.0):  # pylint: disable=unused-argument
    """Synthesize a simple soma with given scale and parameters."""
    L = 2.0 * params["soma_radius"]
    area = params["soma_surface"]
    nseg = params.get("nseg", 3)
    diam = area / (np.pi * L) * nseg
    soma_sec = icell.soma[0]
    soma_sec.pt3dclear()
    for i in range(nseg):
        soma_sec.pt3dadd(0, scale * i * L / (nseg - 1), 0, diam / nseg)


def synth_axon(sim=None, icell=None, params=None, scale=1.0):
    """Replace axon with tappered axon initial segment.

    Args:
        sim and icell: neuron related arguments
        params (list): fixed parameter for an emodel, should be length, strength, taper_scale and
            terminal_diameter
        scale (float): scale parameter for each cell
    """
    if len(params) == 4:
        for section in icell.axonal:
            sim.neuron.h.delete_section(sec=section)

        sim.neuron.h.execute("create axon[2]", icell)

        nseg_total = 10
        L_target = params[0]
        diameters = taper_function(np.linspace(0, L_target, nseg_total), *params[1:], scale=scale)
        count = 0
        for section in icell.axon:
            section.nseg = nseg_total // 2
            section.L = L_target / 2
            for seg in section:
                seg.diam = diameters[count]
                count += 1

            icell.axonal.append(sec=section)
            icell.all.append(sec=section)
    else:
        nseg0 = 5  # number of segments for each of the two axon sections
        nseg_total = nseg0 * 2
        L_target = 60

        diams = params
        for section in icell.axonal:
            sim.neuron.h.delete_section(sec=section)

        #  new axon array
        sim.neuron.h.execute("create axon[2]", icell)

        count = 0
        for _, section in enumerate(icell.axon):
            section.nseg = nseg_total // 2
            section.L = L_target / 2

            for seg in section:
                if count >= len(diams):
                    break
                seg.diam = scale * diams[count]
                count += 1

            icell.axonal.append(sec=section)
            icell.all.append(sec=section)

            if count >= len(diams):
                break

    icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
    icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = 1.0  # diameters[-1]  # this assigns the value of terminal_diameter
    icell.myelin[0].connect(icell.axon[1], 1.0, 0.0)


def replace_axon_hillock(
    sim=None,
    icell=None,
    length_ais=46,
    delta=5,
    taper_scale=1,
    taper_strength=0,
    myelin_diameter=1,
    nseg_frequency=1,
):
    """Replace axon with tappered axon initial segment.
    Args:
        sim and icell: neuron related arguments
        length: length of ais
        delta: distance of ais to soma (length of hillock)
        taper: taper rater of hillock + ais
        myelin_diameter: diameter of myelin, approximately end of ais (if taper is not too large)
        nseg_frequency: nseg freq as in bpopt of hillock + ais
    """
    # connect basal and apical dendrites to soma at loc=0
    for section in icell.basal:
        if section.parentseg().sec in list(icell.soma):
            sim.neuron.h.disconnect(sec=section)
            section.connect(icell.soma[0], 0.0, sim.neuron.h.section_orientation(sec=section))
    for section in icell.apical:
        if section.parentseg().sec in list(icell.soma):
            sim.neuron.h.disconnect(sec=section)
            section.connect(icell.soma[0], 0.0, sim.neuron.h.section_orientation(sec=section))

    # delete all axonal sections
    for section in icell.axonal:
        sim.neuron.h.section_orientation(sec=section)

    # set hillock section
    sim.neuron.h.execute("create hillock[1]", icell)
    nseg_hillock = 1 + 2 * int(delta / nseg_frequency)
    diameters_hillock = taper_function(
        np.linspace(0, delta, nseg_hillock),
        taper_strength,
        taper_scale,
        myelin_diameter,
        delta + length_ais,
    )
    section = icell.hillock[0]
    section.nseg = nseg_hillock
    section.L = delta
    for i, seg in enumerate(section):
        seg.diam = diameters_hillock[i]
    icell.somatic.append(sec=section)
    icell.all.append(sec=section)

    # set ais section
    sim.neuron.h.execute("create ais[1]", icell)
    nseg_ais = 1 + 2 * int(length_ais / nseg_frequency)
    diameters_ais = taper_function(
        np.linspace(delta, delta + length_ais, nseg_ais),
        taper_strength,
        taper_scale,
        myelin_diameter,
        delta + length_ais,
    )
    section = icell.ais[0]
    section.nseg = nseg_ais
    section.L = length_ais
    for i, seg in enumerate(section):
        seg.diam = diameters_ais[i]
    icell.axon_initial_segment.append(sec=section)
    icell.all.append(sec=section)

    # set myelinated axon
    sim.neuron.h.execute("create myelin[1]", icell)
    section = icell.myelin[0]
    icell.myelinated.append(sec=section)
    icell.all.append(sec=section)
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = myelin_diameter

    # connect soma/hillock/ais/myelin
    icell.hillock[0].connect(icell.soma[0], 1.0, 0.0)
    icell.ais[0].connect(icell.hillock[0], 1.0, 0.0)
    icell.myelin[0].connect(icell.ais[0], 1.0, 0.0)


def remove_soma(sim=None, icell=None):
    """Remove the soma and connect dendrites together at the axon.

    For this to work, we leave the soma connected to the axon,
    and with diameter 1e-6. BluePyOp requires a soma for
    parameter scaling, and NEURON fails is the soma size is =0.
    """
    sec = list(icell.axonal)[0]
    for section in icell.basal:
        if section.parentseg().sec in list(icell.soma):
            sim.neuron.h.disconnect(sec=section)
            section.connect(sec, 0, 0)

    for section in icell.apical:
        if section.parentseg().sec in list(icell.soma):
            sim.neuron.h.disconnect(sec=section)
            section.connect(sec, 0, 0)

    for section in icell.soma:
        section.diam = ZERO
        section.L = ZERO


def isolate_soma(sim=None, icell=None):
    """Remove everything except the soma."""
    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.basal:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.apical:
        sim.neuron.h.delete_section(sec=section)


def remove_axon(sim=None, icell=None):
    """Remove the axon."""
    for section in icell.myelin:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)


def isolate_axon(sim=None, icell=None):
    """Remove everything except the axon."""
    for section in icell.basal:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.apical:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.soma:
        sim.neuron.h.delete_section(sec=section)
    for section in icell.myelin:
        sim.neuron.h.delete_section(sec=section)


def get_synth_modifiers(combo, morph_modifiers=None):
    """Insert synth_axon to start of morph_modifiers if AIS info in combo,
    else use replace_axon_with_taper.
    """
    if morph_modifiers is None:
        morph_modifiers = []

    if (
        "soma_model" in combo
        and isinstance(combo["soma_model"], str)
        and remove_soma not in morph_modifiers
    ):
        morph_modifiers.insert(
            0,
            partial(
                synth_soma,
                params=json.loads(combo["soma_model"]),
                scale=combo.get("soma_scaler", 1),
            ),
        )

    if (
        "ais_model" in combo
        and isinstance(combo["ais_model"], str)
        and remove_axon not in morph_modifiers
    ):
        morph_modifiers.insert(
            0,
            partial(
                synth_axon,
                params=json.loads(combo["ais_model"])["popt"],
                scale=combo.get("ais_scaler", 1),
            ),
        )
    elif replace_axon_with_taper not in morph_modifiers:
        morph_modifiers.insert(0, replace_axon_with_taper)
    return morph_modifiers


def replace_axon_justAIS(sim=None, icell=None, diam=1.0, L_target=45):
    """Adapted from original replace_axon_with_taper, but no taper.

    Also named as version used for optimisation for backward compatibility:
    /gpfs/bbp.cscs.ch/project/proj130/singlecell/optimisation/morph_modifiers.py
    """
    nseg0 = 5  # number of segments for each of the two axon sections
    nseg_total = nseg0 * 2

    for section in icell.axonal:
        sim.neuron.h.delete_section(sec=section)

    #  new axon array
    sim.neuron.h.execute("create axon[2]", icell)

    count = 0
    for section in icell.axon:
        section.nseg = nseg_total // 2
        section.L = L_target / 2

        for seg in section:
            seg.diam = diam
            count += 1

        icell.axonal.append(sec=section)
        icell.all.append(sec=section)

    # childsec.connect(parentsec, parentx, childx)
    icell.axon[0].connect(icell.soma[0], 1.0, 0.0)
    icell.axon[1].connect(icell.axon[0], 1.0, 0.0)

    sim.neuron.h.execute("create myelin[1]", icell)
    icell.myelinated.append(sec=icell.myelin[0])
    icell.all.append(sec=icell.myelin[0])
    icell.myelin[0].nseg = 5
    icell.myelin[0].L = 1000
    icell.myelin[0].diam = diam
    icell.myelin[0].connect(icell.axon[1], 1.0, 0.0)


def get_replace_axon_hoc(params):
    """Get replace_axon hoc string."""
    return """
    proc replace_axon(){ local nSec, count, area, ais_scale, soma_scale localobj diams

        L_target = 60  // length of stub axon
        nseg0 = 5  // number of segments for each of the two axon sections

        nseg_total = nseg0 * 2
        chunkSize = L_target/nseg_total

        nSec = 0
        forsec axonal{nSec = nSec + 1}
        ais_scale = $1
        soma_scale = $2

        access soma[0]
        L0 = 2.0 * %s
        area0 = %s
        n = 3
        diam0 = area0 / (PI * L0) * n

        pt3dclear()
        for i=0,n-1 {
            pt3dadd(0, soma_scale * i * L0 / (n - 1), 0, diam0 / n)
        }


        // get rid of the old axon
        forsec axonal{delete_section()}
        execute1("create axon[2]", CellRef)

        diams = new Vector(nseg_total)
        diams.x[0] = %s
        diams.x[1] = %s
        diams.x[2] = %s
        diams.x[3] = %s
        diams.x[4] = %s
        diams.x[5] = %s
        diams.x[6] = %s
        diams.x[7] = %s
        diams.x[8] = %s
        diams.x[9] = %s

        count = 0

        for i=0,1{
            access axon[i]
            L =  L_target/2
            nseg = nseg_total/2

            for (x) {
                if (x > 0 && x < 1) {
                    diam(x) = diams.x[count] * ais_scale
                    count = count + 1
                }
            }

            all.append()
            axonal.append()
        }

        nSecAxonal = 2
        soma[0] connect axon[0](0), 1
        axon[0] connect axon[1](0), 1

        create myelin[1]
        access myelin{
                L = 1000
                diam = diams.x[count-1]
                nseg = 5
                all.append()
                myelinated.append()
        }
        connect myelin(0), axon[1](1)
    }
    """ % tuple(
        params
    )
