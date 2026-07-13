#!/usr/bin/env bash
#
# Usage:
#   make_edep_sim_electron_sample.sh -g GEOMETRY -f EDEP_FILENAME -n NEVENTS -m EDEP_MACRO \
#                      -y EDEP2SUPERA_YAML -o OUTDIR -i INDIR -x FILE_INDEX \
#                      [-s STEPS]
#
# STEPS is a comma-separated list chosen from:
#   macro     - build the seeded/randomized macro file
#   edepsim   - run edep-sim
#   larcv     - run edep2supera (EDEPSIM -> LARCV)
#   hdf5      - convert LARCV -> HDF5
#   organize  - clean up old outputs and move new outputs into OUTDIR subdirs
#   all       - (default) run everything, in order
#
# Examples:
#   make_edep_sim_electron_sample.sh ... -s all
#   make_edep_sim_electron_sample.sh ... -s macro,edepsim          # stop after edep-sim
#   make_edep_sim_electron_sample.sh ... -s larcv                  # only the edep2supera step
#   make_edep_sim_electron_sample.sh ... -s hdf5,organize          # resume from LARCV->HDF5
#
# Steps run in a fixed order (macro -> edepsim -> larcv -> hdf5 -> organize)
# regardless of the order you list them in -s. Organize is always run at the end, even if not requested, 
# to ensure outputs are in the right place for next steps.

#set -euo pipefail

STEPS="all"
ran_macro=false
ran_edepsim=false
ran_larcv=false
ran_hdf5=false
 
usage() {
    awk 'NR==1 && /^#!/ { next } /^#/ { sub(/^#/, ""); print; next } { exit }' "$0"
    exit 1
}
 
while getopts "g:f:n:m:y:o:i:x:s:h" opt; do
    case "$opt" in
        g) GEOMETRY=$OPTARG ;;
        f) EDEP_FILENAME=$OPTARG ;;
        n) NEVENTS=$OPTARG ;;
        m) EDEP_MACRO=$OPTARG ;;
        y) EDEP2SUPERA_YAML=$OPTARG ;;
        o) OUTDIR=$OPTARG ;;
        i) INDIR=$OPTARG ;;
        x) FILE_INDEX=$OPTARG ;;
        s) STEPS=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done
 
: "${GEOMETRY:?-g GEOMETRY is required}"
: "${EDEP_FILENAME:?-f EDEP_FILENAME is required}"
: "${NEVENTS:?-n NEVENTS is required}"
: "${EDEP_MACRO:?-m EDEP_MACRO is required}"
: "${EDEP2SUPERA_YAML:?-y EDEP2SUPERA_YAML is required}"
: "${OUTDIR:?-o OUTDIR is required}"
: "${INDIR:?-i INDIR is required}"
: "${FILE_INDEX:?-x FILE_INDEX is required}"

# --- step selection -------------------------------------------------------
 
if [[ "$STEPS" == "all" ]]; then
    STEPS="macro,edepsim,larcv,hdf5,organize"
fi
 
should_run() {
    [[ ",${STEPS}," == *",$1,"* ]]
}

# --- derived filenames -----------------------------------------------------
 
rndSEED=$(( 1000 + FILE_INDEX ))
baseFileName=${EDEP_FILENAME}.${rndSEED}
EDEP_FILE=${baseFileName}.EDEPSIM.root
EDEP_SEEDED_MACRO=${EDEP_MACRO}_${rndSEED}.mac
LARCV_FILE=${baseFileName}.LARCV.root
HDF5_FILE=${baseFileName}.LARCV2HDF5.hdf5
 
if [[ ! -d "${OUTDIR}/MACROS" ]]; then
    mkdir -p "${OUTDIR}/MACROS"
fi    

if [[ ! -d "${OUTDIR}/EDEPSIM" ]]; then
    mkdir -p "${OUTDIR}/EDEPSIM"
fi

if [[ ! -d "${OUTDIR}/LARCV" ]]; then
    mkdir -p "${OUTDIR}/LARCV"
fi   

if [[ ! -d "${OUTDIR}/HDF5" ]]; then
    mkdir -p "${OUTDIR}/HDF5"
fi    
 
cd "${OUTDIR}"

# --- step: macro -------------------------------------------------------
 
step_macro() {
    echo "[macro] Building seeded macro: ${EDEP_SEEDED_MACRO} (seed=${rndSEED})"
 
    # Add random seed to macro file
    sed "1i /edep/random/randomSeed ${rndSEED}" "${INDIR}/${EDEP_MACRO}.mac" > "${EDEP_SEEDED_MACRO}"
 
    # Also randomize e+/e- 
    PARTICLE=$( ((RANDOM % 2)) && echo "e+" || echo "e-" )
    sed "s|/gps/particle e-.*|/gps/particle ${PARTICLE}|" ${EDEP_SEEDED_MACRO} > ${EDEP_SEEDED_MACRO}.tmp
    mv ${EDEP_SEEDED_MACRO}.tmp ${EDEP_SEEDED_MACRO}
}

# --- step: edepsim -------------------------------------------------------
 
step_edepsim() {
    echo "[edepsim] Generating edep-sim sample with ${NEVENTS} events -> ${EDEP_FILE} (seed=${rndSEED})"
 
    if [[ ! -f "${EDEP_SEEDED_MACRO}" ]]; then
        echo "[edepsim] ERROR: ${EDEP_SEEDED_MACRO} not found. Run the 'macro' step first (or -s all)." >&2
        exit 1
    fi
 
    shifter --image=mjkramer/sim2x2:ndlar011 --module=cvmfs -- /bin/bash << EOF1
set +o posix
source /opt/environment
export CPATH=\$EDEPSIM/include/EDepSim:\$CPATH
 
edep-sim \
    -g ${INDIR}/${GEOMETRY} \
    -o ${EDEP_FILE} \
    -e ${NEVENTS} \
    ${EDEP_SEEDED_MACRO}
EOF1
}

# --- larcv / hdf5 (edep2supera + LARCV -> HDF5, single container launch) --
#
# Kept as two selectable steps, but if both are requested together they run
# inside one `shifter` invocation instead of spinning the container up twice.
 
step_larcv_hdf5() {
    local run_larcv=false
    local run_hdf5=false
    if should_run "larcv"; then
        run_larcv=true
        ran_larcv=true
    else
        LARCV_FILE="${OUTDIR}/LARCV/${LARCV_FILE}"
    fi
    if should_run "hdf5"; then
        run_hdf5=true
        ran_hdf5=true
    else
        HDF5_FILE="${OUTDIR}/HDF5/${HDF5_FILE}"
    fi
 
    $run_larcv || $run_hdf5 || return 0
 
    if $run_larcv && [[ ! -f "${EDEP_FILE}" ]]; then
        echo "[larcv] ERROR: ${EDEP_FILE} not found. Run the 'edepsim' step first (or -s all)." >&2
        exit 1
    fi
 
    # hdf5 needs LARCV_FILE either already on disk, or about to be produced
    # by larcv in this same invocation.
    if $run_hdf5 && ! $run_larcv && [[ ! -f "${LARCV_FILE}" ]]; then
        echo "[hdf5] ERROR: ${LARCV_FILE} not found. Run the 'larcv' step first (or -s all)." >&2
        exit 1
    fi
 
    local cmds=""
    if $run_larcv; then
        echo "[larcv] Converting edep-sim -> LARCV (${LARCV_FILE})"
        cmds+="python3 ${INDIR}/edep2supera/bin/run_edep2supera.py -c ${INDIR}/${EDEP2SUPERA_YAML} -o ${LARCV_FILE} ${EDEP_FILE}"$'\n'
    fi
    if $run_hdf5; then
        echo "[hdf5] Converting LARCV -> HDF5 (${HDF5_FILE})"
        cmds+="python3 ${INDIR}/convert_larcv_root2hdf5.py --root_file ${LARCV_FILE} --output_file ${HDF5_FILE}"$'\n'
    fi
 
    module load python
    shifter --image=deeplearnphysics/larcv2:ub2204-cu121-torch251-larndsim bash -c "$cmds"
}
 
# --- step: organize -------------------------------------------------------
 
step_organize() {
    echo "[organize] Moving outputs into ${OUTDIR}/{MACROS,EDEPSIM,LARCV,HDF5}"

    $ran_macro && [[ -f "${EDEP_SEEDED_MACRO}" ]] && mv "${EDEP_SEEDED_MACRO}" "${OUTDIR}/MACROS/"
    $ran_edepsim && [[ -f "${EDEP_FILE}" ]] && mv "${EDEP_FILE}" "${OUTDIR}/EDEPSIM/"
    $ran_larcv && [[ -f "${LARCV_FILE}" ]] && mv "${LARCV_FILE}" "${OUTDIR}/LARCV/"
    $ran_hdf5 && [[ -f "${HDF5_FILE}" ]] && mv "${HDF5_FILE}" "${OUTDIR}/HDF5/"
}

# --- run selected steps, in fixed order -----------------------------------

# Run the steps in a fixed order, regardless of the order they were specified in -s.
# If a step is not requested, check that its expected output file exists before proceeding to the next step.
# Running assumes that all previous files were already produced and were moved to the appropriate OUTDIR subdirectory.
if should_run "macro"; then
   step_macro
   ran_macro=true
else
    MACRO_FILE="${OUTDIR}/MACROS/${EDEP_SEEDED_MACRO}"
    if [[ ! -f "${MACRO_FILE}" ]]; then
        echo "[macro] ERROR: ${MACRO_FILE} not found. Run the 'macro' step first (or -s all)." >&2
        exit 1
    fi
fi

if should_run "edepsim"; then
    step_edepsim
    ran_edepsim=true
else
    EDEP_FILE="${OUTDIR}/EDEPSIM/${EDEP_FILE}"
    if [[ ! -f "${EDEP_FILE}" ]]; then
        echo "[edepsim] ERROR: ${EDEP_FILE} not found. Run the 'edepsim' step first (or -s all)." >&2
        exit 1
    fi
fi

step_larcv_hdf5
#should_run "organize" && 
step_organize # always organize files, even if not requested, to ensure outputs are in the right place for next steps
 
echo "Done. Steps run: ${STEPS}"
cd "${INDIR}"