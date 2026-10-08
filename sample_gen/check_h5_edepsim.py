import h5py
import numpy as np
import os


EDEP_SIM_PREFIX = "/opt/generators/edep-sim/install"
EDEP_SIM_INCLUDE = f"{EDEP_SIM_PREFIX}/include"
EDEP_SIM_LIB = f"{EDEP_SIM_PREFIX}/lib"
os.environ["LD_LIBRARY_PATH"] = "/opt/generators/edep-sim/install/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ROOTSYS"] = os.environ.get("ROOTSYS", "")

import ROOT
ROOT.gInterpreter.AddIncludePath(EDEP_SIM_INCLUDE)
ROOT.gSystem.AddDynamicPath(EDEP_SIM_LIB)
ROOT.gInterpreter.Declare('#include "EDepSim/TG4Event.h"')
ROOT.gSystem.Load("libedepsim_io.so")


def check_entries():
    ndlar_h5_file = '/pscratch/sd/e/ehinkle/nd_ana/sheep_single_shower/NDLAR_ELECTRON_SAMPLES/HDF5/electron_NDLAr_10MeVto15GeV_1008TEST.0000538.LARCV2HDF5.hdf5'
    ndlar_edepsim_file = '/pscratch/sd/e/ehinkle/nd_ana/sheep_single_shower/NDLAR_ELECTRON_SAMPLES/EDEPSIM/electron_NDLAr_10MeVto15GeV_1008TEST.0000538.EDEPSIM.root'

    # Check the contents of the HDF5 file
    with h5py.File(ndlar_h5_file, 'r') as f:
        print("Keys in HDF5 file:", list(f.keys()))
        print("Number of events:", f.attrs['n_events'])
        for i in range(min(5, f.attrs['n_events'])):
            vx_start = int(f['voxels_offsets'][i])
            vx_end = int(f['voxels_offsets'][i + 1])
            event_vx = f['voxels_flat'][vx_start:vx_end]
            ke = f['ke_initial'][i]
            print(f"Event {i}: Initial KE = {ke}, Number of voxels = {len(event_vx)}, Sum of energy = {np.sum(event_vx[:, 3])}")


    # Make sure the EDepSim dictionary library is loaded before opening the file.
    ROOT.gSystem.Load("libedepsim_io.so")

    print("ROOT version:", ROOT.gROOT.GetVersion())
    print("TG4Event class:", ROOT.gROOT.GetClass("TG4Event"))
    cls = ROOT.gROOT.GetClass("TG4Event")
    if cls:
        print("Decl file:", cls.GetDeclFileName())
        print("Shared libs:", cls.GetSharedLibs())

    root_file = ROOT.TFile.Open(ndlar_edepsim_file)
    tree = root_file.Get("EDepSimEvents")

    print("Number of entries:", tree.GetEntries())
    entries = tree.GetEntries()

    '''for i in range(min(5, tree.GetEntries())):
        nb = tree.GetEntry(i)
        print(f"\nEntry {i}, bytes read = {nb}")

        event = tree.Event

        for container_name, hit_segments in event.SegmentDetectors:
            energies = np.array([hit_segment.GetEnergyDeposit() for hit_segment in hit_segments])
            print(
                f"  Container: {container_name}, "
                f"Number of segments: {len(hit_segments)}, "
                f"Total energy deposit: {energies.sum()}"
            )

        for trajectory in event.Trajectories:
            if trajectory.GetParentId() != -1:
                continue

            start_pt = trajectory.Points[0]
            mass = trajectory.GetInitialMomentum().M()
            p_start = np.array([
                start_pt.GetMomentum().X(),
                start_pt.GetMomentum().Y(),
                start_pt.GetMomentum().Z(),
            ])
            start_energy = np.sqrt(np.sum(p_start**2) + mass**2)
            print("  Start energy:", start_energy)'''

    '''# Check the original EDEPSIM file using ROOT
    root_file = ROOT.TFile.Open(ndlar_edepsim_file)
    #root_file.ls()
    tree = root_file.Get("EDepSimEvents")
    print('Number of entries', tree.GetEntries())
    entries = tree.GetEntries()'''

    # Check the first few entries in the EDEPSIM file
    for i in range(min(5, entries)):
        nb = tree.GetEntry(i)
        print(f"GetEntry return: {nb}")
        print(f"Event {i}:")
        event = tree.Event
        for containerName, hitSegments in event.SegmentDetectors:
            segment = np.empty(len(hitSegments))
            for j, hitSegment in enumerate(hitSegments):
                segment[j] = hitSegment.GetEnergyDeposit()
            total_energy = np.sum(segment)
            print(f"  Container: {containerName}, Number of segments: {len(hitSegments)}, Total energy deposit: {total_energy}")
        for iTraj, trajectory in enumerate(event.Trajectories):
            if trajectory.GetParentId() == -1:
                start_pt= trajectory.Points[0]
                mass = trajectory.GetInitialMomentum().M()
                p_start = (start_pt.GetMomentum().X(), start_pt.GetMomentum().Y(), start_pt.GetMomentum().Z())
                start_energy = np.sqrt(np.sum(np.square(p_start)) + mass**2)
                print(f"Start energy: {start_energy}")
            else:
                continue 

if __name__ == '__main__':
    check_entries()