import subprocess
import sys
import argparse
from importlib_resources import files
import numpy as np
import casadi as ca
import os
import yaml
import traceback

from spline_traj_optm.tests.test_trajectory import get_bspline, get_trajectory_array
from spline_traj_optm.models.trajectory import Trajectory, save_ttl
import spline_traj_optm.models.double_track as dt_dyn
from spline_traj_optm.models.race_track import RaceTrack
import spline_traj_optm.min_time_optm.min_time_optimizer as optm
from spline_traj_optm.models.vehicle import VehicleParams, Vehicle
from spline_traj_optm.simulator.simulator import Simulator

def frange(start, stop, step):
    vals = []
    while start <= stop + 1e-8:
        vals.append(round(start, 8))
        start += step
    return vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regions_yaml', type=str, help='Path to regions.yaml for region encoding', default=None)
    args = parser.parse_args()

    param_file = 'traj_opt_double_track.yaml'
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"{param_file} does not exist.")
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    # Store the original output filename
    original_output = params["output"]

    # Get margin sweep values
    l_cfg = params["model"]["safety_margin_l"]
    r_cfg = params["model"]["safety_margin_r"]
    l_vals = frange(l_cfg["min"], l_cfg["max"], l_cfg["interval"])
    r_vals = frange(r_cfg["min"], r_cfg["max"], r_cfg["interval"])

    failed_combos = []

    for l in l_vals:
        for r in r_vals:
            print(f"\n=== Solving for safety_margin_l={l}, safety_margin_r={r} ===")
            # Set margins
            params["model"]["safety_margin_l"] = l
            params["model"]["safety_margin_r"] = r

            # Always use the original output filename
            out_folder = f"output_l{l:.2f}_r{r:.2f}"
            os.makedirs(out_folder, exist_ok=True)
            out_file = os.path.join(out_folder, original_output)
            params["output"] = out_file

            try:
                # --- (Insert your existing optimization code here, using params as usual) ---
                interval = params["interval"]
                lb = get_trajectory_array(params["left_boundary"])
                rb = get_trajectory_array(params["right_boundary"])
                cl = get_trajectory_array(params["centerline"])
                race_track = RaceTrack(
                    "Test track", lb, rb, cl, s=1.0, interval=interval
                )
                traj_d = race_track.center_d.copy()
                race_track.fill_trajectory_boundaries(traj_d)

                if ("x0" not in params):
                    estimates = params["estimates"]
                    acc_speed_lookup = np.array(estimates["acc_speed_loopup"])
                    dcc_speed_lookup = np.array(estimates["dcc_speed_lookup"])
                    vp = VehicleParams(acc_speed_lookup, dcc_speed_lookup,
                                    estimates["max_lon_acc_mpss"],
                                    estimates["max_lon_dcc_mpss"],
                                    estimates["max_left_acc_mpss"],
                                    estimates["max_right_acc_mpss"],
                                    estimates["max_speed_mps"],
                                    estimates["max_jerk_mpsc"])
                    v = Vehicle(vp)
                    sim = Simulator(v)
                    result = sim.run_simulation(traj_d, False)
                    traj_d = result.trajectory
                else:
                    params["x0"] = ca.DM.from_file(params["x0"], "txt")
                    params["u0"] = ca.DM.from_file(params["u0"], "txt")
                    params["t0"] = ca.DM.from_file(params["t0"], "txt")

                params["N"] = len(traj_d)
                params["traj_d"] = traj_d
                params["race_track"] = race_track

                (X, U, T), (scale_x, scale_u, scale_t), opti = optm.set_up_double_track_problem(params)
                try:
                    sol = opti.solve()
                except Exception as e:
                    print(f"FAILED: l={l}, r={r} - {e}")
                    failed_combos.append((l, r, str(e)))
                    continue

                x = np.array(opti.debug.value(X)) * np.array(scale_x) + np.hstack(
                    [race_track.abscissa[:, np.newaxis], np.zeros((len(traj_d), 5))])
                u = np.array(opti.debug.value(U)) * np.array(scale_u)
                t = np.array(opti.debug.value(T)) * np.array(scale_t)

                print(f"[Optimal lap time: {ca.sum1(t) * scale_t}]")

                ca.DM(traj_d.points[:, :Trajectory.BANK+1]).to_file(os.path.join(out_folder, "ttl_input.txt"), "txt")

                opt_traj_d = traj_d.copy()
                global_pose = race_track.frenet_to_global(x[:, 0].T, x[:, 1].T, x[:, 2].T)
                opt_traj_d[:, 0:2] = global_pose[:, 0:2]
                opt_traj_d[:, Trajectory.YAW] = np.arctan2(
                    np.diff(global_pose[:, 1], prepend=global_pose[-1, 1], axis=0), np.diff(global_pose[:, 0], prepend=global_pose[-1, 0], axis=0)).squeeze()
                opt_traj_d[:, Trajectory.SPEED] = x[:, 5] * np.cos(x[:, 4])
                opt_traj_d[:, Trajectory.YAW_RATE] = x[:, 3]
                opt_traj_d[:, Trajectory.VY] = x[:, 5] * np.sin(x[:, 4])
                race_track.fill_trajectory_boundaries(opt_traj_d)
                opt_traj_d[:, Trajectory.YAW] = global_pose[:, 2].full().squeeze()
                opt_traj_d.fill_distance()
                save_ttl(out_file, opt_traj_d)
                ca.DM(x).to_file(os.path.join(out_folder, "x_optm.txt"), "txt")
                ca.DM(u).to_file(os.path.join(out_folder, "u_optm.txt"), "txt")
                ca.DM(t).to_file(os.path.join(out_folder, "t_optm.txt"), "txt")
                ca.DM(opt_traj_d.points[:, :Trajectory.TIME+1]).to_file(os.path.join(out_folder, "ttl_optm.txt"), "txt")

                # --- Region encoding step ---
                if args.regions_yaml is not None:
                    output_ttl_with_regions = out_file.replace(".csv", "_regions.csv")
                    subprocess.run([
                        sys.executable, "traj_opt_encode_region.py",
                        args.regions_yaml, out_file, output_ttl_with_regions
                    ], check=True)
                    print(f"Regions encoded and saved to {output_ttl_with_regions}")

            except Exception as e:
                print(f"FAILED: l={l}, r={r} - {e}")
                failed_combos.append((l, r, str(e)))

    if failed_combos:
        print("\n=== The following (l, r) combos failed to solve: ===")
        for l, r, msg in failed_combos:
            print(f"  safety_margin_l={l}, safety_margin_r={r} | Reason: {msg}")

if __name__ == "__main__":
    main()