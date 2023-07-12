import itertools
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import yaml
from datareuse import Reuse
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal import cspline2d
from scipy.spatial import ConvexHull

from emodel_generalisation.model.access_point import AccessPoint
from emodel_generalisation.model.evaluation import evaluate_rho
from emodel_generalisation.model.evaluation import evaluate_rho_axon
from emodel_generalisation.model.evaluation import feature_evaluation
from emodel_generalisation.utils import FEATURE_FILTER
from emodel_generalisation.utils import get_scores


def extract_boundaries(df, thresh=5.0):
    xs = {}
    ys = {}
    plot_data = {}
    for name in sorted(df["name"].unique()):
        _df = df[df["name"] == name]
        plot_df = _df.pivot(index="ais_scaler", columns="soma_scaler", values="score")
        f = sc.interpolate.interp2d(
            plot_df.columns.to_list(), plot_df.index.to_list(), plot_df.to_numpy()
        )
        x = np.linspace(plot_df.columns[0], plot_df.columns[-1], 1000)
        y = np.linspace(plot_df.index[0], plot_df.index[-1], 1000)
        z = f(x, y)

        z = plot_df.to_numpy()
        x = plot_df.columns
        y = plot_df.index

        _x, _y = np.meshgrid(x, y)

        _x = _x[abs(z) < thresh]
        _y = _y[abs(z) < thresh]

        points = np.array([_x, _y]).T
        hull = ConvexHull(points)
        _x, _y = points[hull.vertices, 0], points[hull.vertices, 1]
        _x = np.append(_x, _x[0])
        _y = np.append(_y, _y[0])

        xs[name] = _x
        ys[name] = _y
        plot_data[name] = z
    return xs, ys, plot_data


def plot_rho_scan(df, xs, ys):
    model_rho = 9.41
    model_rho_axon = 255.32

    c = {"large": "b", "small": "g", "exemplar": "m", "mean": "r"}
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    for name, ax in zip(xs, axs.flatten()):
        _df = df[df["name"] == name]
        plot_df = _df.pivot(index="ais_scaler", columns="soma_scaler", values="rho")
        fr = sc.interpolate.interp2d(
            plot_df.columns.to_list(), plot_df.index.to_list(), plot_df.to_numpy()
        )

        plot_df = _df.pivot(index="ais_scaler", columns="soma_scaler", values="rho_axon")
        fa = sc.interpolate.interp2d(
            plot_df.columns.to_list(), plot_df.index.to_list(), plot_df.to_numpy()
        )
        ax.scatter(fa(1.0, 1.0), fr(1.0, 1.0), c=c[name])

        xx = xs[name][0]
        yy = ys[name][0]
        for x, y in zip(xx, yy):
            zr = []
            for _x, _y in zip(x, y):
                zr.append(fr(_x, _y)[0])

            za = []
            for _x, _y in zip(x, y):
                za.append(fa(_x, _y)[0])

            ax.plot(za, zr, "--", lw=1, c=c[name])
        xx = xs[name][1]
        yy = ys[name][1]
        for x, y in zip(xx, yy):
            zr = []
            for _x, _y in zip(x, y):
                zr.append(fr(_x, _y)[0])

            za = []
            for _x, _y in zip(x, y):
                za.append(fa(_x, _y)[0])

            ax.plot(
                za,
                zr,
                "-",
                lw=1,
                # label=name,
                path_effects=[
                    patheffects.withTickedStroke(spacing=8, length=1.0, linewidth=0.5, angle=-135)
                ],
                c=c[name],
            )
        ax.scatter(model_rho_axon, model_rho, marker="+", color="r")
        ax.set_xlabel(r"$\rho_\mathrm{axon}$")
        ax.set_ylabel(r"$\rho$")
        ax.set_title(name)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 25)
    plt.tight_layout()
    plt.savefig("rho_line.pdf")
    plt.close()


def scan(emodel, paths, exemplar_data, access_point):
    ais_scales = np.linspace(0.5, 3.0, 20)
    soma_scales = np.linspace(0.3, 3.0, 20)

    dfs = []
    for path in paths:
        _df = pd.DataFrame()

        for i, (a, s) in enumerate(itertools.product(ais_scales, soma_scales)):
            _df.loc[i, "ais_scaler"] = a
            _df.loc[i, "soma_scaler"] = s

        _df["path"] = path
        _df["name"] = Path(path).stem
        _df["ais_model"] = json.dumps(exemplar_data["ais"])
        _df["soma_model"] = json.dumps(exemplar_data["soma"])
        _df["emodel"] = emodel
        dfs.append(_df)

    df = pd.concat(dfs).reset_index(drop=True)
    print("computing features")
    df = feature_evaluation(
        df,
        access_point,
        parallel_factory="multiprocessing",
        timeout=100000,
        trace_data_path="traces",
    )
    print("computing rho_axon")
    df = evaluate_rho_axon(df, access_point, parallel_factory="multiprocessing")

    print("computing rho")
    return evaluate_rho(df, access_point, parallel_factory="multiprocessing")


if __name__ == "__main__":
    path = Path("../cadpyr_l5")
    exemplar_data = yaml.safe_load(open(path / "out/cADpyr_L5/exemplar_models.yaml"))
    emodel = "cADpyr_L5_8825"

    mtype = "L5_TPC:A"
    access_point = AccessPoint(
        emodel_dir=path / "out/cADpyr_L5/configs",
        final_path=path / "out/cADpyr_L5/final.json",
        with_seeds=True,
        legacy_dir_structure=True,
    )
    exemplar_data["paths"][mtype] = str(path / exemplar_data["paths"][mtype])

    morph_map = {
        "rat_P16_S1_RH3_20140129": "large",
        "C270999B-P3": "small",
        "vd100617_idB": "mean",
        "vd110530_idC": "exemplar",
    }

    morphs = ["rat_P16_S1_RH3_20140129", "C270999B-P3", "vd100617_idB", "vd110530_idC"]

    morph_df = pd.read_csv(path / "out/cADpyr_L5/rediametrized_combo_df.csv")
    morph_df = morph_df[morph_df.mtype == "L5_TPC:A"]
    morph_df = morph_df[["name", "path"]].set_index("name")
    morph_df["path"] = str(path) + "/" + morph_df["path"]
    paths = morph_df.loc[morphs, "path"]

    with Reuse("../cadpyr_l5/adaptation_data/scan.csv") as reuse:
        df = reuse(scan, emodel, paths, exemplar_data, access_point)

    for gid in df.index:
        df.loc[gid, "name"] = morph_map[df.loc[gid, "name"]]

    clip = 8
    df = get_scores(df)
    for gid in df.index:
        _s = 0
        try:
            for f, s in df.loc[gid, "scores"].items():
                if not any(f.startswith(_f) for _f in FEATURE_FILTER):
                    _s = max(_s, np.clip(s, 0, clip))
        except AttributeError:
            _s = clip

        df.loc[gid, "score"] = _s

    c = {"large": "b", "small": "g", "exemplar": "m", "mean": "r"}
    xs = {}
    ys = {}
    fig, axs = plt.subplots(2, 2, figsize=(6, 4.5))
    for name, ax in zip(df.name.unique(), axs.flatten()):
        _df = df[df.name == name]
        pivot_df = _df.pivot(index="ais_scaler", columns="soma_scaler", values="score")
        m = pivot_df.to_numpy()

        _id = np.argwhere(m == m.min().min())[0]
        best_soma_scaler = pivot_df.columns[_id[1]]
        best_ais_scaler = pivot_df.index[_id[0]]

        hshift = 0.5 * abs(pivot_df.columns[0] - pivot_df.columns[1])
        vshift = 0.5 * abs(pivot_df.index[0] - pivot_df.index[1])
        _df = df[df["name"] == name]
        plot_df = _df.pivot(index="ais_scaler", columns="soma_scaler", values="score")

        im = ax.imshow(
            plot_df.T,
            cmap="Greys",
            extent=[
                pivot_df.index[0] - vshift,
                pivot_df.index[-1] + vshift,
                pivot_df.columns[0] - hshift,
                pivot_df.columns[-1] + hshift,
            ],
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im, shrink=0.8, label="cost", ax=ax)

        plot_df.loc[:, :] = cspline2d(plot_df.to_numpy(), 1.0, 0.1)
        X, Y = np.meshgrid(pivot_df.index, pivot_df.columns)
        ctrs = ax.contour(X, Y, plot_df.T, levels=[5, 6], colors=c[name])
        ctrs.collections[0].set_linestyle("dashed")

        xs[name] = []
        ys[name] = []

        level = ctrs.collections[0].get_paths()
        xs[name].append([])
        ys[name].append([])
        for lev in level:
            xs[name][0].append(lev.vertices[:, 1])
            ys[name][0].append(lev.vertices[:, 0])

        level = ctrs.collections[1].get_paths()
        xs[name].append([])
        ys[name].append([])
        for lev in level:
            xs[name][1].append(lev.vertices[:, 1])
            ys[name][1].append(lev.vertices[:, 0])

        ax.scatter(1.0, 1.0, c="r", marker="+")

        ax.set_xlabel("AIS scale")
        ax.set_ylabel("soma scale")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig("scan.pdf")
    plot_rho_scan(df, xs, ys)

    for name in df.name.unique():
        _df = df[df.name == name]
        scale = _df.soma_scaler.unique()[5]
        _df = _df[_df.soma_scaler == scale].reset_index(drop=True)
        _df[_df["median_score"] > 100] = np.nan
        _df = _df.dropna(how="all")

        ids = list(_df.index[::3])[:5]
        cmappable = ScalarMappable(
            norm=Normalize(_df.loc[ids[0], "ais_scaler"], _df.loc[ids[-1], "ais_scaler"]),
            cmap="copper",
        )
        colors = plt.cm.copper(np.linspace(0, 1, len(ids)))
        plt.figure(figsize=(4, 5))
        for i, index in enumerate(ids):
            trace_path = _df.loc[index, "trace_data"]
            with open(f"../cadpyr_l5/adaptation_data/{trace_path}", "rb") as f:
                trace = pickle.load(f)[1]
                if "Step_200.soma.v" in trace:
                    response = trace["Step_200.soma.v"]
                    plt.plot(
                        response["time"],
                        response["voltage"] + index * 35,
                        lw=1,
                        c=colors[i],
                    )
        plt.colorbar(cmappable, label="AIS scale", shrink=0.5)
        plt.gca().set_xlim(500, 3000)
        plt.suptitle(scale)
        plt.xlabel("time [ms]")
        plt.ylabel("voltage [mV]")
        plt.tight_layout()
        plt.savefig(f"trace_{name}.pdf")

    for name in df.name.unique():
        _df = df[df.name == name]
        scale = _df.ais_scaler.unique()[6]
        _df = _df[_df.ais_scaler == scale].reset_index(drop=True)
        _df[_df["median_score"] > 100] = np.nan
        _df = _df.dropna(how="all")

        ids = list(_df.index[::3])[:5]
        cmappable = ScalarMappable(
            norm=Normalize(_df.loc[ids[0], "soma_scaler"], _df.loc[ids[-1], "soma_scaler"]),
            cmap="copper",
        )
        colors = plt.cm.copper(np.linspace(0, 1, len(ids)))
        plt.figure(figsize=(4, 5))
        for i, index in enumerate(ids):
            trace_path = _df.loc[index, "trace_data"]

            with open(f"../cadpyr_l5/adaptation_data/{trace_path}", "rb") as f:
                trace = pickle.load(f)[1]
                if "Step_200.soma.v" in trace:
                    response = trace["Step_200.soma.v"]
                    plt.plot(
                        response["time"],
                        response["voltage"] + index * 35,
                        lw=1,
                        c=colors[i],
                    )
        plt.colorbar(cmappable, label="soma scale", shrink=0.5)
        plt.gca().set_xlim(500, 3000)
        plt.suptitle(scale)
        plt.xlabel("time [ms]")
        plt.ylabel("voltage [mV]")
        plt.tight_layout()
        plt.savefig(f"soma_trace_{name}.pdf")
