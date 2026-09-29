#!/usr/bin/env python
"""Panel 3 — the contact-solver repair, as an animation of both settings at once.

Same seed, same plan, same actions, same frames; the only difference between the two
runs is the contact time constant. Both sides are real renders rather than a diagram,
because the claim is about what the simulator does and a drawing of it would not be
evidence.

The penetration trace underneath is what makes the still image legible. The ejection
itself lasts a single 2 ms step — one frame the cube is in the gripper, the next it is
off-screen — so the panel would otherwise show a cube and then no cube with nothing to
explain it.

The trace also corrects the intuitive reading of the fix. Softening the contact does
**not** reduce penetration: the shipped setting reaches 18 mm where the old one reached
10 mm before it ejected. Both are an order of magnitude past the 1 mm `solimp` width at
which stiffness saturates. What changes is how that penetration is resolved — gently
over a longer time constant, rather than with the largest impulse the constraint can
produce. Penetration was never the fault; stiffness was.

Sampling is uneven on purpose: every fourth step through the approach, every step in a
window around the divergence. Uniform sampling renders the event in one frame of ninety.

    conda run -n simple_bev_vldrive python -m pipeline.sim.figures.fig3_solver
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: MuJoCo's `solimp` width for these geoms, in metres. Beyond it the solver is at full
#: stiffness, so a penetration several times this deep is resolved with the largest
#: impulse the constraint can produce.
SOLIMP_WIDTH = 0.001


#: The two configurations, reproduced exactly rather than approximated. "before" is not
#: "everything at 0.006" — the free objects were set to 0.006 while the table and bowl
#: kept MuJoCo's 0.02 default, and it is that PAIRING that ejects the cube. Setting the
#: whole arena to 0.006 does not reproduce the bug, which is how the first attempt at
#: this figure rendered an episode that quietly worked.
FREE = ("cube", "ball")
PADS = ("finger_left", "finger_right")


def configure(model, mode: str) -> None:
    import mujoco

    for i in range(model.ngeom):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if n in PADS:
            model.geom_solref[i, 0] = 0.006      # load-bearing for the grasp, unchanged
        elif mode == "before":
            model.geom_solref[i, 0] = 0.006 if n in FREE else 0.02
        else:
            model.geom_solref[i, 0] = 0.03


def capture(mode: str, seed: int, *, width: int, height: int, steps: int = 360):
    """One episode at a given contact configuration, with everything the panel needs."""
    import mujoco

    from pipeline.sim import affordance
    from pipeline.sim.env import PickPlaceEnv
    from pipeline.sim.policy import ScriptedPickPlace

    env = PickPlaceEnv(seed=0, randomise=True, cameras=("front",),
                       width=width, height=height, max_steps=steps)
    configure(env.model, mode)
    # The env carries its own randomisation stream, so episode `seed` only exists after
    # that many resets. See sim/README.md, "Traps".
    for _ in range(seed):
        env.reset()
    rng = np.random.default_rng(seed)
    obs = env.reset()
    plan = affordance.random_plan(env.object_extent("a small dark cube"),
                                  env.object_extent("an orange plastic bowl"), rng)
    pol = ScriptedPickPlace(rng=rng, **plan.as_kwargs())
    cube_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    cube_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    dofadr = env.model.body_dofadr[cube_bid]

    frames, pen, speed, pos = [], [], [], []
    while not env.done():
        obs = env.step(pol.act(obs.tip, obs.objects))
        depth = 0.0
        for c in env.data.contact[:env.data.ncon]:
            if cube_gid in (c.geom1, c.geom2):
                depth = max(depth, -float(c.dist))
        pen.append(depth)
        speed.append(float(np.linalg.norm(env.data.qvel[dofadr:dofadr + 3])))
        pos.append(env.body_pos("cube").copy())
        frames.append(env.render("front")["rgb"].copy())
    env.close()
    return (frames, np.asarray(pen), np.asarray(speed), np.asarray(pos),
            bool(obs.success))


def sample_steps(n: int, event: int, *, coarse: int = 4, window: int = 14) -> list[int]:
    """Every `coarse` steps, but every step near `event`."""
    keep = set(range(0, n, coarse))
    keep |= set(range(max(0, event - window), min(n, event + window)))
    return sorted(keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=89,
                    help="an episode that blows up at the old setting")
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--fps", type=int, default=18)
    #: 840 px wide at this dpi, which is the size a README renders a GIF at. The still
    #: is re-saved at twice this for the detail a 128-colour GIF cannot hold.
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("pipeline/sim/assets"))
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from pipeline.sim.figures import theme

    old = capture("before", a.seed, width=a.width, height=a.height)
    new = capture("after", a.seed, width=a.width, height=a.height)
    of, op, os_, opos, osucc = old
    nf, npn, ns, npos, nsucc = new
    n = min(len(of), len(nf))
    if not (os_ > 5.0).any():
        raise SystemExit(
            f"seed {a.seed} does not eject under the 'before' configuration "
            f"(peak |v| {os_.max():.2f} m/s) — pick a seed that does, or the panel "
            "illustrates a bug that did not happen")
    event = int(np.argmax(os_ > 5.0))
    print(f"[fig3] ejection at step {event}: |v| {os_[event-1]:.2f} -> "
          f"{os_[event]:.1f} m/s;  peak penetration before it "
          f"{op[:event].max()*1000:.1f} mm;  right side peak "
          f"{npn.max()*1000:.1f} mm, success={nsucc}")

    # Everything after the ejection is the cube bouncing off the floor plane metres
    # away, which is real but not the subject. The window ends shortly after the event.
    tail = min(n, event + 26)
    steps = [s for s in sample_steps(tail, event) if s < tail]

    def rollmax(v, k=3):
        """Contacts alternate on and off between steps, so the raw trace oscillates to
        zero every other frame and reads as noise. A 3-step running maximum keeps the
        depth actually reached without inventing any."""
        return np.array([v[max(0, i - k + 1):i + 1].max() for i in range(len(v))])

    theme.use()
    matplotlib.rcParams["figure.dpi"] = a.dpi
    fig = plt.figure(figsize=(8.6, 6.2), dpi=a.dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.40, wspace=0.22)
    axl, axr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axp, axv = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    ims = []
    for ax, frames, label, colour, note in (
            # MuJoCo's solref is (timeconst, dampratio); the XML reads "0.006 1".
            # Only the time constant differs between these two runs, so the damping
            # ratio is stated once in the caption rather than repeated in both labels,
            # where it would read as part of the number being compared.
            (axl, of, "solref  0.006", theme.BAD,
             "free objects stiffer than the 0.02 table"),
            (axr, nf, "solref  0.03", theme.GOOD,
             "shipped, arena-wide · pads keep 0.006")):
        ims.append(ax.imshow(frames[0]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(colour); s.set_linewidth(1.8)
        ax.set_title(label, color=colour, fontfamily="monospace", fontsize=11.5,
                     pad=5)
        ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.2, color=theme.MUTED)

    vl = axl.text(0.035, 0.95, "", transform=axl.transAxes, va="top", fontsize=11,
                  color=theme.MUTED, fontweight="bold", fontfamily="monospace")
    vr = axr.text(0.035, 0.95, "", transform=axr.transAxes, va="top", fontsize=11,
                  color=theme.MUTED, fontweight="bold", fontfamily="monospace")
    gone = axl.text(0.5, 0.5, "", transform=axl.transAxes, ha="center", va="center",
                    fontsize=12, color=theme.BAD, fontweight="bold")

    x = np.arange(tail)
    op_s, np_s = rollmax(op[:tail]) * 1000, rollmax(npn[:tail]) * 1000
    axp.axhspan(0, SOLIMP_WIDTH * 1000, color=theme.FAINT, alpha=0.4, lw=0)
    axp.axhline(SOLIMP_WIDTH * 1000, color=theme.MUTED, lw=1.0, ls=(0, (4, 3)))
    axp.text(0.99, 0.965, "shaded: solimp width, 1 mm", transform=axp.transAxes,
             ha="right", va="top", fontsize=7.8, color=theme.MUTED)
    axp.plot(x, op_s, color=theme.BAD, lw=1.6)
    axp.plot(x, np_s, color=theme.GOOD, lw=1.6)
    cp = axp.axvline(0, color=theme.INK, lw=1.0, alpha=0.7)
    axp.set_xlim(0, tail - 1)
    axp.set_ylim(0, max(op_s.max(), np_s.max()) * 1.18)
    axp.set_xlabel("simulation step", labelpad=2)
    axp.set_ylabel("cube penetration (mm)")
    axp.set_title("both runs sit far past the width", fontsize=9,
                  color=theme.MUTED, fontweight="normal", pad=4)
    axp.grid(alpha=0.2, lw=0.6)

    axv.semilogy(x, np.maximum(os_[:tail], 1e-3), color=theme.BAD, lw=1.6)
    axv.semilogy(x, np.maximum(ns[:tail], 1e-3), color=theme.GOOD, lw=1.6)
    cv = axv.axvline(0, color=theme.INK, lw=1.0, alpha=0.7)
    axv.set_xlim(0, tail - 1)
    axv.set_xlabel("simulation step", labelpad=2)
    axv.set_ylabel("cube speed (m/s)")
    axv.set_title("and only one of them ejects", fontsize=9, color=theme.MUTED,
                  fontweight="normal", pad=4)
    axv.grid(alpha=0.2, lw=0.6, which="both")
    axv.annotate(f"{os_[event-1]:.2f} → {os_[event]:.0f} m/s\nin one 2 ms step",
                 xy=(event, os_[event]), xytext=(max(2, event - 44), 6.0),
                 fontsize=8.4, color=theme.BAD, va="center",
                 arrowprops=dict(arrowstyle="->", color=theme.BAD, lw=1.0))

    def draw(k):
        s = steps[k]
        ims[0].set_data(of[s])
        ims[1].set_data(nf[s])
        cp.set_xdata([s, s])
        cv.set_xdata([s, s])
        vl.set_text(f"|v| {os_[s]:7.2f} m/s")
        vr.set_text(f"|v| {ns[s]:7.2f} m/s")
        vl.set_color(theme.BAD if os_[s] > 5 else theme.MUTED)
        gone.set_text("cube ejected" if s >= event else "")
        return ims + [cp, cv, vl, vr, gone]

    a.out.mkdir(parents=True, exist_ok=True)
    fig.suptitle("The cube was not launched by the grasp — "
                 "it was launched by the contact solver",
                 fontsize=12.5, fontweight="semibold", y=1.0)
    fig.text(0.5, -0.055,
             "blow-ups over 2000 plan-randomised episodes  5.5% → 0.5%   ·   "
             "task success 75.5% → 77.5%, so this is not a stability-for-fidelity "
             "trade\n"
             "the fix does not reduce penetration — the shipped setting reaches "
             "deeper — it makes the same depth resolve without an impulse\n"
             "(solref is (timeconst, dampratio); only the time constant differs "
             "here, both runs use a damping ratio of 1)",
             ha="center", va="top", fontsize=8.6, color=theme.INK, linespacing=1.6)
    fig.subplots_adjust(top=0.90, bottom=0.14)
    anim = FuncAnimation(fig, draw, frames=len(steps), interval=1000 / a.fps,
                         blit=False)
    gif = a.out / "fig3_solver.gif"
    anim.save(gif, writer=PillowWriter(fps=a.fps))
    # Requantise to 128 colours without dithering: a README GIF of flat UI panels and
    # a line chart has nothing that benefits from dither, and it roughly halves the file.
    from PIL import Image
    im = Image.open(gif)
    frames_q, durations = [], []
    try:
        while True:
            frames_q.append(im.convert("RGB").quantize(colors=128,
                                                       dither=Image.Dither.NONE))
            durations.append(im.info.get("duration", 1000 // a.fps))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    frames_q[0].save(gif, save_all=True, append_images=frames_q[1:], loop=0,
                     duration=durations, optimize=True)
    print(f"[fig3] -> {gif}  ({gif.stat().st_size/1e6:.2f} MB, {len(steps)} frames, "
          f"{fig.get_size_inches()[0]*a.dpi:.0f} px wide)")

    try:
        from matplotlib.animation import FFMpegWriter
        mp4 = a.out / "fig3_solver.mp4"
        anim.save(mp4, writer=FFMpegWriter(fps=a.fps, bitrate=2600))
        print(f"[fig3] -> {mp4}  ({mp4.stat().st_size/1e6:.2f} MB)")
    except Exception as e:                       # ffmpeg is not required to build this
        print(f"[fig3] no mp4: {e}")

    draw(steps.index(min(steps, key=lambda s: abs(s - (event + 3)))))
    png = a.out / "fig3_solver.png"
    fig.savefig(png, dpi=a.dpi * 2, bbox_inches="tight", pad_inches=0.22)
    print(f"[fig3] -> {png}  (still, at the ejection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
