from enum import NAMED_FLAGS
from pathlib import Path
from vpython import button, canvas, color, label, rate, sphere, vector, wtext
import numpy as np

AU = 149597870.700      # km

# --- simulation settings ----------------------------------------------------
NEPTUNE_PERIOD_YEARS = 164.79         # Neptune's orbital period (years)
STEPS_PER_FRAME = 24                 # advance 10 simulated days per drawn frame
FPS         = 60                      # max frames per second
TRAIL_LEN   = 4000                    # points kept per orbital trail
SCENE_RANGE_AU = 35.0                 # initial zoom (AU)
MOON_ORBIT_SCALE = 70.0              # visual-only magnification of moon-to-host offsets
TEXTURE_DIR = "textures"              # folder with <BodyName>.jpg|png; set to None to disable

# Each satellite is drawn at host_pos + (sat_pos - host_pos) * MOON_ORBIT_SCALE,
# which makes the tiny moon orbits visible without touching the physics.
HOST_BY_SATELLITE = {
    "Moon":     "Earth",
    "Io":       "Jupiter",
    "Europa":   "Jupiter",
    "Ganymede": "Jupiter",
    "Callisto": "Jupiter",
    "Titan":    "Saturn",
    "Triton":   "Neptune",
}

# Visual-only radius multiplier per body. Dynamics use the *real* mass/radius;
# this only fattens the spheres so you can see them.
RADIUS_MULT = {
    "Sun":   35,   "Mercury": 2500, "Venus": 2000, "Earth": 2000,
    "Moon":  4500, "Mars": 2500,    "Jupiter": 500, "Io": 5500,
    "Europa":5500, "Ganymede":5000, "Callisto":5000, "Saturn":550,
    "Titan": 5000, "Uranus":900,    "Neptune":900,  "Triton":5500,
}
COLOURS = {
    "Sun":      color.yellow,
    "Mercury":  vector(0.60, 0.55, 0.50),
    "Venus":    vector(0.95, 0.75, 0.35),
    "Earth":    color.blue,
    "Moon":     vector(0.70, 0.70, 0.75),
    "Mars":     color.red,
    "Jupiter":  vector(0.90, 0.55, 0.25),
    "Io":       vector(1.00, 0.90, 0.30),
    "Europa":   vector(0.85, 0.85, 0.65),
    "Ganymede": color.cyan,
    "Callisto": vector(0.50, 0.65, 0.95),
    "Saturn":   vector(0.95, 0.80, 0.45),
    "Titan":    vector(0.95, 0.55, 0.20),
    "Uranus":   vector(0.40, 1.00, 0.80),
    "Neptune":  vector(0.45, 0.35, 1.00),
    "Triton":   vector(0.75, 0.55, 0.80),
}


def display_position(i, state, origin, HOST_INDEX):
    """
    Map a body's real km position to its drawn position in AU.

    Planets are drawn at their true (Sun-relative) location in AU. Moons are
    drawn at host_visual + MOON_ORBIT_SCALE * (moon - host), so their orbits
    are visible but their *physical* positions in `state` are untouched.
    """


    if i in HOST_INDEX:
        h = HOST_INDEX[i]
        host_au = (state[h, :3] - origin)
        offset_au = (state[i, :3] - state[h, :3]) * MOON_ORBIT_SCALE
        p_au = host_au + offset_au
    else:
        p_au = (state[i, :3] - origin)
    return vector(*p_au)

def visual_radius(body):
    """Exaggerated display radius in AU."""
    mult = RADIUS_MULT.get(body.name, 100)/1.5
    return max(body.radius / AU * mult, 0.014)






def load_textures(bodies, texture_dir):
    _TEXTURES = {}
    if texture_dir is not None:
        cwd = Path.cwd() 
        _tex_root = cwd  / texture_dir
        script_root = Path(__file__).resolve().parent
        repo_root = script_root.parent
        if _tex_root.is_dir():
            for b in bodies:
                for ext in (".jpg", ".jpeg", ".png"):
                    for stem in (b.name, b.name.lower(), b.name.upper()):
                        candidate = _tex_root / f"{stem}{ext}"
                        if candidate.is_file():
                            try:
                                rel_path = candidate.relative_to(repo_root).as_posix()
                            except ValueError:
                                rel_path = candidate.name
                            _TEXTURES[b.name] = rel_path
                            break
                    if b.name in _TEXTURES:
                        break

    print(f"[textures] loaded {len(_TEXTURES)}/{len(bodies)} from {texture_dir!r}: "
        f"{sorted(_TEXTURES)}")
    return _TEXTURES


def initialise_animation(state, method, bodies, sun_idx, texture_dir):
        scene = canvas(
        title=f"16-body solar system — {method.upper()}",
        width=1400, height=850, background=color.black,
        )
        scene.range   = SCENE_RANGE_AU
        scene.forward = vector(-1.15, -0.75, -0.55)   # oblique view
        scene.up      = vector(0, 0, 1)

        

        time_text = wtext(text="")

        #Search in given texture directory for textures for all bodies modelled
        _TEXTURES = load_textures(bodies, texture_dir)

        # HOST_INDEX contains indices of each satellite|host pair in the 16 bodies
        NAME_TO_INDEX = {b.name: i for i, b in enumerate(bodies)}
        HOST_INDEX = {
            NAME_TO_INDEX[sat]: NAME_TO_INDEX[host]
            for sat, host in HOST_BY_SATELLITE.items()
            if sat in NAME_TO_INDEX and host in NAME_TO_INDEX
        }

        # Build spheres, all positioned relative to the Sun so the Sun stays at origin.
        origin  = state[sun_idx, :3].copy()

        spheres = []
        name_labels = []
        for i, b in enumerate(bodies):
            is_moon = b.name in HOST_BY_SATELLITE
            pos = display_position(i, state, origin, HOST_INDEX)
            tex = _TEXTURES.get(b.name)
            kwargs = dict(
                pos        = pos,
                radius     = visual_radius(b),
                color      = color.white if tex is not None else COLOURS.get(b.name, color.white),
                emissive   = (b.name == "Sun"),
                make_trail = not is_moon,                 # moons don't leave trails
                retain     = TRAIL_LEN,
                trail_color= COLOURS.get(b.name, color.white),
            )
            if tex is not None:
                kwargs["texture"] = tex
            s = sphere(**kwargs)
            #Rotate objects to match texture orientations. At slight angle to match approximate axial tilt of earth
            s.rotate(angle=5*np.pi/8, axis=vector(1, 0, 0))

            #Not currently working. Used to add ring to saturn
            # Unknown error with vpython compound() function causes program to hang indefinitely
            # if b.name == "Saturn":
            #     print("Adding Ring")
            #     # s = ring(pos = pos,
            #     #                     radius = 2*visual_radius(b),
            #     #                     thickness = 0.01,
            #     #                     opacity = 0.6,
            #     #                     #texture='textures/saturn_ring.png'
            #     # )
            #     #s = compound([s,saturn_ring], pos=pos)
            #     ring_shape = shapes.circle(radius = 2*visual_radius(b),
            #                                thickness=2*visual_radius(b))
            #     s = extrusion(shape = ring_shape,
            #                             path = [vector(0,0,0),
            #                                     vector(0,0,0.01)],
            #                             texture='textures/saturn_ring.png')
            #     print("Get this far")
            #     #s = compound([s,saturn_ring])
            #     print("Ring added")

            spheres.append(s)

            name_labels.append(label(
                pos=pos, text=b.name,
                xoffset=8, yoffset=8, height=10,
                border=4, box=True, opacity=0.85,
                color=color.black,
                background=color.white,
                visible=False,
            ))
        
        return scene, spheres, name_labels, HOST_INDEX, time_text


