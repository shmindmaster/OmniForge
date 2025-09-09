import numpy as np, trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

def outline_to_stl_bytes(outline_mm, base_thickness=0.4, crown=0.3):
    # outline_mm: (N,2) float in millimeters
    pts = np.asarray(outline_mm, dtype=float)
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    # densify boundary to ~128 points
    exterior = np.array(poly.exterior.coords)
    # interpolate along perimeter
    seg_lengths = np.linalg.norm(np.diff(exterior, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum[-1]
    target = np.linspace(0, total, 128, endpoint=False)
    new_pts = []
    for t in target:
        idx = np.searchsorted(cum, t, side='right') - 1
        idx = min(idx, len(exterior)-2)
        local_t = (t - cum[idx]) / (seg_lengths[idx] + 1e-9)
        p = exterior[idx] + local_t * (exterior[idx+1] - exterior[idx])
        new_pts.append(p)
    polygon = Polygon(new_pts)
    # Simple prism extrusion without triangulation engine: create top and bottom faces
    outline = np.array(polygon.exterior.coords)[:-1]
    top_z = base_thickness
    bottom = np.hstack([outline, np.zeros((len(outline),1))])
    top = np.hstack([outline, np.full((len(outline),1), top_z)])
    verts = np.vstack([bottom, top])
    # side faces (quads -> two triangles)
    faces = []
    n = len(outline)
    for i in range(n):
        j = (i+1) % n
        b0, b1 = i, j
        t0, t1 = i + n, j + n
        faces.append([b0, b1, t1])
        faces.append([b0, t1, t0])
    # top face triangulation fan
    for i in range(1, n-1):
        faces.append([n, n+i, n+i+1])
    # bottom face (reverse winding)
    for i in range(1, n-1):
        faces.append([0, i+1, i])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    # crown warp (parabolic) applied to top layer vertices
    c2d = outline.mean(axis=0)
    r = (np.linalg.norm(outline - c2d, axis=1).max() + 1e-6)
    for idx in range(n, 2*n):
        p2d = verts[idx, :2]
        verts[idx, 2] += crown * (1 - (np.linalg.norm(p2d - c2d) / r) ** 2)
    mesh.vertices = verts
    return mesh.export(file_type="stl")
