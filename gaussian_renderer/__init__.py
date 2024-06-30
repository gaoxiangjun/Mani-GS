from gaussian_renderer.render import render
from gaussian_renderer.normal import render_normal
from gaussian_renderer.bind import render_bind
from gaussian_renderer.bind_mesh_adapt import render_bind_mesh_adapt
from gaussian_renderer.neilf import render_neilf


render_fn_dict = {
    "render": render,
    "normal": render_normal,
    "bind": render_bind,
    "bind_mesh_adapt": render_bind_mesh_adapt,
    "neilf": render_neilf,
}