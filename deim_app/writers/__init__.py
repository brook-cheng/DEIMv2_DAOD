"""Writer package: pure functions that serialise a ``PredictionCollection``.

Re-exports the three writers so callers can do ``from deim_app.writers import
write_json`` without knowing the module layout. Every writer imports geometry
via ``deim_app.adapters.geometry`` — never ``engine.*`` or
``tools.model_compare.*`` (enforced by the dep-guard test).
"""

from deim_app.writers.dota_writer import write_dota
from deim_app.writers.json_writer import write_json
from deim_app.writers.visualization import write_visualization

__all__ = ["write_dota", "write_json", "write_visualization"]
