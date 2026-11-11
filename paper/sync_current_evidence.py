"""Copy selected public run summaries into the self-contained manuscript archive."""
import hashlib
import json
from pathlib import Path
paper = Path(__file__).resolve().parent
names = ['ocelot_4096_20261111.json', 'ranking_ablation_20261111.json',
         'ocelot_generator_continued_20261111.json', 'generated_quantum_diagnostic_20261111.json',
         'batched_pipeline_comparison_20261111.json', 'batched_pipeline_reload_20261111.json',
         'clean_deepmd_native_20261111.json']
records = {}
for name in names:
    path = paper.parent/'docs/experiments'/name
    raw = path.read_text()
    records[name] = {'source': 'docs/experiments/'+name, 'sha256': hashlib.sha256(raw.encode()).hexdigest(), 'json': raw}
(paper/'data/unified_evidence.json').write_text(json.dumps(records, indent=2)+'\n')
