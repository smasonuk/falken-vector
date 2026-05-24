# AlphaFold Service Workflow

The folding service tracks job state from request intake through pipeline execution and structure database publication.

Service notes distinguish monomer, dimer, and ligand workflows because each path has different inputs and result validation.

The job monitor reports whether a prediction is queued, running, completed, or failed before artifacts are copied to the shared output area.

