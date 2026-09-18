# Research evidence audit API

Fictional, offline teaching utilities. No platform credentials or integration.

Return five fresh fictional paired rows (seed:int, a/b:float error).

    Lower error is better. Deterministic fixture, no random state or downloads.
    Each seed denotes a matched condition, not an independent real-world dataset.
    

Summarize matched rows using d=a-b, so positive values favor B.

    Require >=2 unique integer seeds and finite nonnegative numerical errors.
    Return n, mean_difference, sample_sd, wins_b and differences. No CI/p-value
    or population generalization is inferred. Raise ValueError on malformed data.
    Does not mutate rows. Example: paired_summary(synthetic_runs())[\"n\"] == 5.
    

Return fresh fictional metadata and rows; identifiers are teaching labels.

    No external artifacts are asserted to exist. Use this to exercise the audit,
    not as provenance for a real experiment. No arguments or side effects.
    

Check required nonempty string fields and summarize record['runs'].

    Returns metadata_complete, missing_fields, comparison, limitations. Structural
    completeness does not verify the declared values. Raises ValueError for a
    non-object input or invalid paired rows. Never modifies the input.
    

UI callback: parse JSON text and return audit or a readable error object.

    Handles malformed JSON and invalid records without pretending success.
    No credentials, platform calls, file writes or network operations.
    

Construct (do not launch) a Gradio Blocks app for fictional JSON records.

    Requires Gradio. Returns a Blocks object; caller decides whether/where to
    launch. Includes complete and incomplete example records. No external access.
    
