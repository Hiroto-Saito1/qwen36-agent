# Video event search: big_buck_bunny_5min.mp4

- Status: found
- Condition: 蝶が木の幹の近くに見えている
- Strategy: retrieve_verify
- Retrieval model: google/siglip2-base-patch16-224
- Retrieval scan interval: 2 seconds
- Local verification interval: 2 seconds
- Samples verified by VL: 40
- Candidate windows: 6

## Search note

retrieve_verify ranks every scan frame with a non-generative image-text encoder, then verifies only the best candidate windows with the VL model. Very short events can still be missed if retrieval does not rank them near the top. Lower score_threshold, increase max_candidate_windows, or reduce scan_interval_seconds to widen the search.

## Occurrences

### Occurrence 1

- Interval: 216.000s to 218.844s
- Representative time: 218.000s
- Confidence: 1.000
- Positive samples: 2
- Matched sample times: 216.000s, 218.000s
- Caption: 蝶が木の幹の近くに見えている
- Evidence: 蝶が木の幹の近くに位置しており、木の幹が見えている。
- Before frame: evidence/occurrence_001/before.jpg (217.000s)
- Representative frame: evidence/occurrence_001/representative.jpg (218.000s)
- After frame: evidence/occurrence_001/after.jpg (219.000s)
