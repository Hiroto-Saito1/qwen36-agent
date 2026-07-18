# Video event search: big_buck_bunny_5min.mp4

- Status: found
- Condition: 木の幹に矢が刺さっている
- Strategy: retrieve_verify
- Retrieval model: google/siglip2-base-patch16-224
- Retrieval scan interval: 10 seconds
- Local verification interval: 2 seconds
- Samples checked: 28
- VL requests: 35
- Sent images: 57
- Verification image max edge: 960 px
- Candidate windows: 3

## Search note

retrieve_verify ranks every scan frame with a non-generative image-text encoder, then verifies only the best candidate windows with the VL model. Very short events can still be missed if retrieval does not rank them near the top. Lower score_threshold, increase max_candidate_windows, or reduce scan_interval_seconds to widen the search.

## Occurrences

### Occurrence 1

- Interval: 295.761s to 296.761s
- Representative time: 295.761s
- Confidence: 1.000
- Positive samples: 1
- Matched sample times: 295.761s
- Caption: 矢が木の幹に刺さっている
- Evidence: 矢が木の幹に刺さっている様子が確認できる。
- Before frame: evidence/occurrence_001/before.jpg (294.761s)
- Representative frame: evidence/occurrence_001/representative.jpg (295.761s)
- After frame: evidence/occurrence_001/after.jpg (296.761s)
