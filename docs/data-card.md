# Data card: Banking77 queue split

- Name and version: banking77-queues, contract 1.0.0
- Source: Banking77, PolyAI, Casanueva et al. 2020, arXiv 2003.04807.
- Licence: CC-BY-4.0. Attribution: PolyAI Ltd. and the Banking77 authors.
- Date range: the public Banking77 release. No bank production messages are included.
- How labels were made: the public 77 intents, mapped by the contract. No model pre-labels were used.
- Agreement: not measured. Banking77 arrived already labelled.
- PII: the public set is not production traffic. Serving redacts card numbers, IBANs, phone numbers, emails, and national-id patterns before scoring.
- Known gaps: English only. No attachments. The unused remainder of the deduplicated train file is not in the 3,000-row training split.
- Owner: ML lead for the dataset build. Business owner has not signed the mapping.

## Counts

- Raw train rows: 10003
- Raw test rows: 3080
- Exact duplicates removed inside train: 7
- Exact train/test overlaps removed: 9
- Near-duplicates removed at cosine 0.90: 692
- Split sizes: {'train': 3000, 'val': 1000, 'test': 2379}
- Validation/test exact overlaps remaining: 0
- Validation/test near-duplicates remaining: 0
- Seed: 42

## Class balance

- Train: {'cards': 689, 'transfers': 664, 'cash': 531, 'fees': 520, 'account': 304, 'fraud': 292}
- Validation: {'cards': 230, 'transfers': 221, 'cash': 177, 'fees': 173, 'account': 101, 'fraud': 98}
- Test: {'cards': 625, 'transfers': 467, 'cash': 411, 'fees': 402, 'account': 239, 'fraud': 235}

## Checksums

- `banking77_train.csv`: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- `banking77_test.csv`: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`
- `train.csv`: `d8f81d9018615f275eedba02ba4e8785374ac4ad181b2959df9ccdd4d6f82263`
- `val.csv`: `e2c64fc240c159f6ffd822cb3a3fe40f76aad9fe7d4759d611151676bfb0ff67`
- `test.csv`: `27159333df8dca17c20b202d7a1d3737f3d96d9381a4f36fa49864caea909426`
