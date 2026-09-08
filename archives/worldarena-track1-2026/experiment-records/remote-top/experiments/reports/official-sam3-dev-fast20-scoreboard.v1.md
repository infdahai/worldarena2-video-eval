# Official SAM3 dev-fast20 scoreboard

Scope: official evaluator on fixed dev-fast20; this is not the official test-1000 result.

| Candidate | mean(1/d), all 20 | vs S1A125 | Valid episodes | Win rate, all 20 | Black | Strict gate |
|---|---:|---:|---:|---:|---:|---:|
| S1A125 incumbent | 1.9091 | - | 6/20 | - | 0% | baseline |
| adapter-only lr2e-5 step25 | 1.2076 | -36.74% | 7/20 | 25% | 10% | FAIL |
| adapter-only lr2e-5 step50 | 0.0000 | -100.00% | 0/20 | 0% | 5% | FAIL |
| adapter-only lr1e-4 recovery75 | 1.6344 | -14.38% | 7/20 | 25% | 0% | FAIL |
| q/v LoRA-only lr2e-6 step75 | 1.8593 | -2.61% | 8/20 | 20% | 0% | FAIL |
| joint adapter1e-4 lora2e-5 step100 | 26.3512 | +1280.33% | 1/20 | 5% | 10% | FAIL |

Decision: retain S1A125. Next experiment is q-only LoRA with no Adapter update.

Caution: joint step100 all-20 mean is dominated by one valid episode; 19/20 are invalid, so it is rejected despite the large numerical mean.

Evaluator commit: 7b3feee108427bee3380064bb5154970ed7468b5
SAM3 SHA256: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
Preprocessing parity: official default JPEG and staging JPEG are byte-identical.
