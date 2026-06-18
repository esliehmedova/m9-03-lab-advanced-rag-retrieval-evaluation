| Method   | Hit Rate | Faithfulness |
| -------- | -------- | ------------ |
| Baseline | 1.00     | 0.20         |
| Reranked | 1.00     | 0.40         |

Analysis

The reranking upgrade did not improve retrieval hit rate, as both baseline and reranked systems already retrieved the correct passages for all evaluation queries (perfect hit rate of 1.0). This indicates that the dataset is small and relatively easy for dense retrieval.

However, reranking improved faithfulness from 0.20 to 0.40. This suggests that better ordering of retrieved documents helped the model rely more consistently on relevant context when generating answers.

Conclusion

The reranking step did not affect retrieval accuracy because the baseline was already strong on this small dataset. However, it improved answer faithfulness, showing that reranking can still add value by improving context quality even when retrieval performance appears saturated.

Overall, the experiment supports the idea that reranking is more impactful for response quality than for simple hit rate metrics in small knowledge bases.